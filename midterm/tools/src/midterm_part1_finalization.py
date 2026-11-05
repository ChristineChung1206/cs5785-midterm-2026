"""Controlled Part 1 feature experiments, submission formatting, and reason provenance."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from midterm_augmentation import logistic_parameters, observed_label_mask, predict_reason_probabilities
from midterm_evaluation import REASONS, evaluate_part1, file_sha256, part1_decisions


def validate_saved_configuration(config, *, parameters, input_columns, input_files, seed, thresholds):
    """Check model identity independently of output paths and source-code fingerprints.

    Historical source hashes and package versions remain in the original record.
    Reusing its frozen model does not rerun the historical training code. Data,
    features, parameters, seed, and decisions must still match the requested run.
    """
    expected = {"hyperparameters": parameters, "input_files": input_files, "seed": seed,
                "thresholds": dict(thresholds)}
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"Existing run has different {field}; choose a new run ID for a different experiment.")
    if config.get("feature_spec", {}).get("input_columns") != list(input_columns):
        raise ValueError("Existing run uses different input columns; choose a new run ID.")
    if config.get("task") != "part1" or config.get("split") != "validation":
        raise ValueError("Expected an existing Part 1 validation run.")


def load_saved_step5_run(directory, *, parameters, input_columns, input_files, seed,
                         thresholds, required_lineage, validation):
    """Verify and load an immutable run without calling any estimator's fit method."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    validate_saved_configuration(config, parameters=parameters, input_columns=input_columns,
                                 input_files=input_files, seed=seed, thresholds=thresholds)
    manifest = json.loads((directory / "artifact_checksums.json").read_text())
    for name, digest in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid relative path in the artifact checksum manifest.")
        path = directory / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"An immutable artifact changed: {path}")
    lineage = json.loads((directory / "lineage.json").read_text())
    for key, value in required_lineage.items():
        if lineage.get(key) != value:
            raise ValueError(f"Existing run has different lineage field {key}.")
    bundle = joblib.load(directory / "model.joblib")
    if bundle.get("step5_parameters") != parameters or bundle["thresholds"] != dict(thresholds):
        raise ValueError("Saved model configuration does not match the requested run.")
    saved = pd.read_csv(directory / "validation_predictions.csv")
    if not saved["ID"].is_unique or set(saved["ID"]) != set(validation["ID"]):
        raise ValueError("Saved validation IDs do not match the fixed validation split.")
    saved = saved.set_index("ID").loc[validation["ID"]]
    probabilities = pd.DataFrame(saved[[f"prob_{r}" for r in REASONS]].to_numpy(), columns=REASONS)
    probabilities.insert(0, "ID", validation["ID"].to_numpy())
    actual = predict_bundle(bundle, validation)
    if not np.allclose(actual, probabilities[list(REASONS)], atol=1e-12, rtol=0):
        raise ValueError("Saved model does not reproduce its recorded validation probabilities.")
    result = evaluate_part1(validation, probabilities, thresholds)
    if not np.array_equal(result.predictions[[f"pred_{r}" for r in REASONS]],
                          saved[[f"pred_{r}" for r in REASONS]]):
        raise ValueError("Saved validation decisions differ from the requested thresholds.")
    metrics = json.loads((directory / "metrics.json").read_text())
    for key, value in result.summary.items():
        if not np.isclose(value, metrics["summary"][key], atol=1e-14, rtol=0):
            raise ValueError(f"Saved metric {key} does not reproduce.")
    return {"bundle": bundle, "probabilities": probabilities, "evaluation": result,
            "lineage": lineage, "config": config}


def keep_or_create_selection(path, proposed):
    """Preserve training provenance while checking all material selection fields."""
    path = Path(path)
    if not path.exists():
        path.write_text(json.dumps(proposed, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return proposed
    saved = json.loads(path.read_text())
    descriptive_fields = {"training_code_sha256", "finalization_module_sha256", "runtime_versions",
                          "selected_model", "output_directory", "report_dir"}
    identity = lambda record: {key: value for key, value in record.items() if key not in descriptive_fields}
    if identity(saved) != identity(proposed):
        raise ValueError("The selected model, settings, data, or candidate results changed; use a new final-output location.")
    return saved


def require_same_inference_model(left, right):
    """Allow provenance-only differences while requiring exactly equal inference state."""
    for key in ["feature_columns", "reason_columns", "thresholds", "step5_parameters"]:
        if left.get(key) != right.get(key):
            raise ValueError(f"Inference model differs in {key}.")
    for name in ["with_mean", "with_std", "n_features_in_"]:
        if getattr(left["scaler"], name) != getattr(right["scaler"], name):
            raise ValueError(f"Scaler setting differs: {name}.")
    for name in ["mean_", "var_", "scale_", "n_samples_seen_", "feature_names_in_"]:
        if not np.array_equal(getattr(left["scaler"], name, None), getattr(right["scaler"], name, None)):
            raise ValueError(f"Scaler state differs: {name}.")
    encoders = [left.get("pathway_encoder"), right.get("pathway_encoder")]
    if (encoders[0] is None) != (encoders[1] is None):
        raise ValueError("Pathway feature blocks differ.")
    if encoders[0] is not None:
        for name in ["vocabulary_", "feature_names_", "min_df", "fit_ids_"]:
            if getattr(encoders[0], name) != getattr(encoders[1], name):
                raise ValueError(f"Pathway encoding differs: {name}.")
    if len(left["classifiers"]) != 5 or len(right["classifiers"]) != 5:
        raise ValueError("Five reason classifiers are required.")
    for a, b in zip(left["classifiers"], right["classifiers"]):
        if type(a) is not type(b) or a.get_params() != b.get_params():
            raise ValueError("Classifier settings differ.")
        for name in ["coef_", "intercept_", "classes_", "n_features_in_"]:
            if not np.array_equal(getattr(a, name), getattr(b, name)):
                raise ValueError(f"Classifier inference state differs: {name}.")
    return True


def semicolon_tokens(value) -> set[str]:
    """Parse pooled categorical values without treating missing values as tokens."""
    if pd.isna(value):
        return set()
    if not isinstance(value, str):
        raise ValueError("Pooled drug fields must be strings or missing values.")
    return {part.strip() for part in value.split(";") if part.strip()}


class BroadPathwayEncoder:
    """Unscaled binary level-0 pathways plus explicit availability indicators.

    Fit the vocabulary using training rows only. The supplied catalog defines the
    external taxonomy; catalog-wide drug/target counts are not predictive features.
    Unknown or infrequent pathways do not create new feature columns at transform.
    """

    def __init__(self, min_df: int = 5):
        if not isinstance(min_df, int) or min_df < 1:
            raise ValueError("min_df must be a positive integer.")
        self.min_df = min_df

    def fit(self, frame: pd.DataFrame, catalog: pd.DataFrame):
        required = {"reactome_id", "pathway_name", "pathway_level"}
        if not required.issubset(catalog) or not catalog["reactome_id"].is_unique:
            raise ValueError("The pathway catalog must contain unique IDs, names, and levels.")
        allowed = catalog.loc[catalog["pathway_level"].eq(0)].set_index("reactome_id")
        counts = Counter(token for value in frame["drug_pathways"]
                         for token in semicolon_tokens(value) if token in allowed.index)
        self.vocabulary_ = sorted(token for token, count in counts.items() if count >= self.min_df)
        if not self.vocabulary_:
            raise ValueError("No broad pathways meet the fixed training-frequency rule.")
        self.training_document_frequency_ = {token: counts[token] for token in self.vocabulary_}
        self.pathway_names_ = allowed.loc[self.vocabulary_, "pathway_name"].to_dict()
        self.feature_names_ = [f"pathway__{token}" for token in self.vocabulary_] + ["has_matched_drug", "has_any_pathway"]
        self.fit_ids_ = frame["ID"].tolist()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "vocabulary_"):
            raise ValueError("Fit the training vocabulary before transforming.")
        tokens = frame["drug_pathways"].map(semicolon_tokens)
        matrix = np.zeros((len(frame), len(self.feature_names_)), dtype=float)
        for j, token in enumerate(self.vocabulary_):
            matrix[:, j] = tokens.map(lambda values: token in values).to_numpy(dtype=float)
        matrix[:, -2] = frame["matched_drugs"].map(semicolon_tokens).map(bool).to_numpy(dtype=float)
        matrix[:, -1] = tokens.map(bool).to_numpy(dtype=float)
        return matrix

    def feature_table(self) -> pd.DataFrame:
        return pd.DataFrame([{"feature": f"pathway__{token}", "reactome_id": token,
                              "pathway_name": self.pathway_names_[token], "pathway_level": 0,
                              "training_document_frequency": self.training_document_frequency_[token]}
                             for token in self.vocabulary_])


def make_features(frame, scaler, embedding_columns, pathway_encoder=None):
    """Keep the inherited embedding scale and append the optional binary block."""
    features = scaler.transform(frame[list(embedding_columns)])
    if pathway_encoder is not None:
        features = np.column_stack([features, pathway_encoder.transform(frame)])
    return features


def fit_classifiers(features, labels, weights, *, C: float, seed: int = 42):
    """Hold all effective baseline settings fixed except the declared C value."""
    labels = np.asarray(labels)
    weights = np.asarray(weights, dtype=float)
    if not observed_label_mask(labels).all():
        raise ValueError("Use observed or recovered binary targets, never -100.")
    if not np.isfinite(C) or C <= 0 or weights.shape != (len(labels),) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("C and all training weights must be finite and positive.")
    estimators = []
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        for j in range(5):
            parameters = {**logistic_parameters(seed), "C": C}
            model = LogisticRegression(**parameters)
            model.fit(features, labels[:, j], sample_weight=weights)
            estimators.append(model)
    if any(issubclass(w.category, ConvergenceWarning) for w in recorded) or any(model.n_iter_.max() >= 500 for model in estimators):
        raise RuntimeError("A classifier failed to converge under the prespecified iteration limit.")
    return estimators, [str(w.message) for w in recorded]


def predict_bundle(bundle, frame):
    features = make_features(frame, bundle["scaler"], bundle["feature_columns"], bundle.get("pathway_encoder"))
    return predict_reason_probabilities(bundle["classifiers"], features)


def select_primary(candidates):
    """Maximize sample-wise F1, retaining the first prespecified candidate on ties."""
    if not candidates:
        raise ValueError("Supply at least one candidate.")
    scores = [row["evaluation"].summary["f1_samples"] for row in candidates]
    if not np.isfinite(scores).all():
        raise ValueError("Selection scores must be finite.")
    return candidates[int(np.argmax(scores))]


def binary_decisions(scores, thresholds, *, empty_set_policy="allow"):
    scores = np.asarray(scores, dtype=float)
    limits = np.array([thresholds[reason] for reason in REASONS], dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 5 or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Provide five finite reason probabilities in [0, 1].")
    if not np.isfinite(limits).all() or ((limits <= 0) | (limits >= 1)).any():
        raise ValueError("Every threshold must lie strictly between zero and one.")
    return part1_decisions(scores, thresholds, empty_set_policy=empty_set_policy)


def reason_strings(decisions):
    decisions = np.asarray(decisions)
    if decisions.ndim != 2 or decisions.shape[1] != 5 or not np.isin(decisions, [0, 1]).all():
        raise ValueError("Reason decisions must be a binary n-by-5 array.")
    return [" ".join(reason for reason, positive in zip(REASONS, row) if positive) for row in decisions]


def build_submission(sample, predictions):
    """Align sample IDs and reject empty sets that Kaggle parses as null."""
    if list(sample.columns) != ["ID", "Predicted"] or set(predictions.columns) != {"ID", *REASONS}:
        raise ValueError("Unexpected sample or prediction columns.")
    for frame in [sample, predictions]:
        if frame["ID"].isna().any() or not frame["ID"].is_unique:
            raise ValueError("Submission IDs must be unique and nonmissing.")
    if set(sample["ID"]) != set(predictions["ID"]):
        raise ValueError("Test and sample IDs must match exactly.")
    aligned = predictions.set_index("ID").loc[sample["ID"], list(REASONS)]
    strings = reason_strings(aligned.to_numpy())
    if any(not value.strip() for value in strings):
        raise ValueError("Empty reason sets cannot be submitted; apply the declared argmax fallback first.")
    return pd.DataFrame({"ID": sample["ID"].to_numpy(), "Predicted": strings})


def build_reason_lookup(frames, scores_by_split, thresholds, model_run_id, *, empty_set_policy="allow"):
    """Prefer observed truth; use model estimates only for masked train and test.

    Training-row predictions are explicitly marked as resubstitution estimates
    from a model trained on their recovered targets. They are not held-out labels.
    """
    tables = []
    for split in ["train", "val", "test"]:
        frame = frames[split]
        scores = np.asarray(scores_by_split[split])
        decisions = binary_decisions(scores, thresholds, empty_set_policy=empty_set_policy)
        raw_empty = binary_decisions(scores, thresholds).sum(axis=1) == 0
        if len(scores) != len(frame):
            raise ValueError("Scores must align with split rows.")
        observed = observed_label_mask(frame[list(REASONS)].to_numpy()) if split != "test" else np.zeros(len(frame), dtype=bool)
        used = decisions.copy()
        if observed.any():
            used[observed] = frame.loc[observed, list(REASONS)].to_numpy(dtype=int)
        source = np.full(len(frame), f"model_prediction_{split}", dtype=object)
        source[observed] = f"observed_{split}"
        scope = np.full(len(frame), "test_inference" if split == "test" else "training_resubstitution_with_pseudo_targets", dtype=object)
        scope[observed] = "observed_label"
        table = pd.DataFrame({"ID": frame["ID"].to_numpy(), "part1_split": split,
            "reason_source": source, "is_observed": observed, "prediction_scope": scope,
            "model_run_id": np.where(observed, None, model_run_id),
            "prediction_is_training_resubstitution": (~observed) & (split == "train")})
        for j, reason in enumerate(REASONS):
            table[reason] = used[:, j]
            table[f"model_score_{reason}"] = np.where(observed, np.nan, scores[:, j])
        if empty_set_policy == "argmax":
            table["empty_set_fallback_applied"] = raw_empty & (~observed)
            table["decision_policy"] = np.where(observed, "observed_label", "threshold_then_argmax_if_empty")
        table["reason_count"] = used.sum(axis=1)
        table["has_assigned_reason"] = used.sum(axis=1) > 0
        table["reasons"] = reason_strings(used)
        tables.append(table)
    lookup = pd.concat(tables, ignore_index=True)
    if lookup["ID"].isna().any() or not lookup["ID"].is_unique:
        raise ValueError("Part 1 split IDs must be unique and disjoint.")
    return lookup


def link_unsuccessful_validation(outcomes, reason_lookup):
    """Keep every failed trial-phase row and distinguish unmatched from empty predictions."""
    failed = outcomes.loc[outcomes["label"].eq(0), ["ID", "trial_id", "phase", "label"]].copy()
    if failed.duplicated(["trial_id", "phase"]).any():
        raise ValueError("A trial must occur at most once within a phase.")
    renamed = reason_lookup.rename(columns={"ID": "part1_id"})
    joined = failed.merge(renamed, left_on="trial_id", right_on="part1_id", how="left", validate="many_to_one", indicator=True)
    joined["matched_part1"] = joined.pop("_merge").eq("both")
    joined["reason_group"] = np.where(~joined["matched_part1"], "no_recorded_stopping_reason",
        np.where(joined["reason_count"].fillna(0).eq(0), "matched_no_predicted_reason", "assigned_reasons"))
    for reason in REASONS:
        joined[reason] = joined[reason].astype("Int64")
    if len(joined) != len(failed):
        raise RuntimeError("A linkage operation changed the failed-row count.")
    return joined
