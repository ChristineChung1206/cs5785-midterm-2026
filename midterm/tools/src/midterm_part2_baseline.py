"""Phase-specific text baselines and provenance-aware failure-reason analysis."""

from __future__ import annotations

from math import sqrt
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from midterm_augmentation import logistic_parameters
from midterm_evaluation import PHASES, REASONS, evaluate_part2, file_sha256
from midterm_part1_finalization import link_unsuccessful_validation

TEXT_COLUMNS = ("criteria", "diseases_cleaned", "drugs")
REASON_GROUPS = (*REASONS, "no_recorded_stopping_reason", "matched_no_predicted_reason")


def build_trial_text(frame: pd.DataFrame) -> pd.Series:
    """Use exactly three text fields; fill missing fields with an empty string.

    Brackets and quotes in the supplied disease/drug list strings are retained
    here and discarded by the vectorizer's word tokenizer. Negations remain.
    Identifiers, outcomes, years, drug-knowledge fields, and reasons are excluded.
    """
    if not set(TEXT_COLUMNS).issubset(frame.columns):
        raise ValueError("Expected criteria, diseases_cleaned, and drugs text fields.")
    return frame[list(TEXT_COLUMNS)].astype("string").fillna("").agg("\n".join, axis=1)


def validate_outcome_rows(frame, *, labels_required=True):
    required = {"ID", "trial_id", "phase", *TEXT_COLUMNS}
    if labels_required:
        required.add("label")
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Expected nonempty trial-phase rows with the declared columns.")
    if frame["ID"].isna().any() or not frame["ID"].is_unique or frame["trial_id"].isna().any():
        raise ValueError("Row IDs must be unique, with nonmissing row and trial IDs.")
    if not frame["phase"].isin(PHASES).all() or frame.duplicated(["trial_id", "phase"]).any():
        raise ValueError("Each trial may occur at most once in each of phases I, II, III.")
    if labels_required:
        if not frame["label"].isin([0, 1]).all():
            raise ValueError("Outcome labels must be binary, with success=1.")
        if frame.groupby("trial_id")["label"].nunique().gt(1).any():
            raise ValueError("Combined-phase copies of one trial must have the same outcome.")


def normalized_json(value):
    """Normalize tuple/list representations before comparing saved parameters."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def fit_phase_baseline(train, *, tfidf_parameters, classifier_parameters, seed=42):
    """Fit one vocabulary and one binary classifier per training phase."""
    validate_outcome_rows(train)
    pipelines, diagnostics, fit_ids, warning_rows = {}, [], {}, []
    for phase in PHASES:
        subset = train.loc[train["phase"].eq(phase)]
        if subset.empty or subset["label"].nunique() != 2:
            raise ValueError(f"Phase {phase} requires training examples of both classes.")
        parameters = {**logistic_parameters(seed), **classifier_parameters, "random_state": seed}
        pipeline = Pipeline([("tfidf", TfidfVectorizer(**tfidf_parameters)),
                             ("classifier", LogisticRegression(**parameters))])
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            pipeline.fit(build_trial_text(subset), subset["label"].to_numpy(dtype=int))
        classifier = pipeline.named_steps["classifier"]
        if any(issubclass(item.category, ConvergenceWarning) for item in recorded) or classifier.n_iter_.max() >= classifier.max_iter:
            raise RuntimeError(f"Phase {phase} failed to converge under the fixed iteration limit.")
        if not np.array_equal(classifier.classes_, [0, 1]):
            raise RuntimeError("The positive class must be success=1.")
        vectorizer = pipeline.named_steps["tfidf"]
        matrix = vectorizer.transform(build_trial_text(subset))
        diagnostics.append({"phase": phase, "training_rows": len(subset),
            "unique_training_trials": subset["trial_id"].nunique(),
            "training_successes": int(subset["label"].sum()),
            "training_failures": int(subset["label"].eq(0).sum()),
            "missing_criteria_rows": int(subset["criteria"].isna().sum()),
            "vocabulary_size": len(vectorizer.vocabulary_), "training_nonzero_values": int(matrix.nnz),
            "zero_training_vectors": int((matrix.getnnz(axis=1) == 0).sum()),
            "classifier_iterations": int(classifier.n_iter_.max()), "converged": True})
        fit_ids[f"tfidf_phase_{phase}"] = subset["trial_id"].tolist()
        fit_ids[f"classifier_phase_{phase}"] = subset["trial_id"].tolist()
        warning_rows.extend({"phase": phase, "warning": str(item.message)} for item in recorded)
        pipelines[phase] = pipeline
    bundle = {"pipelines": pipelines, "text_columns": list(TEXT_COLUMNS),
        "thresholds": dict.fromkeys(PHASES, .5), "positive_class": 1,
        "seed": seed, "tfidf_parameters": normalized_json(tfidf_parameters),
        "classifier_parameters": normalized_json(classifier_parameters),
        "reason_features_used": False}
    return bundle, pd.DataFrame(diagnostics), fit_ids, warning_rows


def predict_outcome_probabilities(bundle, frame):
    """Preserve row order while routing each trial to its phase's fitted pipeline."""
    validate_outcome_rows(frame, labels_required=False)
    if list(bundle["text_columns"]) != list(TEXT_COLUMNS) or bundle["positive_class"] != 1:
        raise ValueError("Unexpected text inputs or positive-class convention.")
    probabilities = np.full(len(frame), np.nan)
    phases = frame["phase"].to_numpy()
    for phase in PHASES:
        positions = np.flatnonzero(phases == phase)
        if not len(positions):
            continue
        pipeline = bundle["pipelines"][phase]
        classes = list(pipeline.classes_)
        probabilities[positions] = pipeline.predict_proba(build_trial_text(frame.iloc[positions]))[:, classes.index(1)]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise RuntimeError("Predictions must be finite success probabilities in [0, 1].")
    return pd.DataFrame({"ID": frame["ID"].to_numpy(), "p_success": probabilities})


def load_saved_baseline(directory, *, hyperparameters, feature_spec, input_files, seed, validation):
    """Reuse immutable runs after validating real settings, inputs, and predictions."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    for field, expected in {"task": "part2", "split": "validation", "hyperparameters": hyperparameters,
            "feature_spec": feature_spec, "input_files": input_files, "seed": seed,
            "thresholds": dict.fromkeys(PHASES, .5)}.items():
        if config.get(field) != normalized_json(expected):
            raise ValueError(f"Existing run has different {field}; use a new run ID for different settings.")
    manifest = json.loads((directory / "artifact_checksums.json").read_text())
    for name, digest in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or file_sha256(directory / relative) != digest:
            raise ValueError(f"An immutable run artifact changed: {name}")
    bundle = joblib.load(directory / "model.joblib")
    probabilities = predict_outcome_probabilities(bundle, validation)
    result = evaluate_part2(validation, probabilities, .5)
    saved = pd.read_csv(directory / "validation_predictions.csv").set_index("ID").loc[validation["ID"]]
    if not np.allclose(saved["p_success"], result.predictions["p_success"], atol=1e-12, rtol=0):
        raise ValueError("Saved model does not reproduce its validation probabilities.")
    if not np.array_equal(saved["prediction"], result.predictions["prediction"]):
        raise ValueError("Saved validation decisions differ from the fixed threshold.")
    metrics = json.loads((directory / "metrics.json").read_text())
    for key, value in result.summary.items():
        if not np.isclose(value, metrics["summary"][key], atol=1e-12, rtol=0):
            raise ValueError(f"Saved primary metrics do not reproduce: {key}")
    return bundle, result, pd.read_csv(directory / "training_diagnostics.csv")


def wilson_interval(k, n, z=1.959963984540054):
    """Descriptive binomial interval; it excludes uncertainty in predicted reasons."""
    if n < 0 or k < 0 or k > n:
        raise ValueError("Require 0 <= k <= n.")
    if n == 0:
        return np.nan, np.nan
    rate = k/n
    denominator = 1 + z*z/n
    center = (rate + z*z/(2*n))/denominator
    half = z*sqrt(rate*(1-rate)/n + z*z/(4*n*n))/denominator
    return max(0., center-half), min(1., center+half)


def _rate_row(subset, phase, group, source_type="all"):
    n = len(subset)
    k = int(subset["prediction"].eq(1).sum())
    lower, upper = wilson_interval(k, n)
    observed = int(subset["source_type"].eq("observed").sum())
    predicted = int(subset["source_type"].eq("predicted").sum())
    unmatched = int(subset["source_type"].eq("unmatched").sum())
    source_count = sum(count > 0 for count in (observed, predicted, unmatched))
    evidence = ("no_trials" if n == 0 else "mixed" if source_count > 1 else
                "unmatched" if unmatched else "observed_only" if observed else "predicted_only")
    return {"phase": phase, "reason_group": group, "source_type": source_type,
        "n_trials": n, "n_predicted_success": k, "false_success_rate": k/n if n else np.nan,
        "wilson_95_lower": lower, "wilson_95_upper": upper,
        "n_observed_reasons": observed, "n_predicted_reasons": predicted, "n_unmatched": unmatched,
        "n_training_resubstitution_reasons": int(subset["prediction_is_training_resubstitution"].eq(True).sum()),
        "label_evidence": evidence, "small_support": n < 5,
        "rate_status": "ok" if n else "undefined_no_trials"}


def analyze_failure_reasons(validation_predictions, reason_lookup):
    """Compute conditional false-success rates, retaining multilabel and unmatched rows."""
    required = {"ID", "trial_id", "phase", "label", "prediction", "p_success"}
    if not required.issubset(validation_predictions):
        raise ValueError("Provide saved outcome labels, probabilities, and binary decisions.")
    if not validation_predictions["ID"].is_unique or validation_predictions.duplicated(["trial_id", "phase"]).any():
        raise ValueError("Outcome IDs and trial-phase pairs must be unique.")
    if not validation_predictions["label"].isin([0, 1]).all() or not validation_predictions["prediction"].isin([0, 1]).all():
        raise ValueError("Outcome labels and decisions must be binary.")
    scores = validation_predictions["p_success"].to_numpy(dtype=float)
    if not validation_predictions["phase"].isin(PHASES).all() or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Use known phases and finite success probabilities in [0, 1].")
    lookup_columns = {"ID", *REASONS, "is_observed", "reason_source", "reason_count", "prediction_is_training_resubstitution"}
    if not lookup_columns.issubset(reason_lookup) or not reason_lookup["ID"].is_unique:
        raise ValueError("Use a unique Part 1 lookup with explicit reason provenance.")
    if not reason_lookup[list(REASONS)].isin([0, 1]).all().all() or not reason_lookup["is_observed"].isin([True, False]).all():
        raise ValueError("Lookup reasons must be binary with known observed/predicted status.")
    if not np.array_equal(reason_lookup[list(REASONS)].sum(axis=1), reason_lookup["reason_count"]):
        raise ValueError("Lookup reason counts do not match its label vectors.")
    if (reason_lookup["is_observed"] & reason_lookup["reason_count"].eq(0)).any():
        raise ValueError("Observed Part 1 reason vectors should contain a recorded reason.")
    joined = link_unsuccessful_validation(validation_predictions, reason_lookup)
    joined = joined.merge(validation_predictions[["ID", "p_success", "prediction"]], on="ID", how="left", validate="one_to_one")
    joined["source_type"] = np.where(~joined["matched_part1"], "unmatched",
        np.where(joined["is_observed"].eq(True), "observed", "predicted"))
    joined["incorrectly_predicted_success"] = joined["prediction"].eq(1)
    primary, by_source, source_detail, overall = [], [], [], []
    for phase in PHASES:
        phase_rows = joined.loc[joined["phase"].eq(phase)]
        overall.append(_rate_row(phase_rows, phase, "all_failed_trials"))
        for group in REASON_GROUPS:
            mask = phase_rows[group].eq(1).fillna(False) if group in REASONS else phase_rows["reason_group"].eq(group)
            subset = phase_rows.loc[mask]
            primary.append(_rate_row(subset, phase, group))
            sources = ("unmatched",) if group == "no_recorded_stopping_reason" else ("predicted",) if group == "matched_no_predicted_reason" else ("observed", "predicted")
            for source in sources:
                by_source.append(_rate_row(subset.loc[subset["source_type"].eq(source)], phase, group, source))
            for source, detail in subset.groupby("reason_source", dropna=False):
                row = _rate_row(detail, phase, group)
                row["reason_source"] = source if pd.notna(source) else "no_part1_match"
                source_detail.append(row)
    return {"failed_rows": joined, "reason_rates": pd.DataFrame(primary),
            "source_rates": pd.DataFrame(by_source), "source_detail_rates": pd.DataFrame(source_detail),
            "overall_failed_rates": pd.DataFrame(overall)}
