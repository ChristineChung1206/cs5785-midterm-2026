"""Reproducible, label-leakage-aware comparison of the two retained Part 1 models.

Run from the curated midterm directory with ``python midterm_part1_parity.py``.
This script only writes a new analysis directory; it never replaces a submission.
"""

from __future__ import annotations

from _historical_support import prepare_output

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from midterm_augmentation import (
    cluster_memberships, estimate_cluster_reason_rates, logistic_parameters,
    observed_label_mask, predict_reason_probabilities, recover_labels,
)
from midterm_evaluation import REASONS, dataset_path, part1_decisions
from midterm_part1_finalization import fit_classifiers


ROOT = None  # Set only by explicit replay(root, output).
OUT = None
EMB = [f"emb_{i}" for i in range(150)]
SEED = 42
THRESHOLDS = {"supervised": 0.50, "semi": 0.35}
COMMON_THRESHOLDS = (0.35, 0.50)
EPS = 1e-12


def train_models(features: pd.DataFrame, labels: np.ndarray):
    """Fit scaler on available features and classifiers on permitted labels only."""
    labels = np.asarray(labels, dtype=int)
    visible = observed_label_mask(labels)
    scaler = StandardScaler().fit(features[EMB])
    x = scaler.transform(features[EMB])

    baseline = OneVsRestClassifier(
        LogisticRegression(**logistic_parameters(SEED)), n_jobs=1
    ).fit(x[visible], labels[visible])

    clusterer = KMeans(
        n_clusters=10, n_init=20, tol=1e-4, max_iter=500,
        algorithm="lloyd", random_state=SEED,
    ).fit(x)
    membership = cluster_memberships(clusterer, x, "kmeans")
    mapping = estimate_cluster_reason_rates(membership, labels, smoothing_strength=2.0)
    recovered = recover_labels(labels, membership, mapping, threshold=0.5, pseudo_weight=0.1)
    semi, _ = fit_classifiers(
        x, recovered["labels"], recovered["sample_weight"], C=0.1, seed=SEED,
    )
    return scaler, baseline, semi


def score_models(models, frame: pd.DataFrame):
    scaler, baseline, semi = models
    x = scaler.transform(frame[EMB])
    return {"supervised": baseline.predict_proba(x),
            "semi": predict_reason_probabilities(semi, x)}


def decisions(probabilities, threshold=None):
    return {name: part1_decisions(p, THRESHOLDS[name] if threshold is None else threshold,
                                  empty_set_policy="argmax")
            for name, p in probabilities.items()}


def sample_scores(truth, predicted):
    intersections = (truth & predicted).sum(axis=1)
    denominators = truth.sum(axis=1) + predicted.sum(axis=1)
    return np.divide(2 * intersections, denominators,
                     out=np.zeros(len(truth), dtype=float), where=denominators != 0)


def metrics(truth, predicted):
    p, r, f, support = precision_recall_fscore_support(
        truth, predicted, average=None, zero_division=0,
    )
    return {"n": len(truth), "sample_f1": float(f1_score(truth, predicted, average="samples", zero_division=0)),
            "exact_sets": int(np.all(truth == predicted, axis=1).sum()),
            "per_reason": {reason: {"support": int(support[j]), "predicted": int(predicted[:, j].sum()),
                                    "precision": float(p[j]), "recall": float(r[j]), "f1": float(f[j])}
                           for j, reason in enumerate(REASONS)}}


def sets(values):
    return [";".join(reason for reason, value in zip(REASONS, row) if value) for row in values]


def detail_frame(frame, truth, probs, pred, *, split, folds=None, early_embeddings=None):
    result = frame[["ID", "start_year"]].reset_index(drop=True).copy()
    result.insert(1, "split", split)
    if folds is not None:
        result["fold"] = folds
    if early_embeddings is not None:
        early_keys = {tuple(row) for row in early_embeddings.to_numpy()}
        result["exact_early_embedding_match"] = [tuple(row) in early_keys for row in frame[EMB].to_numpy()]
    result["truth"] = sets(truth)
    result["supervised"] = sets(pred["supervised"])
    result["semi"] = sets(pred["semi"])
    for name in ("supervised", "semi"):
        result[f"{name}_row_f1"] = sample_scores(truth, pred[name])
        result[f"{name}_exact"] = np.all(truth == pred[name], axis=1)
        for j, reason in enumerate(REASONS):
            result[f"{name}_prob_{reason}"] = probs[name][:, j]
    result["winner"] = np.where(result["supervised_row_f1"] > result["semi_row_f1"] + EPS,
                                "supervised", np.where(result["semi_row_f1"] > result["supervised_row_f1"] + EPS,
                                                       "semi", "tie"))
    result["different_sets"] = np.any(pred["supervised"] != pred["semi"], axis=1)
    return result


def comparison(frame, truth, probs, *, split, folds=None, early_embeddings=None):
    pred = decisions(probs)
    details = detail_frame(frame, truth, probs, pred, split=split, folds=folds,
                           early_embeddings=early_embeddings)
    by_year = []
    for year in sorted(frame["start_year"].unique()):
        sel = frame["start_year"].to_numpy() == year
        by_year.append({"split": split, "start_year": int(year), "n": int(sel.sum()),
                        **{f"{name}_sample_f1": metrics(truth[sel], pred[name][sel])["sample_f1"]
                           for name in pred}})
    return {"metrics": {name: metrics(truth, pred[name]) for name in pred},
            "common_thresholds": {str(t): {name: metrics(truth, chosen)["sample_f1"]
                                          for name, chosen in decisions(probs, t).items()}
                                  for t in COMMON_THRESHOLDS},
            "wins": {name: int((details["winner"] == name).sum()) for name in ("supervised", "semi", "tie")},
            "different_sets": int(details["different_sets"].sum()),
            "by_year": by_year}, details


def verify_saved_predictions(train, val, labels):
    """Check the refit recipe against immutable historical validation outputs."""
    models = train_models(train, labels)
    fitted = score_models(models, val)
    records = {
        "supervised": ROOT / 'models/part1/01_supervised_lr/evidence/original_run/validation_predictions.csv',
        "semi": ROOT / 'models/part1/02_kmeans_lr/evidence/selected/selected_validation/validation_predictions.csv',
    }
    gaps = {}
    for name, path in records.items():
        saved = pd.read_csv(path).set_index("ID").loc[val["ID"]]
        actual = saved[[f"prob_{reason}" for reason in REASONS]].to_numpy()
        gaps[name] = float(np.max(np.abs(fitted[name] - actual)))
        if gaps[name] > 1e-10:
            raise RuntimeError(f"{name} refit failed historical parity: {gaps[name]}")
    # Also check that the saved model artifacts themselves reproduce the records.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        base_bundle = joblib.load(ROOT / 'models/part1/01_supervised_lr/model.joblib')
    saved_base = base_bundle["pipeline"].predict_proba(val[EMB])
    if not np.allclose(saved_base, fitted["supervised"], atol=1e-10, rtol=0):
        raise RuntimeError("Saved supervised model disagrees with the refit recipe")
    return models, fitted, gaps


def grouped_oof(train, labels):
    visible = observed_label_mask(labels)
    gold = train.loc[visible].reset_index(drop=True)
    y = labels[visible]
    keys = pd.factorize(pd.Series([tuple(row) for row in gold[EMB].to_numpy()]), sort=False)[0]
    primary = y.argmax(axis=1)
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    probs = {name: np.full((len(gold), 5), np.nan) for name in THRESHOLDS}
    folds = np.full(len(gold), -1, dtype=int)
    gold_positions = np.flatnonzero(visible)
    for fold, (_, held) in enumerate(splitter.split(gold, primary, keys)):
        # Held rows must be absent even from scaler and K-Means fitting:
        # the original selected model did not fit on competition test features.
        fit_mask = np.ones(len(train), dtype=bool)
        fit_mask[gold_positions[held]] = False
        models = train_models(train.loc[fit_mask], labels[fit_mask])
        scores = score_models(models, gold.iloc[held])
        for name in probs:
            probs[name][held] = scores[name]
        folds[held] = fold
    if np.isnan(probs["supervised"]).any() or (folds < 0).any():
        raise RuntimeError("OOF assignments are incomplete")
    if any(len(set(keys[folds == fold]) & set(keys[folds != fold])) for fold in range(4)):
        raise RuntimeError("Identical gold embeddings crossed a fold boundary")
    return gold, y, probs, folds, int(len(np.unique(keys)))


def temporal_backtest(train, val, labels):
    """Strict pre-2007 fit; later labels are never passed to fitting functions."""
    early = train["start_year"].to_numpy() <= 2006
    early_train = train.loc[early].reset_index(drop=True)
    early_labels = labels[early]
    if not observed_label_mask(early_labels).any():
        raise RuntimeError("No early labels")
    models = train_models(early_train, early_labels)
    late_train = train.loc[(train["start_year"].between(2007, 2009)) & observed_label_mask(labels)]
    late_val = val.loc[val["start_year"].between(2007, 2009)]
    late = pd.concat([late_train, late_val], ignore_index=True)
    truth = late[list(REASONS)].to_numpy(dtype=int)
    probs = score_models(models, late)
    exact_early = early_train.loc[observed_label_mask(early_labels), EMB]
    return late, truth, probs, len(late_train), exact_early


def run(root, output):
    global ROOT, OUT
    ROOT, OUT = prepare_output(root, output)
    train = pd.read_csv(dataset_path(ROOT, "part1", "train"))
    val = pd.read_csv(dataset_path(ROOT, "part1", "val"))
    test = pd.read_csv(dataset_path(ROOT, "part1", "test"))
    labels = train[list(REASONS)].to_numpy(dtype=int)
    assert len(train) == 1171 and observed_label_mask(labels).sum() == 157 and len(val) == 78
    OUT.mkdir(parents=True, exist_ok=True)

    full_models, val_probs, parity = verify_saved_predictions(train, val, labels)
    test_probs = score_models(full_models, test)
    test_pred = decisions(test_probs)
    test_disagreements = test[["ID", "start_year"]].copy()
    for name in THRESHOLDS:
        test_disagreements[name] = sets(test_pred[name])
    test_disagreements["different_sets"] = np.any(
        test_pred["supervised"] != test_pred["semi"], axis=1,
    )
    test_disagreements.loc[test_disagreements["different_sets"]].to_csv(
        OUT / "test_disagreements.csv", index=False,
    )
    validation, val_details = comparison(val, val[list(REASONS)].to_numpy(dtype=int), val_probs,
                                         split="validation")
    gold, truth, oof_probs, folds, unique_groups = grouped_oof(train, labels)
    oof, oof_details = comparison(gold, truth, oof_probs, split="grouped_oof", folds=folds)
    late, late_truth, late_probs, n_train_gold, early_emb = temporal_backtest(train, val, labels)
    late_result, late_details = comparison(late, late_truth, late_probs, split="late_2007_2009",
                                           early_embeddings=early_emb)
    # Primary time check uses only gold rows that originated in the training CSV.
    late_primary, _ = comparison(late.iloc[:n_train_gold], late_truth[:n_train_gold],
                                 {name: p[:n_train_gold] for name, p in late_probs.items()},
                                 split="late_train_only", early_embeddings=early_emb)
    novel = ~late_details.iloc[:n_train_gold]["exact_early_embedding_match"].to_numpy()
    late_novel, _ = comparison(
        late.iloc[:n_train_gold].iloc[novel], late_truth[:n_train_gold][novel],
        {name: p[:n_train_gold][novel] for name, p in late_probs.items()},
        split="late_train_no_exact_early_embedding", early_embeddings=early_emb,
    )

    for name, details in (("validation", val_details), ("grouped_oof", oof_details),
                          ("late_2007_2009", late_details)):
        details.to_csv(OUT / f"{name}_predictions.csv", index=False)
    pd.concat([d.loc[d["different_sets"]] for d in (val_details, oof_details, late_details)],
              ignore_index=True).to_csv(OUT / "changed_trials.csv", index=False)
    pd.DataFrame(validation["by_year"] + oof["by_year"] + late_result["by_year"]).to_csv(
        OUT / "metrics_by_year.csv", index=False,
    )
    per_reason = []
    for split, result in (("validation", validation), ("grouped_oof", oof),
                          ("late_train_only", late_primary), ("late_train_no_exact_early_embedding", late_novel),
                          ("late_2007_2009", late_result)):
        for model, m in result["metrics"].items():
            for reason, values in m["per_reason"].items():
                per_reason.append({"split": split, "model": model, "reason": reason, **values})
    pd.DataFrame(per_reason).to_csv(OUT / "metrics_by_reason.csv", index=False)

    # Selection gate fixed before inspecting the run: a new candidate needs a
    # meaningful OOF gain, a meaningful strict-time gain, and no val regression.
    a, b = "supervised", "semi"
    oof_delta = oof["metrics"][a]["sample_f1"] - oof["metrics"][b]["sample_f1"]
    time_delta = late_primary["metrics"][a]["sample_f1"] - late_primary["metrics"][b]["sample_f1"]
    val_delta = validation["metrics"][a]["sample_f1"] - validation["metrics"][b]["sample_f1"]
    unique_wins = late_details.iloc[:n_train_gold]["winner"].value_counts().to_dict()
    gate = {"oof_gain_at_least_0_01": bool(oof_delta >= 0.01),
            "strict_time_gain_at_least_0_02": bool(time_delta >= 0.02),
            "strict_time_at_least_two_net_row_wins": bool(unique_wins.get(a, 0) - unique_wins.get(b, 0) >= 2),
            "validation_nonworse": bool(val_delta >= -1e-12)}
    result = {"protocol": {"seed": SEED, "folds": 4, "unique_gold_embedding_groups": unique_groups,
                           "oof_held_features_excluded_from_all_fits": True,
                           "temporal_fit": "train rows start_year <= 2006 only",
                           "temporal_primary": "2007-2009 observed labels in train CSV only",
                           "temporal_56_row_check": "includes previously inspected validation labels; diagnostic only",
                           "fallback": "argmax reason on every empty predicted set",
                           "historical_thresholds": THRESHOLDS, "common_thresholds": list(COMMON_THRESHOLDS),
                           "full_refit_max_absolute_validation_probability_gap": parity},
              "validation": validation, "grouped_oof": oof,
              "late_train_only": late_primary,
              "late_train_no_exact_early_embedding": late_novel,
              "late_2007_2009": late_result,
              "supervised_minus_semi": {"oof_sample_f1": oof_delta, "strict_time_sample_f1": time_delta,
                                        "validation_sample_f1": val_delta},
              "unlabeled_test_disagreements": int(test_disagreements["different_sets"].sum()),
              "candidate_gate": gate, "candidate_approved_for_local_generation": all(gate.values())}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"parity": parity, "oof_delta": oof_delta, "strict_time_delta": time_delta,
                      "validation_delta": val_delta, "gate": gate}, indent=2))




if __name__ == "__main__":
    from replay import main
    main(default_study='parity')
