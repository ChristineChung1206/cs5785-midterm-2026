"""Part 1 structure, embedding geometry, and temporal backtest study.

This writes a separate candidate study. The existing final is never replaced.
Held-out labels are excluded before any fitted transform, K-Means, or classifier.
"""
from __future__ import annotations

from _historical_support import prepare_output

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from midterm_augmentation import estimate_cluster_reason_rates, logistic_parameters, observed_label_mask
from midterm_evaluation import REASONS, dataset_path, file_sha256, part1_decisions
from midterm_next_round import p1_assignment
from midterm_part1_finalization import build_submission, predict_bundle
from midterm_part2_improvement import validate_submission
from midterm_refinement import EMBEDDINGS, prepare_reason_context

STUDY = Path("refinement/part1_structure_geometry_2026_11_02")
THRESHOLDS = (.35, .45, .55, .65)
DIMS = (30, 60, 100)
SEED = 42


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_locked_json(path, value):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Locked selection changed: {path}")
    write_json(path, value)


def score(y, decision):
    return float(f1_score(y, decision, average="samples", zero_division=0))


def make_decisions(kind, probabilities, secondary=None, threshold=None):
    p = np.asarray(probabilities)
    if kind == "binary":
        return part1_decisions(p, .35, empty_set_policy="argmax")
    decision = np.zeros((len(p), len(REASONS)), dtype=int)
    first = p.argmax(axis=1)
    decision[np.arange(len(p)), first] = 1
    if kind == "primary_secondary":
        other = np.asarray(secondary).copy()
        other[np.arange(len(p)), first] = -1
        second = other.argmax(axis=1)
        take = other[np.arange(len(p)), second] >= threshold
        decision[np.flatnonzero(take), second[take]] = 1
    return decision


def fit_primary(fit):
    labels = fit[list(REASONS)].to_numpy()
    observed = observed_label_mask(labels)
    single = observed & ((labels == 1).sum(axis=1) == 1)
    x = fit[EMBEDDINGS].to_numpy(dtype=float)
    scaler = StandardScaler().fit(x)
    y = labels[single].argmax(axis=1)
    if set(y) != set(range(len(REASONS))):
        raise ValueError("Every primary class must have a single-label gold example.")
    model = LogisticRegression(C=.1, solver="lbfgs", max_iter=1000, random_state=SEED)
    with threadpool_limits(limits=1):
        model.fit(scaler.transform(x[single]), y)
    return {"kind": "primary", "scaler": scaler, "model": model,
            "gold_single_ids": fit.loc[single, "ID"].tolist()}


def primary_scores(bundle, frame):
    p = bundle["model"].predict_proba(bundle["scaler"].transform(frame[EMBEDDINGS].to_numpy()))
    if not np.array_equal(bundle["model"].classes_, np.arange(len(REASONS))):
        raise ValueError("Unexpected primary class order.")
    return p


class CovarianceGeometry:
    """Rotation-aware PCA projection with Ledoit-Wolf shrunk whitening."""

    def __init__(self, dimensions):
        self.dimensions = dimensions

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        estimator = LedoitWolf(assume_centered=False).fit(x)
        eigenvalues, eigenvectors = np.linalg.eigh(estimator.covariance_)
        order = np.argsort(eigenvalues)[::-1][:self.dimensions]
        self.mean_ = estimator.location_
        self.rotation_ = eigenvectors[:, order] / np.sqrt(eigenvalues[order])[None, :]
        self.eigenvalues_ = eigenvalues[order]
        self.shrinkage_ = float(estimator.shrinkage_)
        return self

    def transform(self, x):
        return (np.asarray(x, dtype=float) - self.mean_) @ self.rotation_


def fit_geometry(fit, dimensions):
    labels = fit[list(REASONS)].to_numpy()
    observed = observed_label_mask(labels)
    transform = CovarianceGeometry(dimensions).fit(fit[EMBEDDINGS].to_numpy())
    x = transform.transform(fit[EMBEDDINGS].to_numpy())
    clusterer = KMeans(n_clusters=10, n_init=20, max_iter=500, random_state=SEED)
    with threadpool_limits(limits=1):
        clusterer.fit(x)
    membership = np.eye(10)[clusterer.labels_]
    mapping = estimate_cluster_reason_rates(membership, labels, 2.0)
    pseudo = (membership[~observed] @ mapping["rates"] >= .5).astype(int)
    recovered = labels.copy().astype(int)
    recovered[~observed] = pseudo
    weights = np.where(observed, 1., .1)
    models = []
    for j in range(len(REASONS)):
        model = LogisticRegression(**{**logistic_parameters(SEED), "C": .1})
        with threadpool_limits(limits=1):
            model.fit(x, recovered[:, j], sample_weight=weights)
        models.append(model)
    return {"kind": "geometry", "dimensions": dimensions, "transform": transform,
            "clusterer": clusterer, "mapping": mapping, "models": models,
            "gold_ids": fit.loc[observed, "ID"].tolist()}


def geometry_scores(bundle, frame):
    x = bundle["transform"].transform(frame[EMBEDDINGS].to_numpy())
    return np.column_stack([m.predict_proba(x)[:, 1] for m in bundle["models"]])


def fit_centroid(fit, dimensions):
    labels = fit[list(REASONS)].to_numpy()
    observed = observed_label_mask(labels)
    single = observed & ((labels == 1).sum(axis=1) == 1)
    transform = CovarianceGeometry(dimensions).fit(fit[EMBEDDINGS].to_numpy())
    x = transform.transform(fit[EMBEDDINGS].to_numpy())
    y = labels[single].argmax(axis=1)
    if set(y) != set(range(len(REASONS))):
        raise ValueError("Every class needs a centroid.")
    centers = np.vstack([x[single][y == j].mean(axis=0) for j in range(len(REASONS))])
    prior = np.bincount(y, minlength=len(REASONS)) / len(y)
    return {"kind": "centroid", "dimensions": dimensions, "transform": transform,
            "centers": centers, "prior": prior, "gold_single_ids": fit.loc[single, "ID"].tolist()}


def centroid_scores(bundle, frame):
    x = bundle["transform"].transform(frame[EMBEDDINGS].to_numpy())
    distance = ((x[:, None, :] - bundle["centers"][None, :, :]) ** 2).sum(axis=2)
    return softmax(-distance / 2 + np.log(bundle["prior"])[None, :], axis=1)


def fit_control(fit):
    from midterm_next_round import fit_reason_weight
    context = prepare_reason_context(fit, 10)
    return fit_reason_weight(fit, context, .1)


def control_scores(bundle, frame):
    x = bundle["scaler"].transform(frame[EMBEDDINGS])
    return np.column_stack([m.predict_proba(x)[:, 1] for m in bundle["models"]])


def fit_candidate(name, fit):
    if name == "control":
        return fit_control(fit)
    if name == "primary":
        return fit_primary(fit)
    method, dimensions = name.split("_")
    return fit_geometry(fit, int(dimensions)) if method == "geometry" else fit_centroid(fit, int(dimensions))


def predict_candidate(name, bundle, frame):
    if name == "control":
        return control_scores(bundle, frame)
    if name == "primary":
        return primary_scores(bundle, frame)
    if name.startswith("geometry_"):
        return geometry_scores(bundle, frame)
    return centroid_scores(bundle, frame)


def metrics_rows(name, split, reference, decisions, probabilities=None):
    y = reference[list(REASONS)].to_numpy(dtype=int)
    rows = [{"model": name, "split": split, "group": "all", "n": len(y), "f1_samples": score(y, decisions),
             "predicted_multi": int((decisions.sum(axis=1) > 1).sum())}]
    for label, mask in (("single_truth", y.sum(axis=1) == 1), ("multi_truth", y.sum(axis=1) > 1),
                        ("late_2007_2009", reference.start_year.between(2007, 2009).to_numpy())):
        if mask.any():
            rows.append({"model": name, "split": split, "group": label, "n": int(mask.sum()),
                         "f1_samples": score(y[mask], decisions[mask]),
                         "predicted_multi": int((decisions[mask].sum(axis=1) > 1).sum())})
    for j, reason in enumerate(REASONS):
        rows.append({"model": name, "split": split, "group": reason, "n": len(y),
                     "f1_samples": float(f1_score(y[:, j], decisions[:, j], zero_division=0)),
                     "predicted_multi": None})
    return rows


def run(root, output):
    root, out = prepare_output(root, output)
    train = pd.read_csv(dataset_path(root, "part1", "train"))
    val = pd.read_csv(dataset_path(root, "part1", "val"))
    test = pd.read_csv(dataset_path(root, "part1", "test"))
    gold, assignment = p1_assignment(train)
    names = ["control", "primary", *[f"geometry_{d}" for d in DIMS], *[f"centroid_{d}" for d in DIMS]]
    write_json(out / "plan.json", {"models": names, "secondary_thresholds": THRESHOLDS,
        "folds": 4, "seed": SEED, "primary_training": "single-label gold only",
        "secondary_training": "existing K-Means augmented binary reason model",
        "geometry": "Ledoit-Wolf shrunk covariance eigenvectors, top 30/60/100, inverse-square-root scaling",
        "temporal_fit": "train rows start_year <= 2006, held-out train gold 2007-2009",
        "validation_in_selection": False, "test_labels_used": False,
        "input_sha256": {split: file_sha256(dataset_path(root, "part1", split)) for split in ("train", "val", "test")}})
    oof = {name: np.full((len(gold), len(REASONS)), np.nan) for name in names}
    fold_by_id = assignment.set_index("ID").loc[gold.ID, "fold"].to_numpy()
    for fold in range(4):
        held = gold.loc[fold_by_id == fold].reset_index(drop=True)
        fit = train.loc[~train.ID.isin(held.ID)].reset_index(drop=True)
        assert set(fit.ID).isdisjoint(held.ID)
        for name in names:
            bundle = fit_candidate(name, fit)
            oof[name][fold_by_id == fold] = predict_candidate(name, bundle, held)
        print(f"OOF fold {fold + 1}/4 done", flush=True)
    for name in names:
        assert np.isfinite(oof[name]).all()
    y = gold[list(REASONS)].to_numpy(dtype=int)
    decisions = {"control": make_decisions("binary", oof["control"]),
                 "primary_only": make_decisions("primary", oof["primary"])}
    structure_grid = []
    for t in THRESHOLDS:
        d = make_decisions("primary_secondary", oof["primary"], oof["control"], t)
        structure_grid.append({"threshold": t, "oof_f1_samples": score(y, d),
                               "oof_predicted_multi": int((d.sum(axis=1) > 1).sum())})
    threshold = max(structure_grid, key=lambda row: (row["oof_f1_samples"], row["threshold"]))["threshold"]
    decisions["primary_secondary"] = make_decisions("primary_secondary", oof["primary"], oof["control"], threshold)
    for name in names:
        if name.startswith("geometry_"):
            decisions[name] = make_decisions("binary", oof[name])
        elif name.startswith("centroid_"):
            decisions[name] = make_decisions("primary", oof[name])
    oof_rows = []
    for name, d in decisions.items():
        oof_rows.extend(metrics_rows(name, "oof", gold, d))
    pd.DataFrame(oof_rows).to_csv(out / "oof_metrics.csv", index=False)
    pd.DataFrame(structure_grid).to_csv(out / "primary_secondary_threshold_oof.csv", index=False)
    flat_oof = gold[["ID", "start_year", *REASONS]].copy()
    flat_oof["fold"] = fold_by_id
    for name in names:
        for j, reason in enumerate(REASONS):
            flat_oof[f"{name}_prob_{reason}"] = oof[name][:, j]
    for name, d in decisions.items():
        for j, reason in enumerate(REASONS):
            flat_oof[f"{name}_pred_{reason}"] = d[:, j]
    flat_oof.to_csv(out / "oof_predictions.csv", index=False)
    # Pre-register the OOF winner before evaluating validation.
    oof_all = {name: score(y, d) for name, d in decisions.items()}
    ranked = sorted(decisions, key=lambda name: (-oof_all[name], name))
    write_locked_json(out / "selection_locked_before_validation.json", {
        "oof_ranked": ranked, "oof_sample_f1": oof_all, "secondary_threshold": threshold,
        "validation_labels_seen": False, "selection_rule": "highest pooled OOF sample F1; time backtest and validation are directional checks"})
    # Temporal backtest excludes every row later than 2006 from fitting.
    temporal_fit = train.loc[train.start_year <= 2006].reset_index(drop=True)
    temporal_hold = gold.loc[gold.start_year.between(2007, 2009)].reset_index(drop=True)
    temporal_scores = {name: predict_candidate(name, fit_candidate(name, temporal_fit), temporal_hold) for name in names}
    td = {"control": make_decisions("binary", temporal_scores["control"]),
          "primary_only": make_decisions("primary", temporal_scores["primary"]),
          "primary_secondary": make_decisions("primary_secondary", temporal_scores["primary"], temporal_scores["control"], threshold)}
    for name in names:
        if name.startswith("geometry_"):
            td[name] = make_decisions("binary", temporal_scores[name])
        elif name.startswith("centroid_"):
            td[name] = make_decisions("primary", temporal_scores[name])
    pd.DataFrame([row for name, d in td.items() for row in metrics_rows(name, "temporal", temporal_hold, d)]).to_csv(out / "temporal_metrics.csv", index=False)
    # Provided validation is a single pre-refit check, not an independent test after refit.
    full_models = {name: fit_candidate(name, train) for name in names}
    val_scores = {name: predict_candidate(name, full_models[name], val) for name in names}
    vd = {"control": make_decisions("binary", val_scores["control"]),
          "primary_only": make_decisions("primary", val_scores["primary"]),
          "primary_secondary": make_decisions("primary_secondary", val_scores["primary"], val_scores["control"], threshold)}
    for name in names:
        if name.startswith("geometry_"):
            vd[name] = make_decisions("binary", val_scores[name])
        elif name.startswith("centroid_"):
            vd[name] = make_decisions("primary", val_scores[name])
    pd.DataFrame([row for name, d in vd.items() for row in metrics_rows(name, "validation", val, d)]).to_csv(out / "validation_metrics_before_refit.csv", index=False)
    frozen = joblib.load(root / 'models/part1/02_kmeans_lr/model.joblib')
    frozen_val = make_decisions("binary", predict_bundle(frozen, val))
    frozen_score = score(val[list(REASONS)].to_numpy(dtype=int), frozen_val)
    frozen_test_hash = file_sha256(root / 'models/part1/02_kmeans_lr/submission.csv')
    # Gate requires improvement in OOF, time backtest, and original validation.
    temporal_all = {name: score(temporal_hold[list(REASONS)].to_numpy(dtype=int), d) for name, d in td.items()}
    val_all = {name: score(val[list(REASONS)].to_numpy(dtype=int), d) for name, d in vd.items()}
    eligible = [name for name in ranked if name != "control"
                and oof_all[name] > oof_all["control"] + 1e-12
                and temporal_all[name] >= temporal_all["control"] - 1e-12
                and val_all[name] >= frozen_score - 1e-12]
    selected = eligible[0] if eligible else "control"
    write_locked_json(out / "pre_refit_selection.json", {"selected": selected, "eligible": eligible,
        "oof_sample_f1": oof_all, "temporal_sample_f1": temporal_all,
        "validation_sample_f1": val_all, "frozen_validation_f1": frozen_score,
        "validation_labels_used_in_fitting_at_this_point": False,
        "old_final_submission_sha256": frozen_test_hash})
    # The final candidate uses all 235 visible labels only after the selection is frozen.
    combined = pd.concat([train, val], ignore_index=True)
    if combined.ID.duplicated().any() or observed_label_mask(combined[list(REASONS)].to_numpy()).sum() != 235:
        raise ValueError("Expected 235 unique visible gold labels in train+validation.")
    final_name = "primary" if selected == "primary_secondary" else selected
    refit = {final_name: fit_candidate(final_name, combined)}
    if selected == "primary_secondary":
        refit["control"] = fit_candidate("control", combined)
    test_scores = {name: predict_candidate(name, model, test) for name, model in refit.items()}
    if selected == "primary_secondary":
        test_decision = make_decisions("primary_secondary", test_scores["primary"], test_scores["control"], threshold)
    elif selected == "control" or selected.startswith("geometry_"):
        test_decision = make_decisions("binary", test_scores[selected])
    else:
        test_decision = make_decisions("primary", test_scores[selected])
    predictions = pd.DataFrame(test_decision, columns=REASONS)
    predictions.insert(0, "ID", test.ID.to_numpy())
    sample = pd.read_csv(dataset_path(root, "part1", "test").parent / "part1_sample_submission.csv")
    submission = build_submission(sample, predictions)
    submission_path = out / "part1_refit_235_candidate.csv"
    submission.to_csv(submission_path, index=False)
    validate_submission(submission, sample, 1)
    joblib.dump({"selected": selected, "models": refit, "threshold": threshold,
                "label_source": "part1_train observed + part1_val", "gold_count": 235},
                out / "refit_model.joblib", compress=3)
    predictions.to_csv(out / "test_decisions.csv", index=False)
    if file_sha256(root / 'models/part1/02_kmeans_lr/submission.csv') != frozen_test_hash:
        raise RuntimeError("Original Part 1 final was modified.")
    write_json(out / "summary.json", {"selected": selected, "oof_f1": oof_all[selected],
        "time_f1": temporal_all[selected], "pre_refit_validation_f1": val_all[selected],
        "frozen_validation_f1": frozen_score, "refit_gold_count": 235,
        "test_predicted_multi": int((test_decision.sum(axis=1) > 1).sum()),
        "candidate_path": str(submission_path.relative_to(out)),
        "candidate_sha256": file_sha256(submission_path), "original_final_replaced": False,
        "kaggle_uploaded": False,
        "warning": "Validation labels entered final refit; validation score is no longer an independent estimate for that fitted model."})
    print(json.dumps(json.loads((out / "summary.json").read_text()), indent=2), flush=True)




if __name__ == "__main__":
    from replay import main
    main(default_study='structure_geometry')
