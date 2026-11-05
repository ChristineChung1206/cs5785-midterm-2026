"""Post-selection sensitivity: include all 78 validation labels with bounded weight.

Cluster mapping stays fitted on train only. Validation labels enter only the
final classifier. Each OOF/temporal held-out ID is excluded before any fitting.
"""
from __future__ import annotations

from _historical_support import prepare_output

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from midterm_augmentation import logistic_parameters, observed_label_mask
from midterm_evaluation import REASONS, dataset_path, file_sha256
from midterm_next_round import p1_assignment
from midterm_part1_finalization import build_submission
from midterm_part1_round2 import make_decisions, metrics_rows, score
from midterm_part2_improvement import validate_submission
from midterm_refinement import EMBEDDINGS, prepare_reason_context

ROOT = None  # Set only by explicit replay(root, output).
OUT = None
WEIGHTS = (.25, .5, 1.)


def fit_with_extra(train, extra, weight):
    if set(train.ID) & set(extra.ID):
        raise ValueError("Training and extra label IDs overlap.")
    if not observed_label_mask(extra[list(REASONS)].to_numpy()).all():
        raise ValueError("Extra validation labels must be observed.")
    context = prepare_reason_context(train, 10)
    observed = context["observed"]
    x = context["x"]
    labels = train[list(REASONS)].to_numpy(dtype=int).copy()
    labels[~observed] = (context["pseudo_q"] >= .5).astype(int)
    extra_x = context["scaler"].transform(extra[EMBEDDINGS])
    both_x = np.vstack([x, extra_x])
    both_y = np.vstack([labels, extra[list(REASONS)].to_numpy(dtype=int)])
    sample_weights = np.concatenate([np.where(observed, 1., .1), np.full(len(extra), weight)])
    models = []
    for j in range(len(REASONS)):
        model = LogisticRegression(**{**logistic_parameters(42), "C": .1})
        with threadpool_limits(limits=1):
            model.fit(both_x, both_y[:, j], sample_weight=sample_weights)
        models.append(model)
    return {"scaler": context["scaler"], "clusterer": context["clusterer"],
            "mapping": context["mapping"], "models": models, "val_weight": weight,
            "fit_train_ids": train.ID.tolist(), "fit_extra_ids": extra.ID.tolist()}


def predict(bundle, frame):
    x = bundle["scaler"].transform(frame[EMBEDDINGS])
    return np.column_stack([model.predict_proba(x)[:, 1] for model in bundle["models"]])


def run(root, output):
    global ROOT, OUT
    ROOT, OUT = prepare_output(root, output)
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(dataset_path(ROOT, "part1", "train"))
    val = pd.read_csv(dataset_path(ROOT, "part1", "val"))
    test = pd.read_csv(dataset_path(ROOT, "part1", "test"))
    gold, assignment = p1_assignment(train)
    fold = assignment.set_index("ID").loc[gold.ID, "fold"].to_numpy()
    combined = pd.concat([train, val], ignore_index=True)
    all_gold = combined.loc[observed_label_mask(combined[list(REASONS)].to_numpy())]
    temporal_hold = all_gold.loc[all_gold.start_year.between(2007, 2009)]
    temporal_train = train.loc[train.start_year <= 2006]
    temporal_extra = val.loc[val.start_year <= 2006]
    assert len(temporal_hold) == 56 and set(temporal_train.ID).isdisjoint(temporal_hold.ID)
    assert set(temporal_extra.ID).isdisjoint(temporal_hold.ID)
    rows = []
    checks = []
    for weight in WEIGHTS:
        oof = np.full((len(gold), len(REASONS)), np.nan)
        for k in range(4):
            held = gold.loc[fold == k]
            fit_train = train.loc[~train.ID.isin(held.ID)]
            assert set(fit_train.ID).isdisjoint(held.ID) and set(val.ID).isdisjoint(held.ID)
            oof[fold == k] = predict(fit_with_extra(fit_train, val, weight), held)
        d_oof = make_decisions("binary", oof)
        d_time = make_decisions("binary", predict(fit_with_extra(temporal_train, temporal_extra, weight), temporal_hold))
        name = f"val_weight_{weight:g}"
        rows.extend(metrics_rows(name, "train_gold_oof", gold, d_oof))
        rows.extend(metrics_rows(name, "temporal_56", temporal_hold, d_time))
        checks.append({"weight": weight, "oof_157_f1": score(gold[list(REASONS)], d_oof),
                       "time_56_f1": score(temporal_hold[list(REASONS)], d_time),
                       "time_predicted_multi": int((d_time.sum(axis=1) > 1).sum())})
        print(name, checks[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "comparison.csv", index=False)
    # Fixed controls from identical folds/splits; no post-refit validation metric.
    baseline_oof = .8895966029723991
    baseline_time = .869047619047619
    qualified = [row for row in checks if row["oof_157_f1"] > baseline_oof + 1e-12
                 and row["time_56_f1"] >= baseline_time - 1e-12]
    selected = sorted(qualified, key=lambda row: (-row["time_56_f1"], -row["oof_157_f1"], row["weight"]))[0] if qualified else None
    summary = {"baseline_oof_157_f1": baseline_oof, "baseline_time_56_f1": baseline_time,
               "selected": selected, "all_labels_235_used_in_final_fit": bool(selected),
               "original_final_replaced": False, "kaggle_uploaded": False,
               "caveat": "Exploratory after earlier validation inspection; no post-refit independent validation score."}
    if selected:
        bundle = fit_with_extra(train, val, selected["weight"])
        d_test = make_decisions("binary", predict(bundle, test))
        pred = pd.DataFrame(d_test, columns=REASONS)
        pred.insert(0, "ID", test.ID.to_numpy())
        sample = pd.read_csv(dataset_path(ROOT, "part1", "test").parent / "part1_sample_submission.csv")
        submission = build_submission(sample, pred)
        path = OUT / "part1_val_weight_candidate.csv"
        submission.to_csv(path, index=False)
        validate_submission(submission, sample, 1)
        joblib.dump(bundle, OUT / "selected_model.joblib", compress=3)
        summary["candidate_path"] = str(path.relative_to(OUT))
        summary["candidate_sha256"] = file_sha256(path)
        summary["test_predicted_multi"] = int((d_test.sum(axis=1) > 1).sum())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)




if __name__ == "__main__":
    from replay import main
    main(default_study='val_refit_weight')
