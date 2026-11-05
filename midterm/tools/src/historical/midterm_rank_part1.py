"""New Part 1 candidate studies; never replaces the locked final submission.

Thresholds use frozen gold out-of-fold probabilities. Selective pseudo labels
are recomputed inside every gold holdout, so held-out reasons never enter K-Means
mapping, scaling, or classifier fitting.
"""
from __future__ import annotations

from _historical_support import prepare_output

import argparse
import itertools
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from midterm_augmentation import logistic_parameters, observed_label_mask
from midterm_evaluation import REASONS, dataset_path, evaluate_part1, file_sha256
from midterm_next_round import verify_oof
from midterm_part1_finalization import build_submission, predict_bundle
from midterm_part2_improvement import validate_submission
from midterm_refinement import EMBEDDINGS, prepare_reason_context

STUDY = Path("refinement/rank_improvement_2026_11_02/part1")
BASE_THRESHOLD = .35
THRESHOLD_GRID = (.25, .30, .35, .40, .45)
PSEUDO_SETTINGS = tuple(
    {"positive_cutoff": pos, "negative_cutoff": neg, "pseudo_weight": weight}
    for pos, neg, weight in itertools.product((.50, .60), (.10, .20), (.03, .10))
)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def scores_from_saved(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["ID", *[f"prob_{r}" for r in REASONS]]].rename(
        columns={f"prob_{r}": r for r in REASONS}
    )


def load_gold_oof(root: Path):
    train = pd.read_csv(dataset_path(root, "part1", "train"))
    gold = train.loc[observed_label_mask(train[list(REASONS)].to_numpy())].reset_index(drop=True)
    assignment = pd.read_csv(root / 'models/part1/exploration/replay_inputs/oof_weight_char/fold_assignments.csv')
    assignment = assignment.set_index("ID").loc[gold.ID].reset_index()
    saved = pd.read_csv(root / 'models/part1/exploration/replay_inputs/oof_weight_char/oof_weight_010.csv')
    oof = verify_oof(gold, saved, assignment, part=1)
    return train, gold, assignment, oof


def choose_reason_thresholds(gold: pd.DataFrame, scores: pd.DataFrame):
    """Five one-dimensional OOF checks, with Funding fixed for low support.

    A reason changes only if its OOF F1 improves by at least .02. This is a
    simple predeclared guard against shifting a threshold on an exact tie.
    """
    thresholds = dict.fromkeys(REASONS, BASE_THRESHOLD)
    rows = []
    aligned = scores.set_index("ID").loc[gold.ID]
    for reason in REASONS:
        truth = gold[reason].to_numpy(dtype=int)
        base = f1_score(truth, aligned[reason].to_numpy() >= BASE_THRESHOLD, zero_division=0)
        candidates = []
        for threshold in THRESHOLD_GRID:
            value = f1_score(truth, aligned[reason].to_numpy() >= threshold, zero_division=0)
            candidates.append((threshold, value))
            rows.append({"reason": reason, "threshold": threshold, "oof_reason_f1": value,
                         "positive_support": int(truth.sum())})
        if reason != "Funding":
            best = sorted(candidates, key=lambda pair: (-pair[1], abs(pair[0] - BASE_THRESHOLD), pair[0]))[0]
            if best[1] >= base + .02:
                thresholds[reason] = best[0]
    return thresholds, pd.DataFrame(rows)


def threshold_study(root: Path, folder: Path):
    train, gold, assignment, oof = load_gold_oof(root)
    source = scores_from_saved(oof)
    selected, grid = choose_reason_thresholds(gold, source)
    control_oof = evaluate_part1(gold, source, BASE_THRESHOLD, empty_set_policy="argmax")
    candidate_oof = evaluate_part1(gold, source, selected, empty_set_policy="argmax")
    out = folder / "thresholds"
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "per_reason_oof_grid.csv", index=False)
    control_oof.save(out / "oof_control")
    candidate_oof.save(out / "oof_candidate")
    fold_rows = []
    for fold in sorted(assignment.fold.unique()):
        mask = assignment.fold.eq(fold).to_numpy()
        for name, rule in (("control", BASE_THRESHOLD), ("candidate", selected)):
            result = evaluate_part1(gold.loc[mask], source.loc[mask], rule, empty_set_policy="argmax")
            fold_rows.append({"fold": int(fold), "setting": name, **result.summary})
    pd.DataFrame(fold_rows).to_csv(out / "oof_by_fold.csv", index=False)
    write_json(out / "selection_locked_before_validation.json", {
        "thresholds": selected, "grid": THRESHOLD_GRID, "funding_fixed": True,
        "selection": "per-reason gold OOF F1, minimum +0.02; then sample F1 gate",
        "validation_labels_used": False, "test_labels_used": False,
    })
    validation = pd.read_csv(dataset_path(root, "part1", "val"))
    current = scores_from_saved(pd.read_csv(root / 'models/part1/02_kmeans_lr/evidence/selected/selected_validation/validation_predictions.csv'))
    model = joblib.load(root / 'models/part1/02_kmeans_lr/model.joblib')
    actual = pd.DataFrame(predict_bundle(model, validation), columns=REASONS)
    actual.insert(0, "ID", validation.ID.to_numpy())
    if not np.allclose(actual[list(REASONS)], current.set_index("ID").loc[validation.ID, list(REASONS)], atol=1e-12, rtol=0):
        raise ValueError("Frozen final model no longer matches saved validation probabilities.")
    control_val = evaluate_part1(validation, current, BASE_THRESHOLD, empty_set_policy="argmax")
    candidate_val = evaluate_part1(validation, current, selected, empty_set_policy="argmax")
    control_val.save(out / "validation_control")
    candidate_val.save(out / "validation_candidate")
    oof_gain = candidate_oof.summary["f1_samples"] - control_oof.summary["f1_samples"]
    val_gain = candidate_val.summary["f1_samples"] - control_val.summary["f1_samples"]
    passes = bool(selected != dict.fromkeys(REASONS, BASE_THRESHOLD) and oof_gain > 1e-12 and val_gain > 1e-12)
    summary = {"selected_thresholds": selected, "oof_control_f1": control_oof.summary["f1_samples"],
               "oof_candidate_f1": candidate_oof.summary["f1_samples"],
               "validation_control_f1": control_val.summary["f1_samples"],
               "validation_candidate_f1": candidate_val.summary["f1_samples"],
               "passes_local_gate": passes, "final_replaced": False, "kaggle_uploaded": False}
    if passes:
        test = pd.read_csv(dataset_path(root, "part1", "test"))
        prob = pd.DataFrame(predict_bundle(model, test), columns=REASONS)
        prob.insert(0, "ID", test.ID.to_numpy())
        from midterm_evaluation import part1_decisions
        pred = pd.DataFrame(part1_decisions(prob[list(REASONS)], selected, empty_set_policy="argmax"), columns=REASONS)
        pred.insert(0, "ID", test.ID.to_numpy())
        sample = pd.read_csv(dataset_path(root, "part1", "test").parent / "part1_sample_submission.csv")
        submission = build_submission(sample, pred)
        submission.to_csv(out / "part1_threshold_candidate.csv", index=False)
        validate_submission(submission, sample, 1)
        summary["candidate_submission"] = str((out / "part1_threshold_candidate.csv").relative_to(out))
    write_json(out / "summary.json", summary)
    return summary


def fit_selective(train: pd.DataFrame, context: dict, setting: dict):
    observed = context["observed"]
    q = context["pseudo_q"]
    x = context["x"]
    gold_y = train.loc[observed, list(REASONS)].to_numpy(dtype=int)
    pseudo_x = x[~observed]
    models, counts = [], []
    for j, reason in enumerate(REASONS):
        high = q[:, j] >= setting["positive_cutoff"]
        low = q[:, j] <= setting["negative_cutoff"]
        # Do not teach a rare class from pseudo negatives alone.
        if not high.any():
            low[:] = False
        selected = high | low
        target = np.concatenate([gold_y[:, j], high[selected].astype(int)])
        features = np.vstack([x[observed], pseudo_x[selected]])
        weights = np.concatenate([np.ones(len(gold_y)), np.full(int(selected.sum()), setting["pseudo_weight"])])
        if len(np.unique(target)) != 2:
            raise ValueError(f"No two observed classes for {reason}.")
        model = LogisticRegression(**{**logistic_parameters(42), "C": .1})
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(features, target, sample_weight=weights)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught) or model.n_iter_.max() >= model.max_iter:
            raise RuntimeError(f"Selective pseudo model did not converge for {reason}.")
        models.append(model)
        counts.append({"reason": reason, "pseudo_positive": int(high.sum()),
                       "pseudo_negative": int(low.sum()), "pseudo_used": int(selected.sum()),
                       "gold_positive": int(gold_y[:, j].sum())})
    return {"scaler": context["scaler"], "models": models, "setting": setting,
            "fit_ids": train.ID.tolist(), "gold_ids": train.loc[observed, "ID"].tolist()}, pd.DataFrame(counts)


def predict_selective(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    x = bundle["scaler"].transform(frame[EMBEDDINGS])
    values = np.column_stack([m.predict_proba(x)[:, 1] for m in bundle["models"]])
    result = pd.DataFrame(values, columns=REASONS)
    result.insert(0, "ID", frame.ID.to_numpy())
    return result


def pseudo_study(root: Path, folder: Path):
    train, gold, assignment, saved_control = load_gold_oof(root)
    out = folder / "selective_pseudo"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "experiment_plan.json", {
        "settings": PSEUDO_SETTINGS, "kmeans_clusters": 10, "lr_C": .1,
        "decision_threshold": BASE_THRESHOLD, "funding_or_other_reason_without_confident_positive": "gold_only",
        "selection": "pooled gold OOF sample F1", "folds": sorted(int(x) for x in assignment.fold.unique()),
        "validation_labels_used": False, "test_labels_used": False,
    })
    frames = {i: [] for i in range(len(PSEUDO_SETTINGS))}
    count_rows = []
    fold_rows = []
    for fold in sorted(assignment.fold.unique()):
        held_ids = set(assignment.loc[assignment.fold.eq(fold), "ID"])
        fit = train.loc[~train.ID.isin(held_ids)].reset_index(drop=True)
        held = gold.loc[gold.ID.isin(held_ids)].reset_index(drop=True)
        if set(fit.ID) & held_ids:
            raise RuntimeError("A held-out gold trial entered a pseudo fit.")
        context = prepare_reason_context(fit, 10)
        for i, setting in enumerate(PSEUDO_SETTINGS):
            bundle, counts = fit_selective(fit, context, setting)
            counts["setting_id"] = i
            counts["fold"] = int(fold)
            count_rows.append(counts)
            probabilities = predict_selective(bundle, held)
            result = evaluate_part1(held, probabilities, BASE_THRESHOLD, empty_set_policy="argmax")
            frames[i].append(result.predictions.assign(fold=int(fold)))
            fold_rows.append({"setting_id": i, "fold": int(fold), **result.summary})
        print(f"Part 1 selective pseudo fold {int(fold)+1}/4 complete", flush=True)
    pd.concat(count_rows).to_csv(out / "pseudo_counts_by_fold.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "oof_by_fold.csv", index=False)
    control_oof = evaluate_part1(gold, scores_from_saved(saved_control), BASE_THRESHOLD, empty_set_policy="argmax")
    results, rows = {}, [{"setting_id": "current", **control_oof.summary}]
    for i in frames:
        oof = verify_oof(gold, pd.concat(frames[i], ignore_index=True), assignment, part=1)
        oof.to_csv(out / f"oof_setting_{i}.csv", index=False)
        result = evaluate_part1(gold, scores_from_saved(oof), BASE_THRESHOLD, empty_set_policy="argmax")
        results[i] = result
        rows.append({"setting_id": i, **PSEUDO_SETTINGS[i], **result.summary})
    pd.DataFrame(rows).to_csv(out / "oof_comparison.csv", index=False)
    winner = max(results, key=lambda i: (results[i].summary["f1_samples"], -i))
    control_oof.save(out / "oof_control")
    results[winner].save(out / "oof_candidate")
    write_json(out / "selection_locked_before_validation.json", {
        "winner": winner, "setting": PSEUDO_SETTINGS[winner],
        "oof_f1": results[winner].summary["f1_samples"], "validation_labels_used": False,
    })
    full_context = prepare_reason_context(train, 10)
    full_bundle, full_counts = fit_selective(train, full_context, PSEUDO_SETTINGS[winner])
    joblib.dump(full_bundle, out / "selected_model.joblib", compress=3)
    full_counts.to_csv(out / "selected_full_training_pseudo_counts.csv", index=False)
    validation = pd.read_csv(dataset_path(root, "part1", "val"))
    current = scores_from_saved(pd.read_csv(root / 'models/part1/02_kmeans_lr/evidence/selected/selected_validation/validation_predictions.csv'))
    control_val = evaluate_part1(validation, current, BASE_THRESHOLD, empty_set_policy="argmax")
    candidate_val = evaluate_part1(validation, predict_selective(full_bundle, validation), BASE_THRESHOLD, empty_set_policy="argmax")
    control_val.save(out / "validation_control")
    candidate_val.save(out / "validation_candidate")
    oof_gain = results[winner].summary["f1_samples"] - control_oof.summary["f1_samples"]
    val_gain = candidate_val.summary["f1_samples"] - control_val.summary["f1_samples"]
    passes = bool(oof_gain > 1e-12 and val_gain > 1e-12)
    summary = {"winner": winner, "setting": PSEUDO_SETTINGS[winner],
               "oof_control_f1": control_oof.summary["f1_samples"],
               "oof_candidate_f1": results[winner].summary["f1_samples"],
               "validation_control_f1": control_val.summary["f1_samples"],
               "validation_candidate_f1": candidate_val.summary["f1_samples"],
               "passes_local_gate": passes, "final_replaced": False, "kaggle_uploaded": False}
    if passes:
        from midterm_evaluation import part1_decisions
        test = pd.read_csv(dataset_path(root, "part1", "test"))
        prob = predict_selective(full_bundle, test)
        pred = pd.DataFrame(part1_decisions(prob[list(REASONS)], BASE_THRESHOLD, empty_set_policy="argmax"), columns=REASONS)
        pred.insert(0, "ID", test.ID.to_numpy())
        sample = pd.read_csv(dataset_path(root, "part1", "test").parent / "part1_sample_submission.csv")
        submission = build_submission(sample, pred)
        submission.to_csv(out / "part1_selective_pseudo_candidate.csv", index=False)
        validate_submission(submission, sample, 1)
        summary["candidate_submission"] = str((out / "part1_selective_pseudo_candidate.csv").relative_to(out))
    write_json(out / "summary.json", summary)
    return summary


def run(root, output, study="all"):
    root, folder = prepare_output(root, output)
    frozen = {name: file_sha256(root / name) for name in (
        'tools/archive/records/final_selection_nov01.json', 'models/part1/02_kmeans_lr/model.joblib',
        'models/part1/02_kmeans_lr/submission.csv', 'models/part2/02_text_blend/submission.csv')}
    summary_path = folder / "summary.json"
    summaries = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if study in ("thresholds", "all"):
        summaries["thresholds"] = threshold_study(root, folder)
    if study in ("pseudo", "all"):
        summaries["selective_pseudo"] = pseudo_study(root, folder)
    for name, digest in frozen.items():
        if file_sha256(root / name) != digest:
            raise RuntimeError(f"A locked final artifact changed: {name}")
    write_json(summary_path, summaries)
    print(json.dumps(summaries, indent=2), flush=True)




if __name__ == "__main__":
    from replay import main
    main(default_study='selective_pseudo')
