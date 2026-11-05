"""Training-only OOF comparisons for pseudo-label weights and character text.

Run explicitly with --allow-training once. Later calls verify and reuse the
completed study. Current final models, submissions, and delivery files are
protected by checksums throughout this separate candidate study.
"""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from midterm_augmentation import logistic_parameters, observed_label_mask
from midterm_evaluation import (
    PHASES, REASONS, dataset_path, evaluate_part1, evaluate_part2,
    file_sha256, part1_decisions,
)
from midterm_finalization import blend_probabilities, predict_final_outcomes
from midterm_part1_finalization import build_submission
from midterm_part2_baseline import fit_phase_baseline, predict_outcome_probabilities
from midterm_part2_improvement import validate_submission
from midterm_refinement import (
    EMBEDDINGS, TFIDF, fit_field_outcomes, predict_field_outcomes,
    predict_reasons, prepare_reason_context,
)

STUDY = "refinement/oof_weight_char"
P1_WEIGHTS = {"weight_000": 0.0, "weight_003": 0.03, "weight_010": 0.1}
P1_CONTROL = "weight_010"
P2_CONTROL = "current_equal_blend"
P2_NEW = ("character", "current_plus_character")
CHAR_TFIDF = {
    "analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 3,
    "max_features": 30000, "lowercase": True, "strip_accents": "unicode",
    "sublinear_tf": True, "smooth_idf": True, "use_idf": True, "norm": "l2",
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def directory_hashes(directory):
    directory = Path(directory)
    return {p.relative_to(directory).as_posix(): file_sha256(p)
            for p in sorted(directory.rglob("*")) if p.is_file()
            and "__pycache__" not in p.parts and p.name != "completed_checksums.json"}


def verify_hashes(directory, hashes):
    directory = Path(directory)
    for relative, digest in hashes.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe artifact path in checksum manifest.")
        if not (directory / path).is_file() or file_sha256(directory / path) != digest:
            from midterm_history import generated_archive_exception, resolve_expected_path
            if generated_archive_exception(directory,relative,digest):
                continue
            try:
                resolve_expected_path(directory,relative,digest)
            except (AssertionError,FileNotFoundError) as error:
                raise ValueError(f"Protected or completed artifact changed: {relative}") from error


def protected_hashes(root):
    """Cover current finals, registered runs, old studies, and hand-in files."""
    folders = ['models/part1/02_kmeans_lr/evidence/selected', 'models/part2/02_text_blend/evidence/selection', 'models', "submissions",
               "deliverables", "report", "refinement"]
    files = [root / 'tools/archive/records/final_selection_nov01.json']
    study = root / STUDY
    for folder in folders:
        files.extend(p for p in (root / folder).rglob("*")
                     if p.is_file() and not p.is_relative_to(study)
                     and "__pycache__" not in p.parts)
    return {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(set(files))}


def validate_fold_scope(fit, held, *, part):
    """Assert held-out identities are absent before any estimator is fitted."""
    if set(fit.ID) & set(held.ID):
        raise ValueError("Held-out row IDs overlap fitting data.")
    if part == 2 and set(fit.trial_id) & set(held.trial_id):
        raise ValueError("A trial crosses a training/held-out fold boundary.")
    return {
        "fit_row_ids": fit.ID.tolist(), "held_out_row_ids": held.ID.tolist(),
        "fit_trial_ids": sorted(fit.trial_id.unique()) if part == 2 else fit.ID.tolist(),
        "held_out_trial_ids": sorted(held.trial_id.unique()) if part == 2 else held.ID.tolist(),
        "row_overlap": 0, "trial_overlap": 0,
        "all_transforms_fitted_after_exclusion": True,
    }


def verify_oof(reference, predictions, assignment, *, part):
    """Require exact coverage, original labels, and one assigned fold per row."""
    if not predictions.ID.is_unique or set(predictions.ID) != set(reference.ID):
        raise ValueError("OOF predictions must cover every reference row exactly once.")
    if not assignment.ID.is_unique or set(assignment.ID) != set(reference.ID):
        raise ValueError("Fold assignments must cover exactly the reference rows.")
    ordered = predictions.set_index("ID").loc[reference.ID].reset_index()
    expected = assignment.set_index("ID").loc[reference.ID, "fold"].to_numpy()
    if not np.array_equal(ordered.fold.to_numpy(), expected):
        raise ValueError("OOF predictions have incorrect fold assignments.")
    if part == 1:
        truth = ordered[[f"true_{r}" for r in REASONS]].to_numpy()
        if not np.array_equal(truth, reference[list(REASONS)].to_numpy()):
            raise ValueError("OOF reason labels differ from the original gold labels.")
        scores = ordered[[f"prob_{r}" for r in REASONS]].to_numpy()
    else:
        if not np.array_equal(ordered.label.to_numpy(), reference.label.to_numpy()):
            raise ValueError("OOF outcome labels changed.")
        if not np.array_equal(ordered.trial_id, reference.trial_id) or not np.array_equal(ordered.phase, reference.phase):
            raise ValueError("OOF trial or phase identities changed.")
        if ordered.groupby("trial_id").fold.nunique().gt(1).any():
            raise ValueError("Combined-phase trial rows must stay in the same fold.")
        scores = ordered.p_success.to_numpy()
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("OOF probabilities must be finite and inside [0, 1].")
    return ordered


def validate_grouped_assignment(train, assignment):
    """Use saved folds as authoritative across sklearn splitter versions."""
    if not assignment.ID.is_unique or set(assignment.ID) != set(train.ID):
        raise ValueError("Saved grouped folds do not cover the training rows exactly.")
    assignment = assignment.set_index("ID").loc[train.ID].reset_index()
    pd.testing.assert_frame_equal(assignment[["ID", "trial_id", "phase", "label"]],
                                  train[["ID", "trial_id", "phase", "label"]].reset_index(drop=True))
    if set(assignment.fold) != set(range(5)) or assignment.groupby("trial_id").fold.nunique().gt(1).any():
        raise ValueError("Saved folds must keep each trial in exactly one of five folds.")
    for fold in range(5):
        for mask in [assignment.fold.eq(fold), ~assignment.fold.eq(fold)]:
            subset = assignment.loc[mask]
            if set(subset.phase) != set(PHASES) or not subset.groupby("phase").label.nunique().eq(2).all():
                raise ValueError("Every training and held-out phase requires both outcomes.")
    return assignment


def reason_error_counts(evaluation):
    rows = []
    for reason in REASONS:
        y = evaluation.predictions[f"true_{reason}"].to_numpy()
        p = evaluation.predictions[f"pred_{reason}"].to_numpy()
        rows.append({"reason": reason, "tp": int(((y == 1) & (p == 1)).sum()),
                     "fp": int(((y == 0) & (p == 1)).sum()),
                     "fn": int(((y == 1) & (p == 0)).sum())})
    return evaluation.by_group.merge(pd.DataFrame(rows), on="reason", validate="one_to_one")


def fit_reason_weight(frame, context, weight):
    """Use identical hard K-Means pseudo targets; vary only their row weight."""
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("Pseudo-label weight must be finite and nonnegative.")
    observed = context["observed"]
    x = context["x"]
    gold = frame.loc[observed, list(REASONS)].to_numpy(dtype=int)
    pseudo = (context["pseudo_q"] >= .5).astype(int)
    if weight == 0:
        train_x, train_y = x[observed], gold
        weights = np.ones(len(gold))
    else:
        train_x = x
        train_y = frame[list(REASONS)].to_numpy(dtype=int).copy()
        train_y[~observed] = pseudo
        weights = np.where(observed, 1.0, weight)
    models = []
    for index in range(len(REASONS)):
        model = LogisticRegression(**{**logistic_parameters(42), "C": .1})
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter("always")
            model.fit(train_x, train_y[:, index], sample_weight=weights)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught) or model.n_iter_.max() >= 500:
            raise RuntimeError("A reason classifier did not converge.")
        models.append(model)
    return {"kind": "kmeans_weight_ablation", "pseudo_weight": weight,
            "scaler": context["scaler"], "clusterer": context["clusterer"],
            "mapping": context["mapping"], "models": models, "threshold": .35,
            "empty_set_policy": "argmax", "fit_ids": frame.ID.tolist(),
            "seed_ids": frame.loc[observed, "ID"].tolist()}


def p1_assignment(train):
    observed = observed_label_mask(train[list(REASONS)].to_numpy())
    gold = train.loc[observed].reset_index(drop=True)
    strata = gold[list(REASONS)].to_numpy().argmax(axis=1)
    if np.bincount(strata).min() < 4:
        raise ValueError("Four-fold first-positive stratification lacks support.")
    folds = np.full(len(gold), -1)
    for fold, (_, held) in enumerate(StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(gold, strata)):
        folds[held] = fold
    return gold, gold[["ID"]].assign(fold=folds)


def finish_cache(directory):
    write_json(directory / "completed_checksums.json", directory_hashes(directory))


def cache_complete(directory):
    manifest = directory / "completed_checksums.json"
    if not manifest.exists():
        return False
    verify_hashes(directory, read_json(manifest))
    return True


def run_p1_oof(root, folder, train):
    gold, assignment = p1_assignment(train)
    assignment.to_csv(folder / "fold_assignments.csv", index=False)
    fold_metrics, frames = [], {key: [] for key in P1_WEIGHTS}
    for fold in range(4):
        cache = folder / "fold_cache" / f"fold_{fold}"
        if not cache_complete(cache):
            cache.mkdir(parents=True, exist_ok=True)
            held_ids = set(assignment.loc[assignment.fold.eq(fold), "ID"])
            fit = train.loc[~train.ID.isin(held_ids)].reset_index(drop=True)
            held = gold.loc[gold.ID.isin(held_ids)].reset_index(drop=True)
            scope = validate_fold_scope(fit, held, part=1)
            scope["label_seed_ids"] = fit.loc[observed_label_mask(fit[list(REASONS)].to_numpy()), "ID"].tolist()
            context = prepare_reason_context(fit, 10)
            if set(context["seed_ids"]) & held_ids or set(context["fit_ids"]) & held_ids:
                raise AssertionError("Held-out rows entered scaling, clustering, or mapping.")
            write_json(cache / "fit_scope.json", scope)
            for setting, weight in P1_WEIGHTS.items():
                bundle = fit_reason_weight(fit, context, weight)
                result = evaluate_part1(held, predict_reasons(bundle, held), .35, empty_set_policy="argmax")
                result.predictions.assign(fold=fold).to_csv(cache / f"{setting}.csv", index=False)
                write_json(cache / f"{setting}_metrics.json", result.summary)
            finish_cache(cache)
        for setting in P1_WEIGHTS:
            frames[setting].append(pd.read_csv(cache / f"{setting}.csv"))
            fold_metrics.append({"setting": setting, "fold": fold, **read_json(cache / f"{setting}_metrics.json")})
        print(f"Part 1 OOF fold {fold + 1}/4 complete", flush=True)
    fold_table = pd.DataFrame(fold_metrics)
    fold_table.to_csv(folder / "fold_metrics.csv", index=False)
    comparisons, by_reason, evaluations = [], [], {}
    for setting, weight in P1_WEIGHTS.items():
        oof = verify_oof(gold, pd.concat(frames[setting], ignore_index=True), assignment, part=1)
        scores = oof[["ID", *[f"prob_{r}" for r in REASONS]]].rename(columns={f"prob_{r}": r for r in REASONS})
        evaluation = evaluate_part1(gold, scores, .35, empty_set_policy="argmax")
        evaluations[setting] = evaluation
        oof.to_csv(folder / f"oof_{setting}.csv", index=False)
        selected_folds = fold_table.loc[fold_table.setting.eq(setting)]
        comparisons.append({"setting": setting, "pseudo_weight": weight, **evaluation.summary,
                            "mean_fold_f1": selected_folds.f1_samples.mean(),
                            "std_fold_f1": selected_folds.f1_samples.std()})
        by_reason.append(reason_error_counts(evaluation).assign(setting=setting))
    table = pd.DataFrame(comparisons)
    table["delta_f1_vs_control"] = table.f1_samples - evaluations[P1_CONTROL].summary["f1_samples"]
    table.to_csv(folder / "oof_comparison.csv", index=False)
    pd.concat(by_reason).to_csv(folder / "oof_by_reason.csv", index=False)
    # Ties preserve the incumbent, then prefer the smaller change in weight.
    winner = max(P1_WEIGHTS, key=lambda s: (evaluations[s].summary["f1_samples"], P1_WEIGHTS[s]))
    lock = {"setting": winner, "pseudo_weight": P1_WEIGHTS[winner], "metric": "pooled_OOF_sample_F1",
            "score": evaluations[winner].summary["f1_samples"],
            "validation_labels_used": False, "test_predictions_used": False,
            "folds": 4, "tie_policy": "Prefer the larger weight, including the incumbent 0.1."}
    write_json(folder / "selection_locked_before_validation.json", lock)
    return lock, evaluations


def fit_character(train):
    return fit_phase_baseline(train, tfidf_parameters=CHAR_TFIDF, classifier_parameters={"C": 1.0}, seed=42)


def p2_fold_predictions(fit, held):
    """Fit all three representations inside this grouped training fold."""
    concat, _, _, _ = fit_phase_baseline(
        fit, tfidf_parameters={**TFIDF, "max_features": 20000}, classifier_parameters={"C": 1.0}, seed=42)
    field, _ = fit_field_outcomes(fit)
    character, diagnostics, fit_ids, caught = fit_character(fit)
    parts = {"concatenated_word": predict_outcome_probabilities(concat, held),
             "separate_field_word": predict_field_outcomes(field, held),
             "character": predict_outcome_probabilities(character, held)}
    parts[P2_CONTROL] = blend_probabilities(held, [parts["concatenated_word"], parts["separate_field_word"]], [.5, .5])
    parts["current_plus_character"] = blend_probabilities(held, [parts[P2_CONTROL], parts["character"]], [.5, .5])
    return parts, diagnostics, {"character_fit_ids": fit_ids, "character_warnings": caught}


def run_p2_oof(root, folder, train):
    assignment = validate_grouped_assignment(train, pd.read_csv(root / 'data/audit/protocol/part2_train_folds.csv'))
    assignment.to_csv(folder / "fold_assignments.csv", index=False)
    settings = ("concatenated_word", "separate_field_word", P2_CONTROL, *P2_NEW)
    frames, metrics, phase_metrics = {s: [] for s in settings}, [], []
    for fold in range(5):
        cache = folder / "fold_cache" / f"fold_{fold}"
        if not cache_complete(cache):
            cache.mkdir(parents=True, exist_ok=True)
            held_ids = set(assignment.loc[assignment.fold.eq(fold), "ID"])
            fit = train.loc[~train.ID.isin(held_ids)].reset_index(drop=True)
            held = train.loc[train.ID.isin(held_ids)].reset_index(drop=True)
            scope = validate_fold_scope(fit, held, part=2)
            scope["feature_columns"] = ["criteria", "diseases_cleaned", "drugs"]
            scope["reason_features_used"] = False
            parts, diagnostics, fit_metadata = p2_fold_predictions(fit, held)
            write_json(cache / "fit_scope.json", {**scope, **fit_metadata})
            diagnostics.to_csv(cache / "character_fit_diagnostics.csv", index=False)
            for setting, probabilities in parts.items():
                result = evaluate_part2(held, probabilities, .5)
                result.predictions.assign(fold=fold).to_csv(cache / f"{setting}.csv", index=False)
                write_json(cache / f"{setting}_metrics.json", result.summary)
                result.by_group.to_csv(cache / f"{setting}_by_phase.csv", index=False)
            finish_cache(cache)
        for setting in settings:
            frames[setting].append(pd.read_csv(cache / f"{setting}.csv"))
            metrics.append({"setting": setting, "fold": fold, **read_json(cache / f"{setting}_metrics.json")})
            phase_metrics.append(pd.read_csv(cache / f"{setting}_by_phase.csv").assign(setting=setting, fold=fold))
        print(f"Part 2 grouped OOF fold {fold + 1}/5 complete", flush=True)
    fold_table = pd.DataFrame(metrics)
    fold_table.to_csv(folder / "fold_metrics.csv", index=False)
    pd.concat(phase_metrics).to_csv(folder / "fold_metrics_by_phase.csv", index=False)
    evaluations, comparisons, by_phase = {}, [], []
    for setting in settings:
        oof = verify_oof(train, pd.concat(frames[setting], ignore_index=True), assignment, part=2)
        result = evaluate_part2(train, oof[["ID", "p_success"]], .5)
        evaluations[setting] = result
        oof.to_csv(folder / f"oof_{setting}.csv", index=False)
        selected_folds = fold_table.loc[fold_table.setting.eq(setting)]
        comparisons.append({"setting": setting, **result.summary,
                            "mean_fold_log_loss": selected_folds.log_loss_equal_phase.mean(),
                            "std_fold_log_loss": selected_folds.log_loss_equal_phase.std()})
        by_phase.append(result.by_group.assign(setting=setting))
    table = pd.DataFrame(comparisons)
    table["delta_log_loss_vs_control"] = table.log_loss_equal_phase - evaluations[P2_CONTROL].summary["log_loss_equal_phase"]
    table.to_csv(folder / "oof_comparison.csv", index=False)
    pd.concat(by_phase).to_csv(folder / "oof_by_phase.csv", index=False)
    # Word-only components are diagnostics; new candidates are char or the fixed blend.
    eligible = (P2_CONTROL, *P2_NEW)
    winner = min(eligible, key=lambda s: (evaluations[s].summary["log_loss_equal_phase"], eligible.index(s)))
    lock = {"setting": winner, "metric": "equal_phase_log_loss_from_all_OOF_rows",
            "score": evaluations[winner].summary["log_loss_equal_phase"],
            "eligible_settings": list(eligible), "validation_labels_used": False,
            "test_predictions_used": False, "folds": 5,
            "character_weight_in_new_blend": .5, "blend_weight_search": False}
    write_json(folder / "selection_locked_before_validation.json", lock)
    return lock, evaluations


def predict_candidate_outcomes(bundle, frame):
    """Loadable prediction entry point for this study's outcome candidates."""
    if bundle.get("kind") == "character_only":
        return predict_outcome_probabilities(bundle["character"], frame)
    if bundle.get("kind") == "current_plus_character" and bundle.get("weights") == [.5, .5]:
        return blend_probabilities(frame, [predict_final_outcomes(bundle["current"], frame),
            predict_outcome_probabilities(bundle["character"], frame)], [.5, .5])
    raise ValueError("Unexpected candidate outcome bundle.")


def row_f1(result):
    y = result.predictions[[f"true_{r}" for r in REASONS]].to_numpy()
    d = result.predictions[[f"pred_{r}" for r in REASONS]].to_numpy()
    return 2 * (y * d).sum(axis=1) / (y.sum(axis=1) + d.sum(axis=1))


def paired_diagnostics(before, after, *, part):
    """Describe paired differences; intervals exclude selection uncertainty."""
    left, right = before.predictions, after.predictions
    if not np.array_equal(left.ID, right.ID):
        raise ValueError("Paired diagnostics require aligned IDs.")
    rng = np.random.default_rng(42)
    if part == 1:
        delta = row_f1(after) - row_f1(before)
        draws = delta[rng.integers(0, len(delta), size=(2000, len(delta)))].mean(axis=1)
        value = float(delta.mean())
        details = {"rows_improved": int((delta > 1e-12).sum()), "rows_worsened": int((delta < -1e-12).sum())}
    else:
        y = left.label.to_numpy()
        p0, p1 = [np.clip(frame.p_success.to_numpy(), 1e-15, 1 - 1e-15) for frame in [left, right]]
        loss0 = -(y * np.log(p0) + (1 - y) * np.log1p(-p0))
        loss1 = -(y * np.log(p1) + (1 - y) * np.log1p(-p1))
        delta = loss1 - loss0
        ids = sorted(left.trial_id.unique())
        sums = np.zeros((len(ids), 3)); counts = np.zeros_like(sums)
        trial_index = {trial: i for i, trial in enumerate(ids)}
        for i, phase in enumerate(PHASES):
            subset = np.flatnonzero(left.phase.to_numpy() == phase)
            positions = np.array([trial_index[t] for t in left.iloc[subset].trial_id])
            np.add.at(sums[:, i], positions, delta[subset])
            np.add.at(counts[:, i], positions, 1)
        draws = []
        for _ in range(2000):
            sampled = rng.integers(0, len(ids), size=len(ids))
            denominator = counts[sampled].sum(axis=0)
            if (denominator > 0).all():
                draws.append(float((sums[sampled].sum(axis=0) / denominator).mean()))
        value = after.summary["log_loss_equal_phase"] - before.summary["log_loss_equal_phase"]
        details = {"bootstrap_unit": "trial_id, preserving all phase rows"}
    return {"delta": value, "descriptive_95_percentile_interval": np.quantile(draws, [.025, .975]).tolist(),
            "bootstrap_draws": 2000, **details,
            "limitation": "Descriptive only; repeated validation use, model selection, and nonrandom label sampling are not accounted for."}


def full_p1(root, folder, train, lock, oof):
    validation = pd.read_csv(dataset_path(root, "part1", "val"))
    current = pd.read_csv(root / 'models/part1/02_kmeans_lr/evidence/selected/selected_validation/validation_predictions.csv')
    current_scores = current[["ID", *[f"prob_{r}" for r in REASONS]]].rename(columns={f"prob_{r}": r for r in REASONS})
    control = evaluate_part1(validation, current_scores, .35, empty_set_policy="argmax")
    context = prepare_reason_context(train, 10)
    pseudo = train.loc[~context["observed"], ["ID"]].copy()
    q = context["pseudo_q"]
    for i, reason in enumerate(REASONS):
        pseudo[f"cluster_rate_{reason}"] = q[:, i]
        pseudo[f"pseudo_{reason}"] = (q[:, i] >= .5).astype(int)
    pseudo["source"] = "training_only_kmeans_cluster_rate_threshold_0.5"
    pseudo.to_csv(folder / "all_masked_pseudo_labels.csv", index=False)
    pd.DataFrame({"reason": REASONS, "gold_positive_rows": train.loc[context["observed"], list(REASONS)].sum().to_numpy(),
                  "pseudo_positive_rows": (q >= .5).sum(axis=0), "masked_rows": len(q)}).to_csv(folder / "pseudo_label_support.csv", index=False)
    control_refit = fit_reason_weight(train, context, .1)
    check = predict_reasons(control_refit, validation)
    error = float(np.abs(check[list(REASONS)].to_numpy() - current_scores.set_index("ID").loc[validation.ID, list(REASONS)].to_numpy()).max())
    if error > 1e-6:
        raise ValueError(f"The weight-0.1 control does not reproduce the retained model: {error}")
    bundle = control_refit if lock["setting"] == P1_CONTROL else fit_reason_weight(train, context, lock["pseudo_weight"])
    candidate = evaluate_part1(validation, predict_reasons(bundle, validation), .35, empty_set_policy="argmax")
    joblib.dump(bundle, folder / "oof_selected_model.joblib", compress=3)
    control.save(folder / "validation" / P1_CONTROL)
    if lock["setting"] != P1_CONTROL:
        candidate.save(folder / "validation" / lock["setting"])
    rows = [{"setting": P1_CONTROL, **control.summary}]
    if lock["setting"] != P1_CONTROL:
        rows.append({"setting": lock["setting"], **candidate.summary})
    pd.DataFrame(rows).to_csv(folder / "validation_comparison.csv", index=False)
    pd.concat([reason_error_counts(control).assign(setting=P1_CONTROL),
               reason_error_counts(candidate).assign(setting=lock["setting"], role="OOF_selected")]).to_csv(folder / "validation_by_reason.csv", index=False)
    oof_gain = oof[lock["setting"]].summary["f1_samples"] - oof[P1_CONTROL].summary["f1_samples"]
    val_gain = candidate.summary["f1_samples"] - control.summary["f1_samples"]
    gate = {"oof_selected_setting": lock["setting"], "oof_f1_gain": oof_gain,
            "validation_f1_gain": val_gain, "control_refit_max_probability_difference": error,
            "passes_candidate_gate": bool(lock["setting"] != P1_CONTROL and oof_gain > 1e-12 and val_gain > 1e-12),
            "final_model_replaced": False, "public_score_known": False}
    write_json(folder / "paired_oof_diagnostics.json", paired_diagnostics(oof[P1_CONTROL], oof[lock["setting"]], part=1))
    write_json(folder / "paired_validation_diagnostics.json", paired_diagnostics(control, candidate, part=1))
    write_json(folder / "candidate_gate.json", gate)
    return gate, bundle


def full_p2(root, folder, train, lock, oof):
    validation = pd.read_csv(dataset_path(root, "part2", "val"))
    word_run = root / 'results/experiments/runs/part2_equal_probability_blend_seed42_v1'
    current_bundle = joblib.load(word_run / "model.joblib")
    current_prob = predict_final_outcomes(current_bundle, validation)
    saved = pd.read_csv(word_run / "validation_predictions.csv")
    error = float(np.abs(current_prob.p_success.to_numpy() - saved.set_index("ID").loc[validation.ID, "p_success"].to_numpy()).max())
    if error > 1e-12:
        raise ValueError("The current outcome model does not reproduce its saved validation predictions.")
    character, diagnostics, fit_ids, caught = fit_character(train)
    diagnostics.to_csv(folder / "full_character_fit_diagnostics.csv", index=False)
    write_json(folder / "full_character_fit_scope.json", {"fit_ids": fit_ids, "warnings": caught,
               "reason_features_used": False, "validation_rows_in_fit": 0})
    char_bundle = {"kind": "character_only", "character": character, "reason_features_used": False}
    blend_bundle = {"kind": "current_plus_character", "current": current_bundle,
                    "character": character, "weights": [.5, .5], "reason_features_used": False}
    bundles = {"character": char_bundle, "current_plus_character": blend_bundle}
    joblib.dump(char_bundle, folder / "character_model.joblib", compress=3)
    probabilities = {P2_CONTROL: current_prob, **{s: predict_candidate_outcomes(b, validation) for s, b in bundles.items()}}
    evaluations = {s: evaluate_part2(validation, p, .5) for s, p in probabilities.items()}
    for setting, result in evaluations.items():
        result.save(folder / "validation" / setting)
    pd.DataFrame([{"setting": s, **r.summary} for s, r in evaluations.items()]).to_csv(folder / "validation_comparison.csv", index=False)
    pd.concat([r.by_group.assign(setting=s) for s, r in evaluations.items()]).to_csv(folder / "validation_by_phase.csv", index=False)
    control, candidate = evaluations[P2_CONTROL], evaluations[lock["setting"]]
    oof_gain = oof[lock["setting"]].summary["log_loss_equal_phase"] - oof[P2_CONTROL].summary["log_loss_equal_phase"]
    val_gain = candidate.summary["log_loss_equal_phase"] - control.summary["log_loss_equal_phase"]
    gate = {"oof_selected_setting": lock["setting"], "oof_log_loss_delta": oof_gain,
            "validation_log_loss_delta": val_gain, "current_model_max_probability_difference": error,
            "passes_candidate_gate": bool(lock["setting"] != P2_CONTROL and oof_gain < -1e-12 and val_gain < -1e-12),
            "final_model_replaced": False, "public_score_known": False,
            "validation_policy": "All three predeclared settings are reported; only the training-OOF winner can pass the gate."}
    if lock["setting"] != P2_CONTROL:
        joblib.dump(bundles[lock["setting"]], folder / "oof_selected_model.joblib", compress=3)
    write_json(folder / "paired_oof_diagnostics.json", paired_diagnostics(oof[P2_CONTROL], oof[lock["setting"]], part=2))
    write_json(folder / "paired_validation_diagnostics.json", paired_diagnostics(control, candidate, part=2))
    # Reasons are joined after predicting, using the unchanged Part 1 lookup.
    from midterm_part2_baseline import analyze_failure_reasons
    lookup = pd.read_csv(root / 'models/part1/02_kmeans_lr/evidence/selected/reason_lookup.csv')
    before = analyze_failure_reasons(control.predictions, lookup)
    after = analyze_failure_reasons(candidate.predictions, lookup)
    for key in ["reason_rates", "source_rates", "overall_failed_rates"]:
        on = ["phase", "reason_group"] + (["source_type"] if key == "source_rates" else [])
        joined = before[key].merge(after[key], on=on, suffixes=("_current", "_candidate"), validate="one_to_one")
        if not np.array_equal(joined.n_trials_current, joined.n_trials_candidate):
            raise AssertionError("Failure-reason group support changed.")
        joined["delta_false_success_rate"] = joined.false_success_rate_candidate - joined.false_success_rate_current
        joined.to_csv(folder / f"validation_{key}_comparison.csv", index=False)
    write_json(folder / "candidate_gate.json", gate)
    return gate, bundles.get(lock["setting"])


def candidate_submissions(root, folder, gates, bundles):
    output = folder / "submissions"
    output.mkdir(exist_ok=True)
    paths = {}
    for task, gate in gates.items():
        if not gate["passes_candidate_gate"]:
            continue
        test = pd.read_csv(dataset_path(root, task, "test"))
        part = 1 if task == "part1" else 2
        sample = pd.read_csv(dataset_path(root, task, "test").parent / f"{task}_sample_submission.csv")
        if part == 1:
            scores = predict_reasons(bundles[task], test)
            decisions = pd.DataFrame(part1_decisions(scores[list(REASONS)], .35, empty_set_policy="argmax"), columns=REASONS)
            decisions.insert(0, "ID", test.ID.to_numpy())
            submission = build_submission(sample, decisions)
        else:
            scores = predict_candidate_outcomes(bundles[task], test)
            submission = sample[["ID"]].merge(scores, on="ID", validate="one_to_one", sort=False).rename(columns={"p_success": "label"})
        scores.to_csv(output / f"{task}_candidate_probabilities.csv", index=False)
        path = output / f"{task}_candidate.csv"
        submission.to_csv(path, index=False)
        validate_submission(pd.read_csv(path), sample, part)
        paths[task] = path.relative_to(root).as_posix()
    return paths


def make_report(folder, summary):
    """Create English tables and a standalone plot from persisted study results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, task, metric, title in zip(axes, ["part1", "part2"], ["f1_samples", "log_loss_equal_phase"],
                                     ["Part 1: training OOF sample F1 (higher is better)", "Part 2: grouped OOF log loss (lower is better)"]):
        table = pd.read_csv(folder / task / "oof_comparison.csv")
        colors = ["#007f8b" if s == summary[task]["oof_selected_setting"] else "#a4b9c7" for s in table.setting]
        bars = ax.barh(table.setting.str.replace("_", " "), table[metric], color=colors)
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
        ax.set_title(title, fontsize=11); ax.set_xlim(0, float(table[metric].max()) * 1.18)
        ax.invert_yaxis(); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(folder / "oof_comparison.png", dpi=180)
    plt.close(fig)
    sections = []
    for task in ["part1", "part2"]:
        sections.append(f"<h2>{html.escape(task.title())}</h2>")
        for name in ["oof_comparison", "validation_comparison", "validation_by_reason" if task == "part1" else "validation_by_phase"]:
            sections.append(f"<h3>{html.escape(name.replace('_', ' ').title())}</h3>" + pd.read_csv(folder / task / f"{name}.csv").to_html(index=False, float_format=lambda v: f"{v:.6f}"))
        sections.append("<pre>" + html.escape(json.dumps(summary[task], indent=2)) + "</pre>")
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><title>OOF Weight and Character Study</title>
<style>body{font:16px system-ui;margin:36px;max-width:1350px;color:#203047}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #cbd5df}th{background:#edf2f6}pre{background:#f3f5f7;padding:16px;white-space:pre-wrap}img{max-width:100%}</style>
<h1>OOF Weight and Character Study</h1>
<p>All settings were fixed before this study's validation evaluation. Selection uses training-only out-of-fold predictions. Current final artifacts remain unchanged.</p>
<p>Part 1 uses 157 observed rows in four folds; held-out rows are excluded from scaling, K-Means, and label mapping. All 1,014 originally masked rows remain unlabeled inputs. Part 2 uses the original five folds grouped by trial_id, with vocabularies fitted inside each training fold. Reasons are used only after prediction for error analysis.</p>
<p>The Part 1 observed labels are nonrandom, and official validation and public scores have been consulted in earlier work. OOF selection and descriptive intervals do not remove these limitations. These candidates have no known public or private score.</p>
<img src="oof_comparison.png" alt="Training-only OOF comparisons">""" + "".join(sections)
    page += "<h2>Candidate submissions</h2><pre>" + html.escape(json.dumps(summary["candidate_submissions"], indent=2)) + "</pre>"
    page += "<p>A candidate CSV is generated only when the training-OOF winner also improves the predeclared validation metric. No model is automatically promoted and no submission is uploaded. The existing report and delivery ZIP remain the prior completed snapshot.</p></html>"
    (folder / "study_report.html").write_text(page)


def run_study(root, *, allow_training=False):
    root = Path(root).resolve()
    folder = root / STUDY
    if cache_complete(folder):
        verify_hashes(root, read_json(folder / "protected_artifacts.json"))
        verify_hashes(root, read_json(folder / "input_and_source_hashes.json"))
        return read_json(folder / "summary.json")
    if not allow_training:
        raise RuntimeError("The study is missing or incomplete. Run midterm_next_round.py --allow-training to fit the predeclared experiments.")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    folder.mkdir(parents=True, exist_ok=True)
    for name in ["part1", "part2"]:
        (folder / name).mkdir(exist_ok=True)
    plan = {
        "study": STUDY, "seed": 42, "part1_weights": P1_WEIGHTS,
        "part1_C": .1, "part1_threshold": .35, "empty_set_policy": "argmax",
        "part1_kmeans_clusters": 10, "part1_smoothing": 2.0, "part1_folds": 4,
        "part1_selection": "pooled OOF sample F1; ties prefer the incumbent",
        "part2_folds": 5, "part2_group": "trial_id", "part2_C": 1.0,
        "character_tfidf": CHAR_TFIDF, "new_blend_weights": [.5, .5],
        "part2_selection": "equal-phase log loss on all OOF rows; ties preserve incumbent",
        "validation_policy": "P1 control and OOF winner; P2 three fixed settings reported, gate only the OOF winner",
        "candidate_gate": "Strict improvement in both OOF and official validation; no automatic promotion",
        "reason_features_used_in_part2": False, "public_feedback_used_for_new_selection": False,
        "existing_work_has_reused_validation_and_public_feedback": True,
        "python": platform.python_version(), "sklearn": sklearn.__version__,
    }
    plan = json.loads(json.dumps(plan))
    plan_path = folder / "experiment_plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("The declared plan changed; use a separate study directory.")
    write_json(plan_path, plan)
    sources = ["midterm_next_round.py", "midterm_refinement.py", "midterm_augmentation.py",
               "midterm_evaluation.py", "midterm_finalization.py", "midterm_part2_baseline.py",
               "midterm_part1_finalization.py", "midterm_part2_improvement.py",
               'data/audit/protocol/part2_train_folds.csv']
    files = [root / 'tools/src' / s for s in sources] + [dataset_path(root, t, split) for t in ["part1", "part2"] for split in ["train", "val", "test"]]
    inputs = {p.relative_to(root).as_posix(): file_sha256(p) for p in files}
    if (folder / "input_and_source_hashes.json").exists():
        if inputs != read_json(folder / "input_and_source_hashes.json"):
            raise ValueError("Inputs or source changed during an incomplete study.")
        verify_hashes(root, read_json(folder / "protected_artifacts.json"))
    else:
        write_json(folder / "input_and_source_hashes.json", inputs)
        write_json(folder / "protected_artifacts.json", protected_hashes(root))
    train1 = pd.read_csv(dataset_path(root, "part1", "train"))
    train2 = pd.read_csv(dataset_path(root, "part2", "train"))
    lock1, oof1 = run_p1_oof(root, folder / "part1", train1)
    lock2, oof2 = run_p2_oof(root, folder / "part2", train2)
    # Both selections are persisted before official validation labels are read.
    gate1, bundle1 = full_p1(root, folder / "part1", train1, lock1, oof1)
    gate2, bundle2 = full_p2(root, folder / "part2", train2, lock2, oof2)
    gates = {"part1": gate1, "part2": gate2}
    paths = candidate_submissions(root, folder, gates, {"part1": bundle1, "part2": bundle2})
    summary = {**gates, "candidate_submissions": paths,
               "protected_final_artifacts_unchanged": True,
               "no_validation_labels_used_in_fitting": True,
               "all_OOF_rows_predicted_once": True, "reason_features_used_in_part2": False,
               "public_or_private_scores_available_for_new_candidates": False,
               "uploads_performed": False, "report_and_delivery_zip_updated": False}
    make_report(folder, summary)
    verify_hashes(root, read_json(folder / "protected_artifacts.json"))
    verify_hashes(root, inputs)
    write_json(folder / "summary.json", summary)
    finish_cache(folder)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--allow-training", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_study(args.project_root, allow_training=args.allow_training), indent=2))


if __name__ == "__main__":
    raise SystemExit("Historical study runner archived. Use python tools/src/train_core.py --help for the six fixed recipes.")
