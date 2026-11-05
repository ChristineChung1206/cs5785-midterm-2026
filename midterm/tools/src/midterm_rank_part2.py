"""Phase-shared and recency-weighted Part 2 candidate experiment.

The temporal screen selects one text model with only outcome_train labels.
Grouped OOF chooses a frozen blend weight against the current final model;
the supplied validation set is checked once after those choices are locked.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from midterm_augmentation import logistic_parameters
from midterm_evaluation import PHASES, dataset_path, evaluate_part2, file_sha256
from midterm_finalization import predict_final_outcomes
from midterm_next_round import CHAR_TFIDF, validate_grouped_assignment, verify_oof
from midterm_part2_baseline import build_trial_text, fit_phase_baseline, predict_outcome_probabilities, validate_outcome_rows
from midterm_part2_improvement import validate_submission
from midterm_refinement import TFIDF, fit_field_outcomes, predict_field_outcomes

STUDY = Path('models/part2/03_phase_shared/evidence/selection')
SETTINGS = (
    {"alpha": 0.0, "half_life": None},
    {"alpha": 0.35, "half_life": None},
    {"alpha": 1.0, "half_life": None},
    {"alpha": 0.35, "half_life": 12},
    {"alpha": 0.35, "half_life": 6},
)
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 1.0)
WORD_PARAMS = {**TFIDF, "max_features": 30000, "dtype": np.float32}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def recency_weights(frame: pd.DataFrame, half_life: int | None, *, reference_year: int) -> np.ndarray:
    if half_life is None:
        return np.ones(len(frame), dtype=float)
    years = pd.to_numeric(frame.start_year, errors="coerce").to_numpy(dtype=float)
    observed = years[np.isfinite(years)]
    if not len(observed) or np.any(observed > reference_year):
        raise ValueError("Recency weights require known training years not later than the reference year.")
    years = np.where(np.isfinite(years), years, np.median(observed))
    weights = np.exp(-math.log(2) * (reference_year - years) / half_life)
    return weights / weights.mean()  # Same total weight keeps LR regularization comparable.


def phase_design(text_matrix, phases: pd.Series, alpha: float):
    """Shared text, optional phase-specific text interactions, and phase intercepts."""
    phase = phases.to_numpy()
    if not np.isin(phase, PHASES).all():
        raise ValueError("Only phases I, II, and III are allowed.")
    indicators = sparse.csr_matrix(np.column_stack([phase == p for p in PHASES]).astype(np.float32))
    blocks = [text_matrix]
    if alpha:
        blocks.extend(text_matrix.multiply((phase == p).astype(np.float32)[:, None]) * alpha for p in PHASES)
    blocks.append(indicators)
    return sparse.hstack(blocks, format="csr", dtype=np.float32)


def fit_phase_shared(train: pd.DataFrame, setting: dict, *, reference_year: int):
    validate_outcome_rows(train)
    vectorizer = TfidfVectorizer(**WORD_PARAMS)
    x_text = vectorizer.fit_transform(build_trial_text(train))
    x = phase_design(x_text, train.phase, setting["alpha"])
    classifier = LogisticRegression(**{**logistic_parameters(42), "C": 1.0, "max_iter": 1000})
    weights = recency_weights(train, setting["half_life"], reference_year=reference_year)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classifier.fit(x, train.label.to_numpy(dtype=int), sample_weight=weights)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught) or classifier.n_iter_.max() >= classifier.max_iter:
        raise RuntimeError("Phase-shared LR failed to converge.")
    return {"vectorizer": vectorizer, "classifier": classifier, "setting": dict(setting),
            "reason_features_used": False, "positive_class": 1,
            "fit_row_ids": train.ID.tolist(), "fit_trial_ids": sorted(train.trial_id.unique()),
            "reference_year": reference_year, "vocabulary_size": len(vectorizer.vocabulary_)}


def predict_phase_shared(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    validate_outcome_rows(frame, labels_required=False)
    if bundle["reason_features_used"] is not False or bundle["positive_class"] != 1:
        raise ValueError("Unexpected model input scope or positive class.")
    text = bundle["vectorizer"].transform(build_trial_text(frame))
    x = phase_design(text, frame.phase, bundle["setting"]["alpha"])
    p = bundle["classifier"].predict_proba(x)[:, list(bundle["classifier"].classes_).index(1)]
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid success probabilities.")
    return pd.DataFrame({"ID": frame.ID.to_numpy(), "p_success": p})


def blend(reference: pd.DataFrame, current: pd.DataFrame, proposed: pd.DataFrame, weight: float) -> pd.DataFrame:
    if not 0 <= weight <= 1 or not current.ID.is_unique or not proposed.ID.is_unique:
        raise ValueError("Invalid weight or duplicate prediction IDs.")
    if set(reference.ID) != set(current.ID) or set(reference.ID) != set(proposed.ID):
        raise ValueError("Both components must cover exactly the reference IDs.")
    a = current.set_index("ID").loc[reference.ID, "p_success"].to_numpy(dtype=float)
    b = proposed.set_index("ID").loc[reference.ID, "p_success"].to_numpy(dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or ((a < 0) | (a > 1)).any() or ((b < 0) | (b > 1)).any():
        raise ValueError("Invalid component probabilities.")
    return pd.DataFrame({"ID": reference.ID.to_numpy(), "p_success": (1-weight)*a + weight*b})


def temporal_screen(train: pd.DataFrame, folder: Path):
    """Past <=2009 predicts 2010-2013, matching the direction of the hidden test."""
    year = pd.to_numeric(train.start_year, errors="coerce")
    fit = train.loc[year.le(2009)].reset_index(drop=True)
    held = train.loc[year.between(2010, 2013)].reset_index(drop=True)
    if len(fit) < 1000 or len(held) < 500 or set(fit.trial_id) & set(held.trial_id):
        raise ValueError("Temporal split is too small or leaks a trial.")
    # Retrain each component of the current 25/25/50 final blend on <=2009.
    # This is the direct temporal comparator for the new pooled model.
    baseline, _, _, _ = fit_phase_baseline(fit, tfidf_parameters={**TFIDF, "max_features": 20000},
                                          classifier_parameters={"C": 1.0})
    word_scores = predict_outcome_probabilities(baseline, held)
    base_result = evaluate_part2(held, word_scores, .5)
    base_result.save(folder / "temporal_phase_word_diagnostic")
    field, _ = fit_field_outcomes(fit)
    field_scores = predict_field_outcomes(field, held)
    character, _, _, _ = fit_phase_baseline(fit, tfidf_parameters=CHAR_TFIDF,
                                            classifier_parameters={"C": 1.0})
    character_scores = predict_outcome_probabilities(character, held)
    word_blend = blend(held, word_scores, field_scores, .5)
    final_scores = blend(held, word_blend, character_scores, .5)
    final_result = evaluate_part2(held, final_scores, .5)
    final_result.save(folder / "temporal_current_final_control")
    rows = [{"setting_id": "phase_word_control", "alpha": None, "half_life": None,
             **base_result.summary},
            {"setting_id": "current_final_control", "alpha": None, "half_life": None,
             **final_result.summary}]
    phase_rows = [base_result.by_group.assign(setting_id="phase_word_control"),
                  final_result.by_group.assign(setting_id="current_final_control")]
    for i, setting in enumerate(SETTINGS):
        model = fit_phase_shared(fit, setting, reference_year=2009)
        result = evaluate_part2(held, predict_phase_shared(model, held), .5)
        result.save(folder / f"temporal_setting_{i}")
        rows.append({"setting_id": i, **setting, **result.summary})
        phase_rows.append(result.by_group.assign(setting_id=i))
        print(f"Part 2 temporal setting {i+1}/{len(SETTINGS)} complete", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(folder / "temporal_comparison.csv", index=False)
    pd.concat(phase_rows).to_csv(folder / "temporal_by_phase.csv", index=False)
    winner = min(range(len(SETTINGS)), key=lambda i: (float(table.loc[table.setting_id.eq(i), "log_loss_equal_phase"].iloc[0]), i))
    return winner, float(final_result.summary["log_loss_equal_phase"]), float(table.loc[table.setting_id.eq(winner), "log_loss_equal_phase"].iloc[0]), {
        "fit_rows": len(fit), "held_rows": len(held), "fit_trials": fit.trial_id.nunique(),
        "held_trials": held.trial_id.nunique(), "cutoff": 2009,
        "missing_year_rows_excluded": int(year.isna().sum()), "trial_overlap": 0,
    }


def grouped_oof(train: pd.DataFrame, assignment: pd.DataFrame, setting: dict, folder: Path):
    pieces = []
    fold_rows = []
    for fold in sorted(assignment.fold.unique()):
        held_ids = set(assignment.loc[assignment.fold.eq(fold), "ID"])
        fit = train.loc[~train.ID.isin(held_ids)].reset_index(drop=True)
        held = train.loc[train.ID.isin(held_ids)].reset_index(drop=True)
        if set(fit.trial_id) & set(held.trial_id):
            raise ValueError("A trial crosses the grouped OOF split.")
        model = fit_phase_shared(fit, setting, reference_year=2013)
        result = evaluate_part2(held, predict_phase_shared(model, held), .5)
        pieces.append(result.predictions.assign(fold=int(fold)))
        fold_rows.append({"fold": int(fold), **result.summary})
        print(f"Part 2 grouped OOF fold {int(fold)+1}/5 complete", flush=True)
    oof = verify_oof(train, pd.concat(pieces, ignore_index=True), assignment, part=2)
    oof.to_csv(folder / "oof_phase_shared.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(folder / "oof_by_fold.csv", index=False)
    return oof


def run_study(root: Path):
    folder = root / STUDY
    folder.mkdir(parents=True, exist_ok=True)
    frozen = {name: file_sha256(root / name) for name in (
        'tools/archive/records/final_selection_nov01.json', 'models/part2/02_text_blend/model.joblib',
        'models/part1/02_kmeans_lr/submission.csv', 'models/part2/02_text_blend/submission.csv')}
    train = pd.read_csv(dataset_path(root, "part2", "train"))
    write_json(folder / "experiment_plan.json", {
        "settings": SETTINGS, "blend_weights": BLEND_WEIGHTS, "word_tfidf": {k: (str(v) if k == "dtype" else v) for k, v in WORD_PARAMS.items()},
        "temporal_cutoff": 2009, "selection_order": "temporal setting, then grouped OOF blend weight, then one official validation check",
        "reason_features_used": False, "test_labels_used": False,
    })
    winner, temporal_control, temporal_winner, split = temporal_screen(train, folder)
    write_json(folder / "temporal_selection_locked.json", {"winner": winner, "setting": SETTINGS[winner],
        "current_final_control_loss": temporal_control, "winner_loss": temporal_winner, "split": split,
        "validation_labels_used": False, "test_labels_used": False})
    assignment = pd.read_csv(root / 'data/audit/protocol/part2_train_folds.csv')
    assignment = validate_grouped_assignment(train, assignment)
    oof = grouped_oof(train, assignment, SETTINGS[winner], folder)
    current_saved = pd.read_csv(root / "refinement/oof_weight_char/part2/oof_current_plus_character.csv")
    current_oof = verify_oof(train, current_saved, assignment, part=2)
    current_scores = current_oof[["ID", "p_success"]]
    proposed_scores = oof[["ID", "p_success"]]
    rows = []
    for weight in BLEND_WEIGHTS:
        result = evaluate_part2(train, blend(train, current_scores, proposed_scores, weight), .5)
        rows.append({"weight": weight, **result.summary})
        result.by_group.assign(weight=weight).to_csv(folder / f"oof_by_phase_weight_{str(weight).replace('.', 'p')}.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(folder / "oof_blend_comparison.csv", index=False)
    chosen_weight = min(BLEND_WEIGHTS, key=lambda w: (float(table.loc[table.weight.eq(w), "log_loss_equal_phase"].iloc[0]), BLEND_WEIGHTS.index(w)))
    write_json(folder / "selection_locked_before_validation.json", {
        "setting_id": winner, "setting": SETTINGS[winner], "blend_weight": chosen_weight,
        "selection_metric": "equal-phase grouped OOF log loss", "validation_labels_used": False,
        "test_labels_used": False, "fit_preprocessing_inside_each_fold": True})
    # Check the exact OOF-selected blend on the historical future holdout.
    temporal_reference = pd.read_csv(folder / "temporal_current_final_control/validation_predictions.csv")
    temporal_final = temporal_reference[["ID", "p_success"]]
    temporal_shared = pd.read_csv(folder / f"temporal_setting_{winner}/validation_predictions.csv")[["ID", "p_success"]]
    temporal_blend = evaluate_part2(temporal_reference,
        blend(temporal_reference, temporal_final, temporal_shared, chosen_weight), .5)
    temporal_blend.save(folder / "temporal_selected_blend")
    temporal_blend_loss = float(temporal_blend.summary["log_loss_equal_phase"])
    full = fit_phase_shared(train, SETTINGS[winner], reference_year=2013)
    joblib.dump(full, folder / "selected_phase_shared_model.joblib", compress=3)
    validation = pd.read_csv(dataset_path(root, "part2", "val"))
    final = joblib.load(root / 'models/part2/02_text_blend/model.joblib')
    current_val = predict_final_outcomes(final, validation)
    saved_val = pd.read_csv(root / 'models/part2/02_text_blend/evidence/selection/selected_validation/validation_predictions.csv')
    error = np.max(np.abs(current_val.set_index("ID").loc[validation.ID, "p_success"].to_numpy() -
                          saved_val.set_index("ID").loc[validation.ID, "p_success"].to_numpy()))
    if error > 1e-12:
        raise ValueError("Current final model does not reproduce saved validation probabilities.")
    shared_val = predict_phase_shared(full, validation)
    control = evaluate_part2(validation, current_val, .5)
    candidate = evaluate_part2(validation, blend(validation, current_val, shared_val, chosen_weight), .5)
    control.save(folder / "validation_control")
    candidate.save(folder / "validation_candidate")
    candidate.by_group.assign(setting="candidate").to_csv(folder / "validation_candidate_by_phase.csv", index=False)
    oof_control = float(table.loc[table.weight.eq(0), "log_loss_equal_phase"].iloc[0])
    oof_candidate = float(table.loc[table.weight.eq(chosen_weight), "log_loss_equal_phase"].iloc[0])
    val_control = float(control.summary["log_loss_equal_phase"])
    val_candidate = float(candidate.summary["log_loss_equal_phase"])
    passes = bool(chosen_weight > 0 and temporal_blend_loss < temporal_control - 1e-12 and
                  oof_candidate < oof_control - 1e-12 and val_candidate < val_control - 1e-12)
    summary = {"setting_id": winner, "setting": SETTINGS[winner], "blend_weight": chosen_weight,
               "temporal_current_final_control_loss": temporal_control, "temporal_phase_shared_loss": temporal_winner,
               "temporal_selected_blend_loss": temporal_blend_loss,
               "oof_current_final_loss": oof_control, "oof_candidate_loss": oof_candidate,
               "validation_current_final_loss": val_control, "validation_candidate_loss": val_candidate,
               "validation_reproduction_max_abs_error": float(error),
               "passes_local_gate": passes, "final_replaced": False, "kaggle_uploaded": False}
    if passes:
        test = pd.read_csv(dataset_path(root, "part2", "test"))
        current_test = predict_final_outcomes(final, test)
        shared_test = predict_phase_shared(full, test)
        probabilities = blend(test, current_test, shared_test, chosen_weight)
        sample = pd.read_csv(dataset_path(root, "part2", "test").parent / "part2_sample_submission.csv")
        submission = probabilities.rename(columns={"p_success": "label"})
        submission.to_csv(folder / "part2_phase_shared_candidate.csv", index=False)
        validate_submission(submission, sample, 2)
        summary["candidate_submission"] = str((folder / "part2_phase_shared_candidate.csv").relative_to(root))
    write_json(folder / "summary.json", summary)
    for name, digest in frozen.items():
        if file_sha256(root / name) != digest:
            raise RuntimeError(f"A locked final artifact changed: {name}")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    run_study(args.project_root.resolve())


if __name__ == "__main__":
    raise SystemExit("Historical study runner archived. Use python tools/src/train_core.py --help for the six fixed recipes.")
