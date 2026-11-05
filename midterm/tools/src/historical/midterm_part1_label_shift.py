"""Year-held-out binary label-shift test for Part 1 reasons.

The EM and BBSE estimators see only target embeddings. Target reason labels
are read for scoring only after the prior estimates have been saved.
"""
from __future__ import annotations

from _historical_support import prepare_output

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from midterm_augmentation import observed_label_mask
from midterm_evaluation import REASONS, dataset_path, file_sha256
from midterm_part1_round2 import fit_control, control_scores, make_decisions, score
from midterm_part1_finalization import build_submission
from midterm_part2_improvement import validate_submission

ROOT = None  # Set only by explicit replay(root, output).
OUT = None
EPS = 1e-6
ALPHAS = (0., .25, .5, 1.)
DECISION_THRESHOLD = .35
RECRUITMENT = REASONS.index("Recruitment")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def em_prior(scores, source_prior, *, tolerance=1e-10, max_iter=2000):
    """Binary Saerens-style EM fixed point with clipped numerical boundaries."""
    p = np.clip(np.asarray(scores, dtype=float), EPS, 1-EPS)
    source = float(np.clip(source_prior, EPS, 1-EPS))
    target = source
    for iteration in range(1, max_iter+1):
        corrected = expit(logit(p) + logit(target) - logit(source))
        updated = float(np.clip(corrected.mean(), EPS, 1-EPS))
        if abs(updated-target) < tolerance:
            return {"prior": updated, "iterations": iteration, "converged": True}
        target = updated
    return {"prior": target, "iterations": max_iter, "converged": False}


def bbse_prior(oof_scores, oof_labels, target_scores, *, threshold=DECISION_THRESHOLD):
    """Binary black-box shift estimate using held-out source predictions."""
    y = np.asarray(oof_labels, dtype=int)
    pred = np.asarray(oof_scores) >= threshold
    target_pred_rate = float((np.asarray(target_scores) >= threshold).mean())
    positives, negatives = y == 1, y == 0
    if not positives.any() or not negatives.any():
        raise ValueError("BBSE requires positive and negative source gold labels.")
    tpr = float(pred[positives].mean())
    fpr = float(pred[negatives].mean())
    determinant = tpr-fpr
    raw = (target_pred_rate-fpr)/determinant if abs(determinant) > 1e-12 else None
    return {"estimate_raw": None if raw is None else float(raw),
            "estimate_clipped": None if raw is None else float(np.clip(raw, 0, 1)),
            "tpr": tpr, "fpr": fpr, "tpr_minus_fpr": determinant,
            "target_predicted_positive_rate": target_pred_rate,
            "source_positive_count": int(positives.sum()),
            "source_negative_count": int(negatives.sum())}


def source_oof(source):
    observed = observed_label_mask(source[list(REASONS)].to_numpy())
    gold = source.loc[observed].reset_index(drop=True)
    y = gold.Recruitment.to_numpy(dtype=int)
    folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    result = np.full((len(gold), len(REASONS)), np.nan)
    assignment = np.full(len(gold), -1)
    for fold, (_, held_idx) in enumerate(folds.split(gold, y)):
        held = gold.iloc[held_idx]
        fit = source.loc[~source.ID.isin(held.ID)].reset_index(drop=True)
        if set(fit.ID) & set(held.ID):
            raise RuntimeError("Held-out source gold entered the OOF model.")
        result[held_idx] = control_scores(fit_control(fit), held)
        assignment[held_idx] = fold
    if not np.isfinite(result).all() or (assignment < 0).any():
        raise RuntimeError("Incomplete source OOF predictions.")
    return gold, result, assignment


def apply_shift(base, source_prior, target_prior, alpha):
    scores = np.asarray(base, dtype=float).copy()
    source = np.clip(source_prior, EPS, 1-EPS)
    target = np.clip(target_prior, EPS, 1-EPS)
    scores[:, RECRUITMENT] = expit(logit(np.clip(scores[:, RECRUITMENT], EPS, 1-EPS))
                                     + alpha*(logit(target)-logit(source)))
    return scores


def row_f1(y, d):
    intersection = (y*d).sum(axis=1)
    denominator = y.sum(axis=1)+d.sum(axis=1)
    return np.where(denominator > 0, 2*intersection/denominator, 0.)


def run(root, output):
    global ROOT, OUT
    ROOT, OUT = prepare_output(root, output)
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(dataset_path(ROOT, "part1", "train"))
    val = pd.read_csv(dataset_path(ROOT, "part1", "val"))
    combined = pd.concat([train, val], ignore_index=True)
    if combined.ID.duplicated().any():
        raise ValueError("Train and validation IDs overlap.")
    source = combined.loc[combined.start_year <= 2006].reset_index(drop=True)
    target_all = combined.loc[combined.start_year.between(2007, 2009)].reset_index(drop=True)
    target_features = target_all.drop(columns=list(REASONS))
    if len(source) != 667 or len(target_all) != 556:
        raise ValueError("Unexpected year split sizes.")
    write_json(OUT / "plan.json", {"source_year_max": 2006,
        "target_years": [2007, 2008, 2009], "source_rows": len(source),
        "target_unlabeled_rows": len(target_features), "alpha_grid": ALPHAS,
        "reason_applied": "Recruitment", "other_reasons_unchanged": True,
        "source_prior": "observed source-gold prevalence, with all-source mean model probability diagnostic",
        "target_prior": "binary EM on all 556 unlabeled target embeddings",
        "cross_check": "BBSE from four-fold source-gold OOF hard decisions at 0.35",
        "pass_gate": "EM converged, BBSE feasible with TPR-FPR >= 0.1, prior disagreement <= 0.10, sample F1 gain >= 0.01 over source-model alpha0 and train-only baseline, >=3 positive rows",
        "input_sha256": {split: file_sha256(dataset_path(ROOT, "part1", split)) for split in ("train", "val", "test")}})
    source_model = fit_control(source)
    target_scores = control_scores(source_model, target_features)
    source_gold, oof_scores, oof_fold = source_oof(source)
    source_y = source_gold[list(REASONS)].to_numpy(dtype=int)
    source_prior = source_y.mean(axis=0)
    source_model_mean = control_scores(source_model, source.drop(columns=list(REASONS))).mean(axis=0)
    priors = []
    for j, reason in enumerate(REASONS):
        em = em_prior(target_scores[:, j], source_prior[j])
        bbse = bbse_prior(oof_scores[:, j], source_y[:, j], target_scores[:, j])
        priors.append({"reason": reason, "source_gold_prior": float(source_prior[j]),
                       "source_model_mean_probability": float(source_model_mean[j]),
                       "target_mean_uncorrected_probability": float(target_scores[:, j].mean()),
                       "em_prior": em["prior"], "em_iterations": em["iterations"],
                       "em_converged": em["converged"], **{f"bbse_{k}": v for k, v in bbse.items()}})
    # Persist predictions and priors before reading hidden target reason labels.
    pd.DataFrame(priors).to_csv(OUT / "prior_estimates_before_reveal.csv", index=False)
    pd.DataFrame({"ID": source_gold.ID, "fold": oof_fold,
                  **{f"true_{r}": source_y[:, j] for j, r in enumerate(REASONS)},
                  **{f"prob_{r}": oof_scores[:, j] for j, r in enumerate(REASONS)}}).to_csv(OUT / "source_gold_oof.csv", index=False)
    pd.DataFrame({"ID": target_features.ID,
                  **{f"prob_{r}": target_scores[:, j] for j, r in enumerate(REASONS)}}).to_csv(OUT / "unlabeled_target_probabilities.csv", index=False)
    joblib.dump(source_model, OUT / "early_source_model.joblib", compress=3)
    # Reveal only the 56 labeled target rows for evaluation after prior estimation.
    hidden = target_all.loc[observed_label_mask(target_all[list(REASONS)].to_numpy())].reset_index(drop=True)
    if len(hidden) != 56:
        raise ValueError("Expected 56 hidden target gold labels.")
    held_positions = target_features.set_index("ID").index.get_indexer(hidden.ID)
    if (held_positions < 0).any():
        raise RuntimeError("Hidden gold IDs missing from target features.")
    y = hidden[list(REASONS)].to_numpy(dtype=int)
    recruitment_prior = next(row for row in priors if row["reason"] == "Recruitment")
    baseline = make_decisions("binary", target_scores[held_positions])
    baseline_f1 = score(y, baseline)
    train_only_early = train.loc[train.start_year <= 2006].reset_index(drop=True)
    train_only_scores = control_scores(fit_control(train_only_early), target_features)
    train_only_f1 = score(y, make_decisions("binary", train_only_scores[held_positions]))
    comparison = []
    decisions = {}
    for alpha in ALPHAS:
        updated = apply_shift(target_scores, recruitment_prior["source_gold_prior"],
                              recruitment_prior["em_prior"], alpha)
        decision = make_decisions("binary", updated[held_positions])
        decisions[alpha] = decision
        delta = row_f1(y, decision)-row_f1(y, baseline)
        comparison.append({"alpha": alpha, "sample_f1": score(y, decision),
                           "gain_vs_alpha0": float(score(y, decision)-baseline_f1),
                           "recruitment_f1": float(f1_score(y[:, RECRUITMENT], decision[:, RECRUITMENT], zero_division=0)),
                           "changed_decision_rows": int((decision != baseline).any(axis=1).sum()),
                           "positive_row_deltas": int((delta > 1e-12).sum()),
                           "negative_row_deltas": int((delta < -1e-12).sum()),
                           "predicted_multi": int((decision.sum(axis=1) > 1).sum())})
    pd.DataFrame(comparison).to_csv(OUT / "alpha_backtest.csv", index=False)
    detail = hidden[["ID", "start_year", *REASONS]].copy()
    for alpha, decision in decisions.items():
        key = str(alpha).replace(".", "p")
        for j, reason in enumerate(REASONS):
            detail[f"alpha_{key}_pred_{reason}"] = decision[:, j]
        detail[f"alpha_{key}_row_f1"] = row_f1(y, decision)
    detail.to_csv(OUT / "heldout_row_diagnostics.csv", index=False)
    best = sorted(comparison, key=lambda row: (-row["sample_f1"], row["alpha"]))[0]
    bbse_est = recruitment_prior["bbse_estimate_clipped"]
    disagreement = None if bbse_est is None else abs(recruitment_prior["em_prior"]-bbse_est)
    gates = {"em_converged": bool(recruitment_prior["em_converged"]),
             "bbse_identifiable": recruitment_prior["bbse_tpr_minus_fpr"] >= .1,
             "bbse_raw_feasible": recruitment_prior["bbse_estimate_raw"] is not None
                 and -.05 <= recruitment_prior["bbse_estimate_raw"] <= 1.05,
             "em_bbse_agree_within_0p10": disagreement is not None and disagreement <= .1,
             "f1_gain_at_least_0p01": best["gain_vs_alpha0"] >= .01,
             "beats_train_only_baseline_by_0p01": best["sample_f1"] >= train_only_f1 + .01,
             "at_least_three_positive_rows": best["positive_row_deltas"] >= 3,
             "nonzero_alpha_wins": best["alpha"] > 0}
    passed = all(gates.values())
    write_json(OUT / "backtest_decision.json", {"baseline_f1": baseline_f1,
        "train_only_early_baseline_f1": train_only_f1,
        "best": best, "recruitment_source_prior": recruitment_prior["source_gold_prior"],
        "recruitment_em_target_prior": recruitment_prior["em_prior"],
        "recruitment_bbse_target_prior": bbse_est, "em_bbse_difference": disagreement,
        "gates": gates, "passed": passed,
        "test_predictions_touched": False,
        "note": "56 hidden labels were read only after priors and target probabilities were saved."})
    if passed:
        # Apply the locked alpha to the 312-row test only after the backtest gate.
        final_model = fit_control(combined)
        test = pd.read_csv(dataset_path(ROOT, "part1", "test"))
        final_prior = float(combined.loc[observed_label_mask(combined[list(REASONS)].to_numpy()), "Recruitment"].mean())
        test_scores = control_scores(final_model, test.drop(columns=list(REASONS)))
        final_em = em_prior(test_scores[:, RECRUITMENT], final_prior)
        full_gold, full_oof, _ = source_oof(combined)
        full_bbse = bbse_prior(full_oof[:, RECRUITMENT], full_gold.Recruitment.to_numpy(dtype=int),
                               test_scores[:, RECRUITMENT])
        test_agreement = full_bbse["estimate_clipped"] is not None and abs(final_em["prior"]-full_bbse["estimate_clipped"]) <= .1
        if not (final_em["converged"] and full_bbse["tpr_minus_fpr"] >= .1
                and full_bbse["estimate_raw"] is not None
                and -.05 <= full_bbse["estimate_raw"] <= 1.05 and test_agreement):
            write_json(OUT / "test_candidate_rejected.json", {"reason": "final EM/BBSE cross-check failed",
                "em_test_prior": final_em["prior"], "bbse_test": full_bbse, "agreement": test_agreement})
            print(json.dumps(json.loads((OUT / "backtest_decision.json").read_text()), indent=2), flush=True)
            return
        shifted = apply_shift(test_scores, final_prior, final_em["prior"], best["alpha"])
        d = make_decisions("binary", shifted)
        pred = pd.DataFrame(d, columns=REASONS)
        pred.insert(0, "ID", test.ID.to_numpy())
        sample = pd.read_csv(dataset_path(ROOT, "part1", "test").parent / "part1_sample_submission.csv")
        submission = build_submission(sample, pred)
        submission.to_csv(OUT / "part1_label_shift_candidate.csv", index=False)
        validate_submission(submission, sample, 1)
        joblib.dump(final_model, OUT / "final_model.joblib", compress=3)
        write_json(OUT / "test_candidate_metadata.json", {"alpha": best["alpha"],
            "source_prior": final_prior, "em_test_prior": final_em["prior"],
            "bbse_test": full_bbse,
            "em_converged": final_em["converged"], "test_predicted_multi": int((d.sum(axis=1)>1).sum()),
            "submission_sha256": file_sha256(OUT / "part1_label_shift_candidate.csv"),
            "kaggle_uploaded": False})
    print(json.dumps(json.loads((OUT / "backtest_decision.json").read_text()), indent=2), flush=True)




if __name__ == "__main__":
    from replay import main
    main(default_study='label_shift')
