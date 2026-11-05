"""Targeted Part 1 Other correction and independent text-view audit.

The baseline four named reason decisions are preserved. New candidates only
add Other. Outer OOF folds, inner residual-model folds, and time holdouts all
exclude held-out labels before fitting every transform and classifier.
"""
from __future__ import annotations

from _historical_support import prepare_output

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from midterm_augmentation import logistic_parameters, observed_label_mask
from midterm_evaluation import REASONS, dataset_path, file_sha256
from midterm_next_round import p1_assignment
from midterm_part1_round2 import fit_control, control_scores, make_decisions, score
from midterm_part1_finalization import build_submission
from midterm_part2_improvement import validate_submission
from midterm_refinement import EMBEDDINGS

ROOT = None  # Set only by explicit replay(root, output).
OUT = None
OTHER = REASONS.index("Other")
TEXT_FIELDS = ("criteria", "diseases_cleaned", "drugs")
TEXT_LIMITS = (12000, 4000, 4000)
RESIDUAL_THRESHOLDS = (.65, .75, .85)
TEXT_THRESHOLDS = (.25, .35, .45)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def write_locked_json(path, obj):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != obj:
        raise ValueError(f"Locked selection changed: {path}")
    write_json(path, obj)


def row_f1(y, decision):
    numerator = 2 * (y * decision).sum(axis=1)
    denominator = y.sum(axis=1) + decision.sum(axis=1)
    return np.divide(numerator, denominator, out=np.zeros(len(y), dtype=float), where=denominator > 0)


def text_values(frame, field):
    return frame[field].fillna("").astype(str).tolist()


def fit_text(frame):
    observed = observed_label_mask(frame[list(REASONS)].to_numpy())
    vectorizers, blocks = {}, []
    for field, limit in zip(TEXT_FIELDS, TEXT_LIMITS):
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=limit,
                                    lowercase=True, strip_accents="unicode", sublinear_tf=True,
                                    norm="l2", stop_words="english")
        blocks.append(vectorizer.fit_transform(text_values(frame, field)))
        vectorizers[field] = vectorizer
    features = hstack(blocks, format="csr") / np.sqrt(len(TEXT_FIELDS))
    y = frame.loc[observed, list(REASONS)].to_numpy(dtype=int)
    models = []
    for j in range(len(REASONS)):
        model = LogisticRegression(**{**logistic_parameters(42), "C": 1.0})
        with threadpool_limits(limits=1):
            model.fit(features[observed], y[:, j])
        models.append(model)
    return {"vectorizers": vectorizers, "models": models,
            "fit_ids": frame.ID.tolist(), "gold_ids": frame.loc[observed, "ID"].tolist()}


def text_scores(bundle, frame):
    blocks = [bundle["vectorizers"][field].transform(text_values(frame, field)) for field in TEXT_FIELDS]
    features = hstack(blocks, format="csr") / np.sqrt(len(TEXT_FIELDS))
    return np.column_stack([model.predict_proba(features)[:, 1] for model in bundle["models"]])


def residual_features(bundle, fit_frame, query, base_scores):
    observed = observed_label_mask(fit_frame[list(REASONS)].to_numpy())
    gold = fit_frame.loc[observed]
    scaler = bundle["scaler"]
    x_gold = scaler.transform(gold[EMBEDDINGS])
    x_query = scaler.transform(query[EMBEDDINGS])
    distances = cdist(x_query, x_gold, metric="euclidean")
    other_mask = gold.Other.to_numpy(dtype=int) == 1
    if not other_mask.any() or other_mask.all():
        raise ValueError("Residual distance features need Other and non-Other gold.")
    named = np.asarray(base_scores)[:, :OTHER]
    top_two = np.sort(named, axis=1)[:, -2:]
    return np.column_stack([base_scores[:, OTHER], top_two[:, 1],
                            top_two[:, 1]-top_two[:, 0],
                            distances[:, other_mask].min(axis=1),
                            distances[:, ~other_mask].min(axis=1)])


def fit_residual_meta(frame):
    observed = observed_label_mask(frame[list(REASONS)].to_numpy())
    gold = frame.loc[observed].reset_index(drop=True)
    labels = gold.Other.to_numpy(dtype=int)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    meta_x = np.full((len(gold), 5), np.nan)
    for _, held_idx in inner.split(gold, labels):
        held = gold.iloc[held_idx]
        inner_fit = frame.loc[~frame.ID.isin(held.ID)].reset_index(drop=True)
        assert set(inner_fit.ID).isdisjoint(held.ID)
        bundle = fit_control(inner_fit)
        p = control_scores(bundle, held)
        meta_x[held_idx] = residual_features(bundle, inner_fit, held, p)
    if not np.isfinite(meta_x).all():
        raise RuntimeError("Missing inner OOF residual features.")
    scaler = StandardScaler().fit(meta_x)
    classifier = LogisticRegression(C=.25, solver="liblinear", max_iter=500,
                                    class_weight="balanced", random_state=42)
    with threadpool_limits(limits=1):
        classifier.fit(scaler.transform(meta_x), labels)
    return {"scaler": scaler, "classifier": classifier,
            "training_gold_ids": gold.ID.tolist(), "inner_oof_features": meta_x}


def residual_scores(meta, base_bundle, fit_frame, query, base_prob):
    features = residual_features(base_bundle, fit_frame, query, base_prob)
    return meta["classifier"].predict_proba(meta["scaler"].transform(features))[:, 1]


def add_other(base_prob, other_signal, threshold, *, method):
    decision = make_decisions("binary", base_prob)
    if method == "residual":
        eligible = (other_signal >= threshold) & (base_prob[:, OTHER] >= .08)
    elif method == "text":
        eligible = (other_signal >= threshold) & (base_prob[:, OTHER] >= .10)
    else:
        raise ValueError(method)
    decision[eligible, OTHER] = 1
    return decision


def fit_and_score(fit, query):
    base = fit_control(fit)
    base_prob = control_scores(base, query)
    text = fit_text(fit)
    text_prob = text_scores(text, query)
    meta = fit_residual_meta(fit)
    residual_prob = residual_scores(meta, base, fit, query, base_prob)
    return {"base": base_prob, "text": text_prob, "residual": residual_prob,
            "models": {"base": base, "text": text, "meta": meta}}


def split_scores(fit, query, *, include_models=False):
    result = fit_and_score(fit, query)
    if not include_models:
        result.pop("models")
    return result


def candidate_decisions(scores, residual_threshold, text_threshold):
    return {
        "baseline": make_decisions("binary", scores["base"]),
        "residual_other": add_other(scores["base"], scores["residual"], residual_threshold, method="residual"),
        "text_other": add_other(scores["base"], scores["text"][:, OTHER], text_threshold, method="text"),
    }


def difference_rows(split, reference, baseline, candidate, method):
    y = reference[list(REASONS)].to_numpy(dtype=int)
    before = row_f1(y, baseline)
    after = row_f1(y, candidate)
    changed = np.flatnonzero((baseline != candidate).any(axis=1))
    rows = []
    for i in changed:
        reason_set = lambda d: " ".join(r for j, r in enumerate(REASONS) if d[i, j])
        rows.append({"split": split, "method": method, "ID": reference.iloc[i].ID,
                     "start_year": int(reference.iloc[i].start_year),
                     "truth": reason_set(y), "before": reason_set(baseline),
                     "after": reason_set(candidate), "row_f1_before": float(before[i]),
                     "row_f1_after": float(after[i]), "delta": float(after[i]-before[i]),
                     "outcome": "fixed" if after[i] > before[i] else "new_error" if after[i] < before[i] else "unchanged_f1"})
    return rows


def evaluate_rows(split, reference, decisions):
    y = reference[list(REASONS)].to_numpy(dtype=int)
    baseline = decisions["baseline"]
    rows = []
    for name, d in decisions.items():
        rows.append({"split": split, "candidate": name, "n": len(reference),
                     "sample_f1": score(y, d),
                     "exact_sets": int((y == d).all(axis=1).sum()),
                     "other_f1": float(f1_score(y[:, OTHER], d[:, OTHER], zero_division=0)),
                     "other_false_negatives": int(((y[:, OTHER] == 1) & (d[:, OTHER] == 0)).sum()),
                     "other_false_positives": int(((y[:, OTHER] == 0) & (d[:, OTHER] == 1)).sum()),
                     "fixed_rows": int((row_f1(y,d) > row_f1(y,baseline)+1e-12).sum()),
                     "new_error_rows": int((row_f1(y,d) < row_f1(y,baseline)-1e-12).sum())})
    return rows


def audit_labels(train, val, out):
    combined = pd.concat([train, val], ignore_index=True)
    gold = combined.loc[observed_label_mask(combined[list(REASONS)].to_numpy())].reset_index(drop=True)
    x = gold[EMBEDDINGS].to_numpy(dtype=float)
    distances = cdist(x, x, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    finite = distances[np.triu_indices(len(gold), k=1)]
    quantile = np.quantile(finite, .02)
    y = gold[list(REASONS)].to_numpy(dtype=int)
    upper_i, upper_j = np.triu_indices(len(gold), k=1)
    conflict = (y[upper_i] != y[upper_j]).any(axis=1)
    very_close = finite <= quantile
    exact_duplicate = finite < 1e-8
    pairs = []
    for i in range(len(gold)):
        for j in np.argsort(distances[i])[:5]:
            if j <= i:
                continue
            intersection = np.logical_and(y[i], y[j]).sum()
            union = np.logical_or(y[i], y[j]).sum()
            overlap = intersection/union if union else 1.
            if overlap >= 1.:
                continue
            pairs.append({"ID_a": gold.iloc[i].ID, "ID_b": gold.iloc[j].ID,
                          "year_a": int(gold.iloc[i].start_year), "year_b": int(gold.iloc[j].start_year),
                          "raw_embedding_distance": float(distances[i, j]),
                          "below_pairwise_2pct_distance": bool(distances[i, j] <= quantile),
                          "label_jaccard": float(overlap),
                          "labels_a": " ".join(np.array(REASONS)[y[i] == 1]),
                          "labels_b": " ".join(np.array(REASONS)[y[j] == 1])})
    pd.DataFrame(pairs).sort_values(["raw_embedding_distance", "label_jaccard"]).head(30).to_csv(out / "nearest_discordant_gold_pairs.csv", index=False)
    oof = pd.read_csv(ROOT / 'models/part1/exploration/replay_inputs/oof_weight_char/oof_weight_010.csv')
    val_pred = pd.read_csv(ROOT / 'models/part1/02_kmeans_lr/evidence/selected/selected_validation/validation_predictions.csv')
    review = []
    for source, frame in (("train_gold_oof", oof), ("validation_holdout", val_pred)):
        for reason in REASONS:
            y_r = frame[f"true_{reason}"].to_numpy(dtype=int)
            p_r = frame[f"prob_{reason}"].to_numpy(dtype=float)
            high_wrong = (y_r == 0) & (p_r >= .75)
            low_wrong = (y_r == 1) & (p_r <= .15)
            for i in np.flatnonzero(high_wrong | low_wrong):
                review.append({"source": source, "ID": frame.iloc[i].ID,
                               "reason": reason, "observed_label": int(y_r[i]),
                               "oof_or_holdout_probability": float(p_r[i]),
                               "flag": "high_probability_negative" if high_wrong[i] else "low_probability_positive",
                               "action": "manual_review_only_no_label_change"})
    pd.DataFrame(review, columns=["source", "ID", "reason", "observed_label",
                                  "oof_or_holdout_probability", "flag", "action"]).to_csv(out / "high_confidence_label_review.csv", index=False)
    return {"gold_rows": len(gold), "discordant_pairs_saved": min(30, len(pairs)),
            "pairwise_distance_2pct": float(quantile), "high_confidence_flags": len(review),
            "all_pairs_within_2pct": int(very_close.sum()),
            "discordant_pairs_within_2pct": int((very_close & conflict).sum()),
            "exact_duplicate_pairs": int(exact_duplicate.sum()),
            "discordant_exact_duplicate_pairs": int((exact_duplicate & conflict).sum()),
            "labels_modified": 0}


def save_scores(path, reference, scores, fold=None):
    result = reference[["ID", "start_year", *REASONS]].reset_index(drop=True).copy()
    if fold is not None:
        result["fold"] = fold
    for j, reason in enumerate(REASONS):
        result[f"base_prob_{reason}"] = scores["base"][:, j]
        result[f"text_prob_{reason}"] = scores["text"][:, j]
    result["residual_other_score"] = scores["residual"]
    result.to_csv(path, index=False)


def run(root, output):
    global ROOT, OUT
    ROOT, OUT = prepare_output(root, output)
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(dataset_path(ROOT, "part1", "train"))
    val = pd.read_csv(dataset_path(ROOT, "part1", "val"))
    gold, assignment = p1_assignment(train)
    fold = assignment.set_index("ID").loc[gold.ID, "fold"].to_numpy()
    write_json(OUT / "plan.json", {"candidate_scope": "add Other only; four named reasons preserved",
        "residual_features": ["base_other_probability", "max_named_probability", "top_two_named_gap",
                              "nearest_other_gold_distance", "nearest_nonother_gold_distance"],
        "residual_thresholds": RESIDUAL_THRESHOLDS, "text_fields": TEXT_FIELDS,
        "text_thresholds": TEXT_THRESHOLDS, "text_min_df": 3, "text_ngram": [1,2],
        "selection": "OOF sample F1 threshold winner per method; then require both OOF and 56-row temporal gain and no validation loss",
        "time_fit": "train+validation rows <=2006, time holdout 2007-2009 labeled rows (56)",
        "label_changes_allowed": False,
        "input_sha256": {split: file_sha256(dataset_path(ROOT, "part1", split)) for split in ("train", "val", "test")}})
    oof_base = np.full((len(gold), len(REASONS)), np.nan)
    oof_text = np.full((len(gold), len(REASONS)), np.nan)
    oof_meta = np.full(len(gold), np.nan)
    for k in range(4):
        held = gold.loc[fold == k]
        fit = train.loc[~train.ID.isin(held.ID)].reset_index(drop=True)
        assert set(fit.ID).isdisjoint(held.ID)
        result = split_scores(fit, held)
        oof_base[fold == k] = result["base"]
        oof_text[fold == k] = result["text"]
        oof_meta[fold == k] = result["residual"]
        print(f"Other-view outer OOF fold {k+1}/4 done", flush=True)
    if not np.isfinite(oof_base).all() or not np.isfinite(oof_text).all() or not np.isfinite(oof_meta).all():
        raise RuntimeError("Incomplete OOF predictions.")
    saved_oof = pd.read_csv(ROOT / 'models/part1/exploration/replay_inputs/oof_weight_char/oof_weight_010.csv').set_index("ID").loc[gold.ID]
    saved_prob = saved_oof[[f"prob_{r}" for r in REASONS]].to_numpy()
    if not np.allclose(oof_base, saved_prob, atol=1e-12, rtol=0):
        raise RuntimeError("Baseline OOF no longer reproduces the locked model.")
    oof_scores = {"base": oof_base, "text": oof_text, "residual": oof_meta}
    base_decision = make_decisions("binary", oof_base)
    y_oof = gold[list(REASONS)].to_numpy(dtype=int)
    grid = []
    for method, thresholds in (("residual", RESIDUAL_THRESHOLDS), ("text", TEXT_THRESHOLDS)):
        signal = oof_meta if method == "residual" else oof_text[:, OTHER]
        for threshold in thresholds:
            d = add_other(oof_base, signal, threshold, method=method)
            grid.append({"method": method, "threshold": threshold, "sample_f1": score(y_oof, d),
                         "fixed_rows": int((row_f1(y_oof, d) > row_f1(y_oof, base_decision)+1e-12).sum()),
                         "new_error_rows": int((row_f1(y_oof, d) < row_f1(y_oof, base_decision)-1e-12).sum())})
    pd.DataFrame(grid).to_csv(OUT / "oof_threshold_grid.csv", index=False)
    selected_threshold = {method: sorted((row for row in grid if row["method"] == method),
                                         key=lambda row: (-row["sample_f1"], -row["threshold"]))[0]["threshold"]
                          for method in ("residual", "text")}
    write_locked_json(OUT / "selection_locked_before_time_validation.json", {"thresholds": selected_threshold,
        "oof_baseline_f1": score(y_oof, base_decision), "oof_grid": grid,
        "time_and_validation_labels_used_for_threshold_selection": False})
    oof_d = candidate_decisions(oof_scores, selected_threshold["residual"], selected_threshold["text"])
    save_scores(OUT / "oof_scores.csv", gold, oof_scores, fold=fold)
    combined = pd.concat([train, val], ignore_index=True)
    early = combined.loc[combined.start_year <= 2006].reset_index(drop=True)
    late = combined.loc[combined.start_year.between(2007, 2009)].reset_index(drop=True)
    late_gold = late.loc[observed_label_mask(late[list(REASONS)].to_numpy())].reset_index(drop=True)
    if len(late_gold) != 56 or set(early.ID) & set(late_gold.ID):
        raise RuntimeError("Unexpected time holdout or overlap.")
    time_scores = split_scores(early, late_gold)
    time_d = candidate_decisions(time_scores, selected_threshold["residual"], selected_threshold["text"])
    save_scores(OUT / "temporal_scores.csv", late_gold, time_scores)
    val_scores = split_scores(train, val)
    val_d = candidate_decisions(val_scores, selected_threshold["residual"], selected_threshold["text"])
    save_scores(OUT / "validation_scores.csv", val, val_scores)
    all_metrics = (evaluate_rows("oof", gold, oof_d)
                   + evaluate_rows("temporal_56", late_gold, time_d)
                   + evaluate_rows("validation_78", val, val_d))
    pd.DataFrame(all_metrics).to_csv(OUT / "comparison.csv", index=False)
    text_standalone = []
    for split, reference, scores in (("oof", gold, oof_scores),
                                     ("temporal_56", late_gold, time_scores),
                                     ("validation_78", val, val_scores)):
        d = make_decisions("binary", scores["text"])
        y = reference[list(REASONS)].to_numpy(dtype=int)
        text_standalone.append({"split": split, "n": len(reference),
                                "text_only_sample_f1": score(y, d),
                                "text_only_other_f1": float(f1_score(y[:, OTHER], d[:, OTHER], zero_division=0))})
    pd.DataFrame(text_standalone).to_csv(OUT / "text_standalone_metrics.csv", index=False)
    changes = []
    for split, ref, decisions in (("oof", gold, oof_d), ("temporal_56", late_gold, time_d),
                                  ("validation_78", val, val_d)):
        for method in ("residual_other", "text_other"):
            changes.extend(difference_rows(split, ref, decisions["baseline"], decisions[method], method))
    pd.DataFrame(changes, columns=["split", "method", "ID", "start_year", "truth", "before", "after",
                                   "row_f1_before", "row_f1_after", "delta", "outcome"]).to_csv(OUT / "changed_trials.csv", index=False)
    # Error complementarity: text probability on baseline Other false negatives versus negatives.
    complement = []
    for split, ref, scores, decisions in (("oof", gold, oof_scores, oof_d),
                                           ("temporal_56", late_gold, time_scores, time_d),
                                           ("validation_78", val, val_scores, val_d)):
        truth = ref.Other.to_numpy(dtype=int)
        miss = (truth == 1) & (decisions["baseline"][:, OTHER] == 0)
        nonother = truth == 0
        complement.append({"split": split, "baseline_other_false_negatives": int(miss.sum()),
                           "mean_text_other_p_on_misses": float(scores["text"][miss, OTHER].mean()) if miss.any() else None,
                           "mean_text_other_p_on_nonother": float(scores["text"][nonother, OTHER].mean()),
                           "base_other_average_precision": float(average_precision_score(truth, scores["base"][:, OTHER])),
                           "text_other_average_precision": float(average_precision_score(truth, scores["text"][:, OTHER]))})
    pd.DataFrame(complement).to_csv(OUT / "text_complementarity.csv", index=False)
    audit = audit_labels(train, val, OUT)
    write_json(OUT / "label_audit_summary.json", audit)
    table = pd.DataFrame(all_metrics)
    eligible = []
    for method in ("residual_other", "text_other"):
        metrics = {split: table.loc[(table.split == split) & (table.candidate == method), "sample_f1"].item()
                   for split in ("oof", "temporal_56", "validation_78")}
        baseline = {split: table.loc[(table.split == split) & (table.candidate == "baseline"), "sample_f1"].item()
                    for split in ("oof", "temporal_56", "validation_78")}
        if (metrics["oof"] > baseline["oof"] + 1e-12
                and metrics["temporal_56"] > baseline["temporal_56"] + 1e-12
                and metrics["validation_78"] >= baseline["validation_78"] - 1e-12):
            eligible.append(method)
    selected = sorted(eligible, key=lambda name: (-table.loc[(table.split == "oof") & (table.candidate == name), "sample_f1"].item(), name))[0] if eligible else None
    write_json(OUT / "decision.json", {"selected": selected, "eligible": eligible,
        "thresholds": selected_threshold, "no_new_model_if_failed": True,
        "kaggle_uploaded": False, "original_final_replaced": False,
        "label_changes": 0, "co_training_attempted": False,
        "reason_for_no_co_training": "require evidence of complementary text errors before adding pseudo labels"})
    if selected is not None:
        # Keep a separate candidate; never replace the locked final.
        test = pd.read_csv(dataset_path(ROOT, "part1", "test"))
        full = split_scores(combined, test, include_models=True)
        final_d = candidate_decisions(full, selected_threshold["residual"], selected_threshold["text"])[selected]
        pred = pd.DataFrame(final_d, columns=REASONS)
        pred.insert(0, "ID", test.ID.to_numpy())
        sample = pd.read_csv(dataset_path(ROOT, "part1", "test").parent / "part1_sample_submission.csv")
        submission = build_submission(sample, pred)
        submission.to_csv(OUT / "part1_other_candidate.csv", index=False)
        validate_submission(submission, sample, 1)
        joblib.dump(full["models"], OUT / "selected_models.joblib", compress=3)
        write_json(OUT / "candidate_metadata.json", {"selected": selected,
            "submission_sha256": file_sha256(OUT / "part1_other_candidate.csv"),
            "test_changed_rows": int((final_d != make_decisions("binary", full["base"])).any(axis=1).sum()),
            "kaggle_uploaded": False})
    print(pd.DataFrame(all_metrics).to_string(index=False), flush=True)
    print(json.dumps(json.loads((OUT / "decision.json").read_text()), indent=2), flush=True)




if __name__ == "__main__":
    from replay import main
    main(default_study='other_views')
