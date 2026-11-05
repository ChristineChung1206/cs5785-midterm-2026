"""Train one of six fixed core recipes into a new output directory.

Examples: python tools/src/train_core.py part1_ridge --output models/retrained_ridge
          python tools/src/train_core.py part2_shared --output models/retrained_shared

This is an explicit training entry point, with no search or automatic promotion.
Part 1 LR/K-Means use the supplied training split; the selected ridge uses all
235 visible labels plus the 1,014 unlabeled training embeddings for geometry.
Part 2 uses outcome_train only. Use reproduce.py to verify the preserved models;
retraining can vary across library versions and is not a byte-for-byte restore.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from midterm_augmentation import (cluster_memberships, estimate_cluster_reason_rates,
                                  observed_label_mask, recover_labels)
from midterm_evaluation import PHASES, REASONS, dataset_path, file_sha256
from midterm_finalization import predict_final_outcomes
from midterm_next_round import CHAR_TFIDF
from midterm_part1_finalization import fit_classifiers, predict_bundle
from midterm_part1_local_consensus import Data, EMB, fit_calibrated, near_groups
from midterm_part2_baseline import fit_phase_baseline, predict_outcome_probabilities
from midterm_rank_part2 import blend, fit_phase_shared, predict_phase_shared
from midterm_refinement import TFIDF, fit_field_outcomes

FAMILIES = ("part1_supervised", "part1_kmeans", "part1_ridge",
            "part2_baseline", "part2_text_blend", "part2_shared")
RIDGE_CONFIG = dict(geometry="global", family="ridge", bandwidth=.5, strength=.1)
SHARED_SETTING = dict(alpha=.35, half_life=6)
RECIPES = {
    "part1_supervised": dict(scaler_scope="all training embeddings", classifier_scope="observed training labels",
                            C=1., threshold=.5, seed=42),
    "part1_kmeans": dict(clusters=10, smoothing=2., pseudo_threshold=.5, pseudo_weight=.1,
                        C=.1, threshold=.35, seed=42),
    "part1_ridge": dict(**RIDGE_CONFIG, threshold=.5, calibration_C=1., calibration_folds=3,
                       calibration_seed=942, scope="train plus validation; test excluded"),
    "part2_baseline": dict(C=1., max_word_features=20000, seed=42, scope="outcome_train"),
    "part2_text_blend": dict(word=.25, field=.25, character=.5, seed=42, scope="outcome_train"),
    "part2_shared": dict(**SHARED_SETTING, reference_year=2013, text_blend_weight=.5,
                        shared_weight=.5, seed=42, scope="outcome_train"),
}


def fit_part1_supervised(train):
    """Train-fold scaler over all rows; C=1 classifiers over observed labels only."""
    labels = train[list(REASONS)].to_numpy(dtype=int)
    observed = observed_label_mask(labels)
    scaler = StandardScaler().fit(train[EMB])
    x = scaler.transform(train[EMB])
    classifiers, _ = fit_classifiers(x[observed], labels[observed],
                                     np.ones(observed.sum()), C=1., seed=42)
    return dict(scaler=scaler, feature_columns=EMB, reason_columns=list(REASONS),
                classifiers=classifiers, pathway_encoder=None,
                thresholds=dict.fromkeys(REASONS, .5))


def fit_part1_kmeans(train):
    """K=10, smoothed hard pseudo-labels at .5, weight .1, C=.1, output .35."""
    labels = train[list(REASONS)].to_numpy(dtype=int)
    observed_label_mask(labels)
    scaler = StandardScaler().fit(train[EMB])
    x = scaler.transform(train[EMB])
    clusterer = KMeans(n_clusters=10, n_init=20, max_iter=500, tol=1e-4,
                       random_state=42, algorithm="lloyd").fit(x)
    memberships = cluster_memberships(clusterer, x, "kmeans")
    mapping = estimate_cluster_reason_rates(memberships, labels, smoothing_strength=2.)
    recovered = recover_labels(labels, memberships, mapping, threshold=.5, pseudo_weight=.1)
    classifiers, _ = fit_classifiers(x, recovered["labels"], recovered["sample_weight"],
                                     C=.1, seed=42)
    return dict(scaler=scaler, feature_columns=EMB, reason_columns=list(REASONS),
                classifiers=classifiers, pathway_encoder=None, clusterer=clusterer,
                reason_mapping=mapping, method="kmeans", thresholds=dict.fromkeys(REASONS, .35))


def ridge_data(train, validation):
    """Build the original pooled scope without loading any test features."""
    train, validation = train.copy(), validation.copy()
    observed = observed_label_mask(train[list(REASONS)].to_numpy())
    if not validation[list(REASONS)].isin([0, 1]).all(axis=None):
        raise ValueError("Ridge final fitting requires observed validation labels.")
    train["fit_order"] = np.arange(len(train))
    validation["fit_order"] = len(train) + np.arange(len(validation))
    frame = pd.concat([train.loc[observed].assign(source="train"),
                       validation.assign(source="validation"),
                       train.loc[~observed].assign(source="unlabeled")], ignore_index=True)
    if not frame.ID.is_unique:
        raise ValueError("Part 1 train and validation IDs must be unique and disjoint.")
    x, y = frame[EMB].to_numpy(), frame[list(REASONS)].to_numpy(dtype=int)
    n_gold = int(observed.sum()) + len(validation)
    return Data(frame, x, y, near_groups(x), np.arange(n_gold), np.arange(n_gold, len(frame)))


def fit_part1_ridge(train, validation):
    """Fixed selected kernel with three group-separated calibration folds, seed 942."""
    data = ridge_data(train, validation)
    scope = data.scope(data.gold, np.arange(len(data.x)))
    return fit_calibrated(scope, dict(RIDGE_CONFIG), seed=942)


def fit_part2_baseline(train):
    return fit_phase_baseline(train, tfidf_parameters={**TFIDF, "max_features": 20000},
                              classifier_parameters={"C": 1.}, seed=42)[0]


def fit_part2_text_blend(train):
    """The retained 25/25/50 concatenated word / separate field / character recipe."""
    baseline = fit_part2_baseline(train)
    field, _ = fit_field_outcomes(train)
    character = fit_phase_baseline(train, tfidf_parameters=CHAR_TFIDF,
                                   classifier_parameters={"C": 1.}, seed=42)[0]
    common = dict(weights=[.5, .5], positive_class=1, reason_features_used=False,
                  thresholds=dict.fromkeys(PHASES, .5))
    words = dict(kind="equal_probability_blend", baseline=baseline, field=field, **common)
    return dict(kind="word_character_probability_blend", word_blend=words,
                character=character, flat_weights={"concatenated_word": .25,
                    "separate_field_word": .25, "character": .5}, **common)


def fit_part2_shared(train):
    """The retained 50/50 text-blend / phase-shared recency-weighted recipe."""
    return dict(text_blend=fit_part2_text_blend(train),
                shared=fit_phase_shared(train, dict(SHARED_SETTING), reference_year=2013),
                shared_weight=.5)


def train_family(root: Path, family: str):
    """Return a self-contained fitted bundle; never save or select a production model."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown family: {family}")
    task = "part1" if family.startswith("part1_") else "part2"
    train_path = dataset_path(root, task, "train")
    train = pd.read_csv(train_path)
    inputs = {str(train_path.relative_to(root)): file_sha256(train_path)}
    with threadpool_limits(limits=1):
        if family == "part1_ridge":
            val_path = dataset_path(root, "part1", "val")
            validation = pd.read_csv(val_path)
            inputs[str(val_path.relative_to(root))] = file_sha256(val_path)
            model = fit_part1_ridge(train, validation)
            label_ids = train.loc[observed_label_mask(train[list(REASONS)].to_numpy()), "ID"].tolist() + validation.ID.tolist()
            feature_ids = train.ID.tolist() + validation.ID.tolist()
        else:
            fitters = dict(part1_supervised=fit_part1_supervised, part1_kmeans=fit_part1_kmeans,
                           part2_baseline=fit_part2_baseline, part2_text_blend=fit_part2_text_blend,
                           part2_shared=fit_part2_shared)
            model = fitters[family](train)
            label_ids = (train.loc[observed_label_mask(train[list(REASONS)].to_numpy()), "ID"].tolist()
                         if task == "part1" else train.ID.tolist())
            feature_ids = train.ID.tolist()
    return dict(family=family, model=model, decision_threshold=.35 if family == "part1_kmeans" else .5,
                empty_set_policy="argmax" if task == "part1" else None,
                metadata=dict(recipe=dict(RECIPES[family]), input_sha256=inputs, fit_feature_ids=feature_ids,
                              fit_observed_label_ids=label_ids, seed=42,
                              calibration_seed=942 if family == "part1_ridge" else None,
                              validation_labels_used=family == "part1_ridge", test_features_used=False,
                              hyperparameter_search=False, production_selection_changed=False))


def predict_core(bundle, frame):
    """Return reason probabilities (Part 1) or ID/p_success (Part 2)."""
    family, model = bundle["family"], bundle["model"]
    if family in ("part1_supervised", "part1_kmeans"):
        return predict_bundle(model, frame)
    if family == "part1_ridge":
        return model.predict(frame[EMB].to_numpy())
    if family == "part2_baseline":
        return predict_outcome_probabilities(model, frame)
    if family == "part2_text_blend":
        return predict_final_outcomes(model, frame)
    if family == "part2_shared":
        return blend(frame, predict_final_outcomes(model["text_blend"], frame),
                     predict_phase_shared(model["shared"], frame), model["shared_weight"])
    raise ValueError(f"Unknown family: {family}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing directories are rejected.")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fitted = train_family(args.project_root.resolve(), args.family)
    joblib.dump(fitted, output / "model.joblib", compress=3)
    manifest = dict(family=args.family, decision_threshold=fitted["decision_threshold"],
                    empty_set_policy=fitted["empty_set_policy"], **fitted["metadata"],
                    model_sha256=file_sha256(output / "model.joblib"))
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(family=args.family, output=str(output), production_selection_changed=False)))


if __name__ == "__main__":
    main()
