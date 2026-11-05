"""Fixed small Gaussian-mixture comparison after the locked round-2 selection.

Exploratory only: it cannot retroactively change selection or claim independent
validation, because the first round-2 validation results were already inspected.
"""
from __future__ import annotations

from _historical_support import prepare_output

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from midterm_augmentation import observed_label_mask
from midterm_evaluation import REASONS, dataset_path
from midterm_next_round import p1_assignment
from midterm_part1_round2 import CovarianceGeometry, make_decisions, metrics_rows, score
from midterm_refinement import EMBEDDINGS

ROOT = None  # Set only by explicit replay(root, output).
OUT = None
DIMS = (30, 60, 100)


def fit_mixture(frame, dimensions):
    labels = frame[list(REASONS)].to_numpy()
    observed = observed_label_mask(labels)
    single = observed & ((labels == 1).sum(axis=1) == 1)
    transform = CovarianceGeometry(dimensions).fit(frame[EMBEDDINGS].to_numpy())
    x = transform.transform(frame[EMBEDDINGS].to_numpy())[single]
    y = labels[single].argmax(axis=1)
    models = []
    for j in range(len(REASONS)):
        # Rare classes need one Gaussian when a fold has fewer than four seeds.
        n_components = 2 if (y == j).sum() >= 4 else 1
        model = GaussianMixture(n_components=n_components, covariance_type="diag", reg_covar=.1,
                                n_init=3, max_iter=200, random_state=42)
        with threadpool_limits(limits=1):
            model.fit(x[y == j])
        if not model.converged_:
            raise RuntimeError("Class mixture failed to converge.")
        models.append(model)
    return {"transform": transform, "models": models,
            "prior": np.bincount(y, minlength=len(REASONS)) / len(y)}


def predict(bundle, frame):
    x = bundle["transform"].transform(frame[EMBEDDINGS].to_numpy())
    logp = np.column_stack([model.score_samples(x) for model in bundle["models"]])
    return softmax(logp + np.log(bundle["prior"])[None, :], axis=1)


def run(root, output):
    global ROOT, OUT
    ROOT, OUT = prepare_output(root, output)
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(dataset_path(ROOT, "part1", "train"))
    val = pd.read_csv(dataset_path(ROOT, "part1", "val"))
    gold, assignment = p1_assignment(train)
    folds = assignment.set_index("ID").loc[gold.ID, "fold"].to_numpy()
    oof = {dimension: np.full((len(gold), len(REASONS)), np.nan) for dimension in DIMS}
    for fold in range(4):
        held = gold.loc[folds == fold]
        fit = train.loc[~train.ID.isin(held.ID)]
        assert set(fit.ID).isdisjoint(held.ID)
        for dimension in DIMS:
            oof[dimension][folds == fold] = predict(fit_mixture(fit, dimension), held)
        print(f"Mixture OOF {fold + 1}/4", flush=True)
    temporal_fit = train.loc[train.start_year <= 2006]
    temporal_hold = gold.loc[gold.start_year.between(2007, 2009)]
    rows = []
    for dimension in DIMS:
        name = f"mixture_{dimension}"
        od = make_decisions("primary", oof[dimension])
        td = make_decisions("primary", predict(fit_mixture(temporal_fit, dimension), temporal_hold))
        vd = make_decisions("primary", predict(fit_mixture(train, dimension), val))
        rows.extend(metrics_rows(name, "oof", gold, od))
        rows.extend(metrics_rows(name, "temporal", temporal_hold, td))
        rows.extend(metrics_rows(name, "validation", val, vd))
        print(name, "OOF", score(gold[list(REASONS)], od),
              "time", score(temporal_hold[list(REASONS)], td),
              "val", score(val[list(REASONS)], vd), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    (OUT / "status.json").write_text(json.dumps({
        "fixed_model": "up to two diagonal Gaussians per single-label class (one if fewer than four seeds), empirical class prior",
        "geometry": "Ledoit-Wolf covariance whitening", "dimensions": DIMS,
        "selection_status": "exploratory_after_round2_validation_inspection",
        "validation_is_independent_for_this_comparison": False,
        "selected_model_changed": False, "test_prediction_written": False}, indent=2) + "\n")




if __name__ == "__main__":
    from replay import main
    main(default_study='mixture_followup')
