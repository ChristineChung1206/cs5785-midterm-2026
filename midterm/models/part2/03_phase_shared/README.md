# P2-03 — Phase-shared component and final outcome blend

`component.joblib` is a word-text LR component sharing coefficients across phases, with phase interactions scaled by **0.35** and training recency weights with a **six-year half-life**. Outcome training rows determine its fit; supplied validation outcomes are not added to training. Start year changes training weights, not the prediction feature vector. Stopping reasons and embeddings are excluded.

The complete current model averages this component's probability with `../02_text_blend/model.joblib`, **50/50**. `submission.csv` stores that complete blend, with public log loss **0.59011** and equal-phase validation log loss **0.568393**. These blend scores must not be attributed to `component.joblib` by itself. Exact composition and hashes are in `../../selection.json`.

`evidence/selection/` contains the historical 2010–2013 holdout, trial-grouped training OOF, locked setting, validation control/candidate predictions, and per-phase metrics. The stored candidate validation is the selected blend. Before/after failure analysis in `../../../report/analysis/` uses the same historical Part 1 reason lookup for both outcome models.

Historical evidence remains unchanged, including then-current names and upload flags. See `../../../tools/release/path_migration.json` for retained locations and `../../../tools/archive/` for the complete verified snapshot.
