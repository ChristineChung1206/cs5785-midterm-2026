# P1-01 — Supervised logistic-regression baseline

`model.joblib` is the original five-reason LR bundle. A StandardScaler fits 1,171 training embeddings; classifiers use only the 157 observed training label vectors. The recipe uses L2 LR, C=1, seed 42, and a 0.5 decision threshold. It does not fit the validation labels.

`evidence/original_run/` contains the original configuration, per-row validation probabilities, support/convergence diagnostics, and scores. Its original allow-empty sample F1 is **0.8654**. Applying the same argmax fallback as submitted models gives **0.8782**; the corrected paired comparison is in `../02_kmeans_lr/evidence/baseline_parity/` and the current report tables in `../../../report/analysis/`.

This is a baseline, not the current public-best model. A separate public score for this exact fitted artifact is not established. Use `../../selection.json` for current model/submission identities. Frozen evidence keeps its original run names and paths; see `../../../tools/release/path_migration.json` and the complete snapshot under `../../../tools/archive/`.
