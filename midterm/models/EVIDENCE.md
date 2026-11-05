# Retained evidence and how to interpret it

Paths below are relative to the project root (`midterm/`). Each core model keeps its evidence under its own folder.

| Location | Purpose |
| --- | --- |
| `data/audit/`, `data/audit/protocol/` | Source identities, data counts, split rules, and trial-grouped folds |
| `models/part1/01_supervised_lr/evidence/original_run/` | Original supervised LR settings and allow-empty validation predictions |
| `models/part1/02_kmeans_lr/evidence/initial_run/` | K-Means memberships, recovered labels, provenance, and initial classifier results |
| `models/part1/02_kmeans_lr/evidence/augmentation/` | K-Means/GMM comparison and augmentation settings |
| `models/part1/02_kmeans_lr/evidence/selected_run/`, `models/part1/02_kmeans_lr/evidence/selected/` | Selected C=0.1 / threshold=0.35 run, final decision policy, and fixed historical reason lookup |
| `models/part1/02_kmeans_lr/evidence/baseline_parity/` | LR versus K-Means under the same empty-set fallback |
| `models/part1/03_kernel_ridge/evidence/selection_procedure/` | Local-model selection procedure, near groups, repeated OOF, and temporal diagnostics |
| `models/part1/exploration/` | Compact negative studies, including later fixed-ridge comparisons |
| `models/part2/01_tfidf_lr/evidence/` | Outcome baseline predictions, original training configuration, and failure analysis |
| `models/part2/02_text_blend/evidence/` | Historical word/field/character composition and selection evidence |
| `models/part2/03_phase_shared/evidence/selection/` | Shared-component selection, historical holdout, grouped OOF, and final-blend validation predictions |
| `report/analysis/`, `report/figures/` | Consistently recomputed report metrics and figures |
| `tools/archive/records/`, `tools/archive/` | Earlier registries/verification records and the complete verified snapshot |
| `tools/verification/current/` | Verification of the current layout; older layout checks remain separate |

## Decision policies and evaluation scopes

Original run outputs keep their original policy. The supervised LR score **0.8654** and initial K-Means/GMM scores **0.8205 / 0.6667** allow empty reason sets. The report's fair comparison applies threshold then argmax if empty: supervised LR and retained K-Means both score **0.8782** on the same 78 validation trials. Historical evidence has not been recalculated or relabeled to erase that difference.

The local selection-procedure OOF score **0.8983** selects a recipe inside each outer fold. The later **0.9026** score evaluates the already-selected fixed ridge recipe; its baseline rows are retained in `models/part1/exploration/local_followup/`. These are different estimands. The 705 OOF rows are three predictions of each of 235 trials, not 705 independent trials.

Historical ridge validation predictions use a separate fit on the original 157 train labels. The current public model uses all 235 visible labels. Late-20 uses train-only gold, while late-56 combines train and previously inspected validation gold. Both are historical diagnostics; neither erases earlier selection feedback.

## Fixed reason lookup for outcome analysis

Both baseline and current outcome predictions are analyzed with `models/part1/02_kmeans_lr/evidence/selected/reason_lookup.csv`. This keeps reason-group assignments constant. The table prioritizes observed labels and distinguishes predicted labels, training-resubstitution estimates, and unmatched trials. It is not a new lookup made from the 235-label ridge. Multi-reason groups overlap; the analysis is descriptive.

## Frozen records versus current locations

`models/selection.json` is the current identity registry. Historical JSON/CSV paths and hashes describe the code and artifacts at their original execution, and flags such as `kaggle_uploaded: false` are not current submission status. `models/experiment_log.csv` remains unchanged. Use `tools/release/path_migration.json` to locate retained files and `tools/archive/` for the complete snapshot; do not rewrite old observations to match the new layout.
