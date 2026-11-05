# Experiment decisions

Paths below are relative to the project root. `models/selection.json` is the current identity registry; `models/experiment_log.csv` remains the unchanged historical observation log.

## Six core experiment families

| ID | Method | Evaluation evidence | Public result | Decision |
| --- | --- | --- | ---: | --- |
| P1-01 | Five supervised logistic regressions | 157 training labels; validation sample F1 0.8782 with argmax fallback | Not separately established | Baseline |
| P1-02 | K-Means pseudo-labels + weighted LR | Same validation fallback: 0.8782; retain compact GMM and pathway comparisons | 0.85405 | Required unsupervised augmentation study; historical best |
| P1-03 | Calibrated local RBF kernel ridge | Fixed-recipe grouped OOF 0.9026; historical validation 0.8932; final fit uses 235 labels | **0.88568** | Final Part 1 |
| P2-01 | Phase-specific word TF-IDF + LR | Equal-phase validation log loss 0.583773 | 0.60149 | Baseline |
| P2-02 | Word/field/character blend, 25/25/50 | Equal-phase validation log loss 0.580900 | 0.59696 | Retained component of final model |
| P2-03 | 50/50 text blend + phase-shared, recency-weighted LR | Equal-phase validation log loss 0.568393 | **0.59011** | Final Part 2 |

See `models/experiment_index.csv` for the evidence directories and `report/analysis/` for
recomputed, consistently defined metrics. Part 1 thresholds are model-specific, but all
main comparisons use the same empty-set argmax rule. Original LR F1 0.8654 omitted that
rule and must not be compared with 0.8782 as proof of a semi-supervised improvement.
The two models tie in sample F1 while exactly matching 61 and 64 of 78 reason sets.

K-Means versus GMM at the original C=1, threshold 0.5 gave historical validation F1
0.8205 versus 0.6667; adding pathways gave 0.8120. These are original-policy ablations,
not the standardized main comparison. Filled labels, provenance and parameters are retained
in `models/part1/02_kmeans_lr/evidence/initial_run/` and
`models/part1/02_kmeans_lr/evidence/augmentation/`.

## Representative unsuccessful submission

Graph soft labels raised validation sample F1 to 0.8932, but public F1 fell to 0.84081.
Its exact uploaded CSV remains in `models/part1/exploration/graph/submission.csv`.
This documents why local validation improvement alone was insufficient for promotion.

## Exploration summary

| Study | Main observation | Decision |
| --- | --- | --- |
| Baseline parity | With common fallback, LR/K-Means validation tie; grouped OOF 0.8399/0.8875 and late-train-20 0.7500/0.8417 under that study's protocol | Keep corrected comparison |
| Primary/secondary reasons and whitening | Candidate OOF 0.8960, validation 0.8590 versus retained 0.8782 | Reject |
| Recruitment label shift | Late-year F1 0.8393 to 0.8452, gain from one trial | Reject |
| Residual Other and text view | OOF 0.8896 to 0.8917, late-year 0.8393 to 0.8333 | Reject |
| Local ensembles / Other kernel / expected-F1 decoding | Fixed-ridge OOF 0.9026 to 0.9002; historical validation 0.8932 to 0.8761 | Reject; no submission |
| Ridge bandwidth/regularization and group weights | Fixed-ridge OOF 0.9026 to 0.8983; late-56 0.8512 to 0.8333; all three repeats declined | Reject; no submission |

Compact plans, summaries and key predictions are in `models/part1/exploration/`. Detailed grids,
intermediate models and exploratory runners remain in the complete verified snapshot under
`tools/archive/`; retained locations are mapped in `tools/release/path_migration.json`. This table is a decision history, not a common benchmark across rows.

## Protocol distinctions

The initial local-kernel study reports **0.8983 for a nested selection procedure**, compared
with fixed K-Means 0.8877. It selected among local models, optional consensus and blending
inside each outer split. Consensus was selected in 0/12 outer folds; blends in 3/12. The final
full-data selection chose supervised RBF ridge, global Euclidean geometry, bandwidth multiplier
0.5, ridge penalty 0.1 and threshold 0.5. No consensus or blend was selected for that fit.

The later **0.9026 OOF** measures this fixed winning recipe on the same four group folds and
three seeds (42, 137, 2026). It is not the original selection-procedure score. There are 235
labeled trials, not 705 independent trials: 705 counts three repeated predictions per trial.
Near-identical vectors share a group; held groups and their unlabeled counterparts are excluded
from all fitting scopes. Calibration uses grouped cross-fitting inside the training partition.

Historical validation uses only the original 157 training labels for fitting. The final submission then refits
on all 235 visible labels. Its in-sample performance on those labels is not reported as validation.
The 56-row late-year checks include previously viewed validation labels and remain diagnostic.

Part 2's phase-shared candidate was chosen using a 2010–2013 historical holdout and grouped
training folds before checking supplied validation. Historical equal-phase loss improved from
0.58425 to 0.57288 and grouped OOF from 0.59649 to 0.59155. The earlier word/field intermediate
submission scored 0.59962 and remains in the submission history.

All public scores come from user-provided Kaggle screenshots. Hidden labels and private scores
are unavailable. Sequential validation/public-feedback inspection limits generalization claims.

## Historical records and model components

Frozen evidence JSON/CSV files keep their original paths, hashes, and then-current upload flags.
`tools/release/path_migration.json` is the current location map; the prior snapshot in
`tools/archive/` preserves their original context. This reorganization does not rewrite observed
metrics, model recipes, or the historical experiment log.

For P2-03, `models/part2/03_phase_shared/component.joblib` is only the shared component.
The full 0.59011 result uses equal probabilities from it and
`models/part2/02_text_blend/model.joblib`; its submission is
`models/part2/03_phase_shared/submission.csv`.
