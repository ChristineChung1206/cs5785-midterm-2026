# P1-02 — K-Means pseudo-labels and weighted LR

`model.joblib` is the retained historical K-Means + LR bundle. The unsupervised stage uses 10 clusters on standardized training embeddings and maps them to reasons using the 157 observed labels. All 1,014 masked rows receive recovered labels; gold/pseudo weights are 1/0.1. The selected classifier uses C=0.1 and shared threshold 0.35, followed by argmax if no reason passes.

`submission.csv` is the previous public-best Part 1 file (**0.85405**). Its 78-trial validation sample F1 is **0.8782**, equal to supervised LR under the same nonempty decision rule. This model was not refitted with validation labels.

| Evidence folder | Meaning |
| --- | --- |
| `evidence/initial_run/` | Fixed initial K-Means mapping, memberships, recovered labels, provenance, and C=1 results |
| `evidence/augmentation/` | Required K-Means/GMM comparison and settings; original policy allows empty sets |
| `evidence/selected_run/` | Selected C=0.1 / threshold=0.35 run and its lineage |
| `evidence/selected/` | Final historical decision policy, validation, verification, and reason lookup |
| `evidence/baseline_parity/` | Corrected common-fallback comparison against supervised LR |

`evidence/selected/reason_lookup.csv` deliberately stays the fixed historical lookup for Part 2 before/after analysis. It prioritizes 235 observed train/validation labels, estimates the rest with this model, and flags training-resubstitution predictions. Having observed validation reasons in the lookup does not mean this classifier trained on them.

Use `../../selection.json` for the current best model. Historical paths/flags remain observations of their original run. `../../../tools/release/path_migration.json` maps retained files; `../../../tools/archive/` preserves the complete snapshot.
