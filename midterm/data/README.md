# Supplied data and audit

- `part1/`: official reason-prediction train, validation, test, sample submission and pathway catalog.
- `part2/`: official outcome-prediction equivalents.
- `audit/`: original dataset inventory and integrity summaries; `protocol/` contains the earlier
  evaluation protocol and Part 2 trial-grouped folds.

All ten official CSV files preserve their pre-migration bytes. Part 1 has 157 observed training
labels, 1,014 unlabeled training rows, 78 observed validation rows and 312 test rows.
Part 2 has 8,037/1,013/1,500 train/validation/test trial-phase rows; several phase rows may refer
to one trial. Sample submissions show the expected ID order and output format, not test truth.

Generated pseudo-labels and predictions belong under their `models/` experiment evidence.
The later Part 1 near-duplicate grouped protocol is recorded under the kernel-ridge experiment;
it is distinct from the early data/audit/protocol convention.
