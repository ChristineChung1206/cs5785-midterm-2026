# P1-03 — Calibrated RBF kernel ridge

`model.joblib` and `submission.csv` are the current Part 1 public-best artifacts: **sample F1 0.88568**. Exact hashes and paths belong to `../../selection.json`.

The selected recipe uses a single global scale, an RBF kernel with gamma = 0.5 / median nonzero squared distance among fitting gold vectors, ridge penalty 0.1, grouped cross-fit sigmoid calibration, and threshold 0.5 plus argmax for an empty set. Its final fit uses **235 visible label vectors**. Eligible gold and unlabeled rows supply the geometry pool (235 + 1,014); regression and calibration use gold targets. No consensus pseudo-label student or final fusion was selected.

`evidence/selection_procedure/` preserves the original 24-configuration local-model search, grouping, fold selections, and diagnostics. Its final **OOF 0.8983** evaluates the nested selection procedure. The later fixed-ridge **OOF 0.9026** is a different evaluation, stored as baseline evidence in `../exploration/local_followup/`. Neither should be substituted for the other.

Historical validation **0.8932** comes from a separate original-157-label fit, not the final 235-label artifact. The late-56 diagnostic includes previously viewed validation labels. All 235 labels and earlier public feedback had informed the development history; repeated OOF is not a new external holdout.

Frozen evidence keeps historical file names and paths. Use `../../../tools/release/path_migration.json` for locations and `../../../tools/archive/` for the complete verified snapshot. Normal notebook execution demonstrates saved inference; optional training remains disabled.
