# P2-02 — Word, field, and character probability blend

`model.joblib` combines fitted models with probabilities weighted **0.25 concatenated-word / 0.25 separate-field-word / 0.50 character**. Each component retains phase-specific training. `submission.csv` is this historical ensemble's identifiable public submission (**log loss 0.59696**); its equal-phase validation log loss is **0.580900**.

`evidence/original_run/` records the frozen component sources and 25/25/50 composition. `evidence/selection/` records the earlier comparisons, selected validation predictions, and failure analysis. These records keep their original decision history; no component is retrained merely by loading the saved bundle.

This entire text blend supplies **half** the probability in the current Part 2 model. The other half comes from `../03_phase_shared/component.joblib`. The final **0.59011** submission is in `../03_phase_shared/submission.csv`; it is not the score of this bundle alone.

Use `../../selection.json` for exact current identities and composition. Historical paths remain traceable via `../../../tools/release/path_migration.json` and the complete snapshot in `../../../tools/archive/`.
