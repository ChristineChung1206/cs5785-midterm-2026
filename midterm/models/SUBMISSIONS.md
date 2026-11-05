# Submission identity and composition

Paths below are relative to the project root. Exact hashes, fitted sources, and current status are recorded in [selection.json](selection.json).

| Submission | Public metric | Role |
| --- | ---: | --- |
| `models/part1/03_kernel_ridge/submission.csv` | Sample F1 **0.88568** | Current Part 1 best; fitted 235-label ridge |
| `models/part2/03_phase_shared/submission.csv` | Log loss **0.59011** | Current Part 2 best; 50/50 P2-02 text blend and P2-03 shared component |
| `models/part1/02_kmeans_lr/submission.csv` | Sample F1 0.85405 | Previous Part 1 best |
| `models/part1/exploration/graph/submission.csv` | Sample F1 0.84081 | Rejected graph candidate |
| `models/part2/exploration/word_field/submission.csv` | Log loss 0.59962 | Earlier word/field model |
| `models/part2/02_text_blend/submission.csv` | Log loss 0.59696 | Historical word/field/character model; retained final-model component |

`models/part2/03_phase_shared/component.joblib` produces only the shared model's probability. Reproducing the registered **0.59011** submission also requires `models/part2/02_text_blend/model.joblib` and an equal probability blend. The matching validation evidence is the **blend's** evidence, not a score to attribute to the component alone.

Filenames alone do not identify historical uploads. Scores are user-provided Kaggle screenshot observations; private results are unknown. Current entries use known hashes. Uploads whose exact provenance was not established are not assigned invented model identities.

Frozen November 1 delivery records retain their own names and checksums in `tools/archive/records/`; the full pre-reorganization package is preserved in `tools/archive/`. `tools/release/path_migration.json` provides the old-to-new map. No submissions were uploaded and no Kaggle selections were changed during this reorganization.
