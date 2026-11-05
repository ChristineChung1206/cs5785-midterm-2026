# Models and experiment evidence

Each core family keeps its fitted artifact, retained evidence, and any identifiable submission together. Paths in the table are relative to this directory.

| Family | Fitted artifact | Role |
| --- | --- | --- |
| [P1-01 supervised LR](part1/01_supervised_lr/README.md) | `part1/01_supervised_lr/model.joblib` | Original 157-label baseline |
| [P1-02 K-Means + LR](part1/02_kmeans_lr/README.md) | `part1/02_kmeans_lr/model.joblib` | Required augmentation study and previous public best |
| [P1-03 calibrated RBF ridge](part1/03_kernel_ridge/README.md) | `part1/03_kernel_ridge/model.joblib` | Current Part 1 public best, fitted with 235 labels |
| [P2-01 word TF-IDF + LR](part2/01_tfidf_lr/README.md) | `part2/01_tfidf_lr/model.joblib` | Phase-specific baseline |
| [P2-02 word/field/character blend](part2/02_text_blend/README.md) | `part2/02_text_blend/model.joblib` | Historical submission and half of the current outcome ensemble |
| [P2-03 phase-shared component](part2/03_phase_shared/README.md) | `part2/03_phase_shared/component.joblib` | Other half of the current outcome ensemble |

## Current selections

[selection.json](selection.json) is the current registry of exact model/submission paths, hashes, composition, and user-reported public scores. Part 1 scores **0.88568 sample F1**. Part 2 scores **0.59011 log loss** and combines P2-02 and P2-03 probabilities with weights **0.5 / 0.5**. `component.joblib` alone is not the complete Part 2 model.

Run `python reproduce.py` from the project root to load saved fits and verify the registered submissions. The loader adds `tools/src/` to the module path so original custom pickle classes retain their identities. Model and historical prediction bytes are preserved; directory reorganization does not alter any recipe or retrain anything.

[experiment_index.csv](experiment_index.csv) is the current six-family navigation index, with paths relative to the project root. [experiment_log.csv](experiment_log.csv) is the unchanged early-run observation log; it is not a current file-location registry or a complete list of later explorations. See [EXPERIMENTS.md](EXPERIMENTS.md), [EVIDENCE.md](EVIDENCE.md), and [SUBMISSIONS.md](SUBMISSIONS.md) for the distinctions.

Historical evidence may contain its original paths, run IDs, hashes, or `kaggle_uploaded: false` flags. Those describe that run's state at the time. [Path migration](../tools/release/path_migration.json) maps retained files, and [the archive](../tools/archive/) contains the complete verified pre-reorganization snapshot and earlier records. Do not edit frozen evidence to make its historical names look current.
