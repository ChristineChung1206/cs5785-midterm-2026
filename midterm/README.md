# Clinical Trial Midterm Project 2026

## Team members and contributions

- Pei-Chen Chung (NetID: pc786)
- Yu-Jye Hu (NetID: yh2444)

Both team members contributed equally to data preprocessing, exploratory data analysis,
feature engineering, model development, performance evaluation, and preparation of the final
report. They shared responsibilities, jointly discussed key modeling decisions, and contributed
equally to the final deliverables. Frozen historical source copies retain their original wording;
the current two-author attribution supersedes those older individual-project descriptions.

## Current delivery

| Task | Selected method | Recorded best public score | Canonical prediction |
| --- | --- | ---: | --- |
| Part 1 | Calibrated RBF kernel ridge, 235 visible labels | sample F1 **0.88568** | `models/part1/03_kernel_ridge/submission.csv` |
| Part 2 | 50/50 text blend and phase-shared recency model | log loss **0.59011** | `models/part2/03_phase_shared/submission.csv` |

Public scores are from the user-provided November 2 Kaggle screenshots, not a fresh leaderboard query.
Private scores are unknown. `models/selection.json` records exact model/prediction hashes and
the Part 2 component weights. Neither final model was retrained during the layout migration.

The parent `deliverables/` contains exactly `part1.csv`, `part2.csv`, `final_report.pdf`,
and `midterm_code.zip`. The CSVs are byte-identical copies of the registered best-public files.
The old November 1 delivery and both earlier archives are inside the verified snapshot under
`tools/archive/`; they have not been discarded or relabeled as current results.

## Four folders

| Folder | Contents |
| --- | --- |
| `data/` | Unchanged official Part 1/2 CSVs; separate audit tables and split protocol |
| `models/` | Model families with fitted artifacts, original evidence, predictions and decisions |
| `report/` | LaTeX sections, generated tables/figures, report analysis, scripts, PDF and verification |
| `tools/` | Shared Python modules, notebooks, tests, verification, delivery tooling and archival snapshot |

Keep the few entry files here: this README, requirements, `reproduce.py`, and ignore rules.
The six experiment families are indexed in `models/experiment_index.csv`; read
`models/EXPERIMENTS.md` for the experiment narrative and evaluation caveats.

## Environment and saved-model reproduction

Use Python 3.12 and the pinned dependencies:

```sh
python -m pip install -r requirements.txt
python reproduce.py
python reproduce.py --output-dir /path/to/new-predictions
python -m unittest discover -s tools/tests
```

The default reproduction verifies four saved-model predictions (two historical and two current)
and all six registered submission identities. The optional output directory receives the two
verified best-public CSVs as `part1.csv` and `part2.csv`; existing nonempty destinations are rejected.
This uses saved models and checks their outputs, without fitting models or uploading to Kaggle.

Fixed-recipe retraining is a separate, explicit operation:

```sh
python tools/src/train_core.py --help
```

Training requires a new output directory and never replaces the selected models automatically.
The serialized Part 1 objects use custom classes in `tools/src/midterm_part1_local_consensus.py`;
keep the original module and class names importable. The selected ridge fit does not use consensus
pseudo-labels or a final model blend, despite the historical module name.

## Notebook and report walkthrough

Read `tools/notebooks/01_data_protocol.ipynb` through `05_historical_experiments.ipynb`.
Run them with `python tools/notebooks/run_all.py`. Their default `RUN_TRAINING=False` and `RUN_HISTORICAL_TRAINING=False` settings
prevent model fitting; execution does regenerate report tables/figures and save notebook outputs.

```sh
python report/scripts/report_analysis.py
python report/scripts/generate_tables.py
```

These commands rebuild analysis from saved predictions. The current report is
`report/final_report.pdf`; see `report/README.md` for compilation and PDF verification.
Report-only Python dependencies are listed in `report/requirements.txt`.

## Build and verify the delivery

```sh
python tools/release/build_release.py --output-dir /path/to/new-deliverables
```

The destination must not exist and must be outside this `midterm/` folder. This validates
saved predictions and report provenance, then creates exactly four files. It excludes full
historical ZIP snapshots, caches, and TeX build intermediates; compact historical provenance
records remain included. Reported historical experiment runners and their required compact inputs
are included separately, so replay does not require an archival ZIP. The release receipt is `tools/release/current_release.json`.

After extracting `midterm_code.zip` into a new location, enter its `midterm/` directory:

```sh
python tools/release/verify_package.py
python reproduce.py --output-dir /path/to/new-predictions
```

The code ZIP contains its own data, models, necessary evidence, report and source modules.
Notebook 05 provides historical replay entries and a readable source appendix. Its final frozen-source
section preserves older independent studies and superseded utilities as exact references; those original
paths are not current replay entry points. All supported training
requires an explicit opt-in and a new output directory.
It does not require the original project path, the sibling deliverables folder, or the full
historical ZIP. Check the package manifest before commands that regenerate recorded outputs.

## Interpretation and history

The original 78-row Part 1 validation has subsequently been reused for the final 235-label fit.
Historical validation fits and repeated/grouped OOF remain explicitly distinguished in the report.
Part 2 failure analysis uses a fixed historical K-Means reason lookup so before/after changes
reflect outcome models; estimated reasons and training resubstitution are not observed labels.

Historical evidence retains the original run's settings, paths, upload flags and source hashes.
Use `tools/release/path_migration.json` for the current relocation and the older map in
`tools/archive/records/path_migration.json` for pre-November-2 paths. Do not interpret a historical
`final`, `selected`, or `kaggle_uploaded: false` value as current selection status.
The exact pre-migration bytes are recorded in `tools/archive/snapshot_manifest.json`.
