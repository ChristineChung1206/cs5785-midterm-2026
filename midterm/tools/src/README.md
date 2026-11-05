# Code entry points

Run these commands from the `midterm/` root:

- `python reproduce.py`: load saved models and verify all four retained submission
  variants plus the six records in `models/selection.json`; no fitting or upload.
- `python reproduce.py --output-dir NEW_OR_EMPTY_DIRECTORY`: perform the same checks,
  then export the current best files as `part1.csv` and `part2.csv`. Export preserves
  their exact recorded bytes and rejects a nonempty output directory.
- `python tools/src/train_core.py --help`: explicitly train one of six fixed recipes
  into a new output directory. No parameter search or automatic promotion is performed.
- `python report/scripts/report_analysis.py --root .`: rebuild report tables and
  figures from saved predictions, without training.
- `python -m unittest discover -s tools/tests`: run the behavior and migration tests.

The four notebooks under `tools/notebooks/` import these functions and document the
fixed recipes. Default execution uses saved evidence; the explicit training switch
must be enabled deliberately to refit. Notebook execution can update notebook outputs
and derived report files, but does not overwrite production models.

Import modules from `tools/src/`. Their original `midterm_*` names remain intact because
joblib stores custom class module names, particularly `midterm_part1_local_consensus`.
These files provide the current inference, calibration, augmentation and evaluation
functions even when their names refer to earlier experiments. Deleting or renaming
them can break loading or prediction.

Direct historical study CLIs are guarded. An optional replay of the local-model study
is available via `python tools/src/midterm_part1_local_consensus.py --out NEW_DIRECTORY`;
it reruns the historical search and requires a nonexistent directory. Full earlier
source versions and detailed history are preserved under `tools/archive/`, but neither
the archive nor a sibling `deliverables/` directory is needed for saved-model inference.
