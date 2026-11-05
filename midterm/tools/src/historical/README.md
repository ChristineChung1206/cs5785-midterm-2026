# Historical experiment source

These ten modules recover the original code behind the report's historical comparisons.
`source_provenance.json` records the source archive/member, original SHA-256, migrated
SHA-256 and the six small restored evidence files. The archives are provenance references;
they are not needed at runtime. No numerical experiment was rerun for this restoration.

Use `tools/notebooks/05_historical_experiments.ipynb` for recorded summaries, source/hash
checks, full source listings and an optional replay cell. Its default
`RUN_HISTORICAL_TRAINING=False` skips all fitting. The four core notebooks and
`tools/src/train_core.py` cover the fixed core model recipes.

```sh
python tools/src/historical/replay.py --check
python tools/src/historical/replay.py --study label_shift --run --output models/new_experiments/label_shift_replay
```

The second command explicitly trains a historical study. It requires a new, nonexistent
output directory, rejects writes inside curated data/evidence/report/tools folders, checks
that the existing data/models remain byte-identical, and does not select a new production
model or upload any prediction. Other choices are printed by `--help`. The `local_selection`
choice uses the existing core local-consensus study module; the ten remaining runners live
here. Their module names stay stable for any historical objects created by a replay.

Filesystem paths and execution entry points were ported to the four-folder layout. Model
settings, fitting equations, folds and selection rules were preserved. New-library refits
may differ numerically; only compile/import, source integrity, available inputs and output
safeguards were checked during restoration. These checks are not a claim that full searches
were rerun. The two recovered OOF CSVs are original saved inputs used by selective-pseudo
and Other-view comparisons; their original bytes and hashes are retained.

No restored source imports `midterm_l2_geometry.py`. The normalization alternatives in the
reported nested study are implemented in the included core
`midterm_part1_local_consensus.Geometry` class. Earlier standalone L2 searches are not
required to execute these reported experiments.

## Earlier original source references

Notebook 05 also preserves twelve exact older experiment, diagnostic and delivery/report scripts.
These frozen original paths are not current replay entry points and may require old artifacts.
Their source SHA-256 and original members are recorded in `tools/archive/records/early_source_reference.json`.
The supported replay studies above were checked for source/import/input availability; historical searches
were not rerun during this submission audit.
