# Delivery tools

`build_release.py` validates saved-model inference and the verified report, then creates a NEW
output directory containing exactly four delivery files. Canonical CSVs and PDF are copied
without changing their bytes. A new `midterm_code.zip` is built from the active project.

The ZIP excludes full historical ZIP archives, Python/Notebook caches, report build intermediates,
and the local release receipt. It includes all runtime data/model/evidence dependencies, report
sources, the verified PDF, compact history records and an embedded `package_manifest.json`.

`verify_package.py` checks all inventoried bytes after extraction; run it before executing notebooks
or generators that intentionally rewrite derived outputs. `reproduce.py` then verifies actual model
inference and can export both current predictions. No external original-project path is required.

`current_release.json` records the four deliverable checksums, sizes and verification result;
it is deliberately outside the code ZIP to avoid a checksum cycle.
`path_migration.json` maps every file from layout v2 to its layout v3 location. Original observations
inside historical JSON/CSV documents are unchanged and can be interpreted using this map.
`migration_verification.json` documents byte preservation and completed verification for this move.

Before packaging, the builder checks restored historical sources and saved replay inputs,
and the core/report source copies embedded in notebook 05. After editing core/report code,
refresh these review copies with `python tools/release/sync_source_appendix.py --write`.
