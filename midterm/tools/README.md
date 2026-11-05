# Project tools

- `src/`: importable model, feature, evaluation and training modules. Old module names remain necessary for serialized models.
- `notebooks/`: five readable project walkthroughs and a runner; default execution does not train models.
- `src/historical/`: reported historical experiment runners, a guarded replay entry, and source provenance.
- `tests/`: unit and integration-contract tests; run from the project root using `python -m unittest discover -s tools/tests`.
- `verification/layout_v2/`: frozen November 2 validation evidence.
- `verification/current/`: verification of the four-folder migration and executed notebooks.
- `release/`: package builder, standalone package verifier, path mapping and current release receipt.
- `archive/`: complete pre-migration snapshot and compact historical records.

Report-specific generators live in `report/scripts/`. These tools do not submit to Kaggle.
