# P2-01 — Phase-specific word TF-IDF + LR

`model.joblib` is the original outcome baseline: one training-only word TF-IDF vocabulary and LR classifier for each of Phases I, II, and III. It concatenates criteria, disease, and drug text; word unigrams/bigrams use min_df=3 and at most 20,000 features. LR uses C=1 and seed 42. Reasons and reason embeddings are excluded from predictors.

The model's equal-phase validation log loss is **0.583773**; its user-reported public log loss is **0.60149**. `evidence/original_run/` contains settings, training diagnostics, and saved validation predictions. `evidence/analysis/` contains baseline failure/reason analysis and its provenance.

This is the baseline, not the current public-best blend. `../../selection.json` identifies current artifacts. Historical evidence keeps its original paths and state; `../../../tools/release/path_migration.json` maps retained files, and `../../../tools/archive/` holds the full snapshot.
