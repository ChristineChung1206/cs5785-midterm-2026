# Report and reproducible analysis

`final_report.pdf` is the current report: five body pages and one references page, using the original unmodified CVPR style. The final submission audit adds the overall Part 2 metrics, explicitly discusses the validation/public gap, and cites EM/BBSE label-shift methods. Existing experiment results and selected models are unchanged.

## Four-directory package

At the project root (`midterm/`):

- `data/`: supplied Part 1 and Part 2 CSVs plus data audits and evaluation protocols.
- `models/`: model artifacts, submissions, and historical evidence grouped by task and model; `models/selection.json` identifies current public-best artifacts.
- `tools/`: implementation modules, tests, executable notebooks, and archived studies.
- `report/`: this PDF and its analysis, figures, LaTeX sections, table generators, and verification records.

## Rebuild the saved-evidence analysis

From the project root:

```sh
python report/scripts/report_analysis.py
python report/scripts/generate_tables.py
```

These commands locate the project relative to their own files, so an independently extracted package works without an original-machine path or `PYTHONPATH`. Analysis imports shared evaluation code from `tools/src/`, reads model-specific saved predictions and `models/selection.json`, and verifies selected submission hashes. It writes `report/analysis/` and `report/figures/`. Table generation reads `report/analysis/report_metrics.json` and writes `report/tables/`. Neither command fits a model or runs inference.

The historical Part 1 validation comparisons use 157-label fits, including the saved fixed-ridge reference, rather than evaluating the final 235-label model in-sample. Part 2 before/after failure analysis uses the same frozen historical reason lookup, distinguishing observed, predicted, training-resubstitution, and unmatched records. Trial-phase rows are not unique trials. Repeatedly inspected validation evidence is diagnostic, not a new untouched test.

## Compile the PDF

From this directory:

```sh
tectonic --only-cached --keep-logs main.tex
mv main.pdf final_report.pdf
mv main.log verification/compile.log
```

The tested compiler is Tectonic 0.17.0 with cached TeX packages and installed Times New Roman/Courier New fonts. A first-time installation needs its TeX packages populated before using `--only-cached`. The XeTeX font mapping uses font names rather than absolute paths. Standard pdfLaTeX retains the normal Times configuration from unmodified `cvpr.sty`; that alternative was not tested here.

`main.tex` includes `sections/*.tex`. `verification/source_provenance.json` records current relative evidence paths, hashes, and evaluation scope. `verification/layout_verification.json` records the newly compiled PDF/source hashes and visual inspection of all six pages. Rebuilding may change PDF metadata bytes; rerender and inspect the new artifact before replacing its layout audit. Earlier verification records remain in the full pre-reorganization snapshot rather than being presented as verification of new files.
