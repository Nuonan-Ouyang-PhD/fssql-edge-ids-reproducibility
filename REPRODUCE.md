# Reproducing the submission-ready results

## 1. Select the right evidence

Use the accepted Pi4B8 release described by the manifest below. The release contains the current analysis, manuscript, tests and frozen evidence references; it does not train a new policy or alter accepted raw evidence.

The R6 accepted-cell selector is:

```text
evidence/pi4b8_corrected/v5_r6_execution/PI4B8_FORMAL_EVIDENCE_FINAL_VALIDATION.json
```

It binds 60 accepted cells, including the authorized replacements for cells 21 and 45. Never mix power from one attempt with predictions from another.

## 2. Environment and files

Use Python with NumPy and pandas for analysis and Matplotlib for figures. The original `requirements-lock.txt` and frozen runtime dependency manifest describe the execution environment; do not overwrite them with the analyst's environment. `validation/environment.json` records the environment used for this release check.

For a public checkout, restore the exact archived evidence and non-public development inputs into their recorded relative paths before tests that need those files. The complete reproduction archive includes the R6 evidence. Missing inputs are not permission to regenerate labels from predictions or open the official test.

## 3. File identity and behavior tests

```bash
python scripts/validate_public_release.py
python -m unittest discover -s tests -v
```

The local release check ran 118 tests. Strict SHA checks apply to stored frozen artifacts. Re-fitting the calibration on another numerical platform uses absolute tolerance 1e-10 and relative tolerance 1e-12, and additionally requires exactly unchanged held-out coverage and cap-admission decisions. This is not permission to edit the frozen predictor files.

## 4. Recompute the accepted R6 analysis

Run from any working directory. Use a new output directory; the analyzer refuses to overwrite an existing one.

```bash
python scripts/analyze_pi4b8_r6.py --output /tmp/tsusc-r6-recomputed
```

The script checks the complete 4-policy x 3-condition x 5-seed grid; verifies the raw Pi, power and run-manifest hashes; requires 600 windows and 16 scored records/window; checks reward component sums; re-integrates each power trace on valid ACK boundaries; and reproduces masked Q choices and training support. It does not train or execute hardware.

`results/pi4b8_r6_revision/` is the derived reference. It includes FPR contrasts, all five matched seeds, exact two-sided sign-flip tests over 32 assignments, and paired percentile intervals using 2,000 resamples with seed 20261022. P-values are unadjusted descriptive endpoints; zero-variance dz is left undefined.

## 5. Tables, figures and PDFs

```bash
python scripts/build_paper_tables.py
python scripts/plot_pi4b8_revision.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error response_to_reviewers.tex
pdflatex -interaction=nonstopmode -halt-on-error cover_letter.tex
```

The response retains the editor/reviewer/author/change/excerpt box format. Main and supplement tables point to the same corrected analysis. Compiler logs and temporary files are not submission material. IEEEtran and standard LaTeX packages are required.

## 6. Existing guard reproduction (no new collection)

Fit/margin arguments are **split directories**, not `raw_windows.csv` filenames:

```bash
python device_calibration/fit_guard_predictors.py \
  --collection-config calibration/pi4b8/PI4B8_CALIBRATION_CONFIG.json \
  --predictor-config configs/pi4b8_guard_predictor_protocol_v1.json \
  --fit calibration/pi4b8/formal_fit_20260910_v3 \
  --margin calibration/pi4b8/formal_margin_20260910 \
  --output /tmp/tsusc-pi4b-guard-check
```

The evaluator likewise takes a coverage split directory. Do not tune predictors or margins from held-out coverage. A local numerical reproduction does not replace the original frozen artifacts or constitute a new calibration experiment.

## 7. Hardware execution is separate

No new run is needed to reproduce the current tables. The executed package is `evidence/pi4b8_corrected/pi4b8_formal_runtime_v5_r6/`; the final authorization is under `v5_r6_authorization/`. The worker and Mac orchestrator are hash-bound there. Any future run needs its own plan, output directory and preserved execution identity.

The model remains resident. CPU-load process execution, timing exclusions, Mac/Pi clock domains and meter-boundary coverage must be verified independently of filename/configuration labels. Raw meter packets and CSVs are kept unchanged.

## 8. Official test is already accessed

The saved 82,332-row detector results are in `results/official_test/`. The one-shot lock prevents repeat execution. Do not reopen the test or use its outcomes to choose correction parameters. The saved confusion metrics can be algebraically checked.
