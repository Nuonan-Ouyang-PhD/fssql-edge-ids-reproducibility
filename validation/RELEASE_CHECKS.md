# Release checks — 22 October 2026

- All 118 local unit tests passed; log: `unit_tests.txt`.
- All 60 selected R6 runs were recomputed from retained raw evidence, including 16 scored records/window, reward sums, masked Q selections, training support and meter integration.
- Five generated CSVs exactly match a second independent invocation; `ANALYSIS_RECOMPUTATION.json`. All 18 policy-pair/condition FPR contrasts are retained.
- 1,541 original frozen input, protocol, calibration, model, Q, runtime and raw evidence files are byte-identical; `FROZEN_INPUTS_CHECK.json`.
- Main (8 pages), response (14), supplement (4), and cover letter (1) compile with no unresolved citations/references. All pages were rendered for layout inspection. Main's final-page vertical-box warning is approximately 1.6 pt; the rendered page has no clipped or overlapping content.
- The main and supplemental tables are generated from the retained seed-level files; the current summaries do not use the withdrawn Pi 4B V4 outcomes.
- No new hardware collection, detector training, Q training, threshold selection or official-test access was performed for this release.

These are checks of the supplied record and current analysis/manuscript, not an external replication of physical execution or a guarantee of journal acceptance. Historical result-copy provenance, thermal/queue nonbinding scope, development-workload reuse, five-pair inference and timing boundaries remain disclosed. The release is ready for coauthor review and submission preparation.

## Manuscript consistency corrections

- Main text, supplement and response now describe Safe-Greedy as maximizing the complete immediate reward under the shared guard, consistent with the frozen executed worker. The incorrect different-objective explanation was removed.
- Q lookup and policy selection are described as separate measured intervals within controller time. Guard subcomponents remain included in their parent totals.
- All three documents and the submission README distinguish derived summaries in the public/source package from the separate R6 raw-evidence archive required for complete recomputation.
- Four PDFs were rebuilt with Tectonic 0.17.0. Explicit T1 encoding in the IEEEtran sources preserves the Times font. Page counts remain 8/4/14/1 for main/supplement/response/cover letter. Final logs contain no unresolved references or citations; all 27 pages were rendered and inspected. Remaining underfull-box warnings do not produce clipped or overlapping content.
- All 118 unit tests passed. The 333 tracked experiment, model, Q, configuration, analysis-result and supporting table/figure files checked against the preceding commit are byte-identical. No experimental or numerical result changes accompany these text corrections.
