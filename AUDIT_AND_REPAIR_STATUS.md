# Implementation correction and release status

Updated: 22 October 2026.

## Historical Pi 4B V4

The earlier V4 policy/energy results are excluded from current scientific claims because the executed implementation did not match the declared experiment. Files are retained under their original locations and in the manuscript history, not rewritten as corrected measurements.

## Corrected R6

R6 is complete: 60 accepted cells, 36,000 windows and 576,000 record evaluations. Its runtime artifacts, guarded policies, true n16 batches, label-free reward, background-process evidence and bound meter traces have been reviewed against the source. Cell 21 and 45 temperature-start failures have full retained replacements.

The October 22 analysis update corrects a field-prefix filter that omitted FPR from paired contrasts. The accepted set, raw records, scientific configurations, detector thresholds and Q tables are unchanged. Current analysis is `results/pi4b8_r6_revision/`. The earlier analysis remains under `results/pi4b8_r6/`.

## Current release changes

- All 18 condition-specific policy FPR contrasts are now included, alongside other endpoints.
- Main manuscript, supplement, point-by-point response, cover letter and figures now use R6, not V4 rankings or the old seed-5102 account.
- The test chronology states September 11 access followed by development-only October implementation corrections. No test is reopened.
- Guard tests separate immutable byte identity from cross-platform numerical reproduction; coverage/admission decisions must still match exactly.
- README and reproduction commands point to R6. The release validator no longer presents legacy Q/V4 data as the current experiment.
- Updated checksums describe the new release; historical manifests are retained separately.

## Limits not closed by these changes

Original submitted execution assets remain unavailable. Historical official-test output-copy provenance and ranking predictions are incompletely retained. Pi 4B thermal/queue constraints are not binding; background process duty is configured, not an independently sampled utilization percentage. The support diagnostic is post hoc and the online workload is reused development data. These points remain in the manuscript.

The release changes do not constitute new training, new physical measurements, or certification of the entire historical Pi 3B workflow. Publication decisions and author approval remain separate from code/PDF checks.
