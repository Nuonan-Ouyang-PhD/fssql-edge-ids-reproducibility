# Calibrated IDS scheduling on Raspberry Pi

Research artifact for TSUSC-2026-07-0200, *Shielded Q-learning for Thermally-Safe, Energy-Aware Multi-Model Intrusion Detection Scheduling on Lightweight Edge Devices*.

## Submission-ready release

This repository contains the synchronized manuscript, reproducible analysis, frozen runtime specifications, and accepted evidence for the current release. The Pi 4B8 campaign contains 60 accepted policy/condition/seed cells, 36,000 true 16-record windows, and 576,000 record evaluations. The Pi 3B+ results and the one-shot official detector test are reported as separate evidence sources.

| Item | Location |
|---|---|
| Current manuscript, supplement and response | `paper/` |
| Corrected analysis (includes FPR) | `results/pi4b8_r6_revision/` |
| Pi4B8 analysis | `results/pi4b8_r6_revision/` |
| Accepted R6 selector | `evidence/pi4b8_corrected/v5_r6_execution/PI4B8_FORMAL_EVIDENCE_FINAL_VALIDATION.json` |
| Frozen R6 runtime | `evidence/pi4b8_corrected/pi4b8_formal_runtime_v5_r6/` |
| Source manifest | `protocols/PI4B8_CORRECTED_RUNTIME_SOURCE_MANIFEST_V5_R6.json` |
| Saved official test | `results/official_test/` |
| Reproduction instructions | `REPRODUCE.md` |
| Audit and repair record | `AUDIT_AND_REPAIR_STATUS.md` |
| Release checksums | `PUBLIC_RELEASE_MANIFEST_SHA256.txt` |

The public repository package does not include the 60 Pi4B8 raw window files or the 60 raw KM003C meter captures. Complete R6 raw-evidence recomputation requires the existing evidence archive referenced by the accepted manifest; the public package is not itself a complete raw-measurement archive.

## Findings

Safe-Greedy outperforms FSSQL-R in the Pi 3B stationary binding comparison. In Pi4B8 runs, their mean F1 is close under B0/B1; under B2 it is 0.8869 versus 0.9608, with FPR 0.5199 versus 0.1327. The learned policy reaches training-unvisited states in 1,322/3,000 B2 windows and follows its fixed zero-Q tie rule. This is a post hoc execution diagnostic, not a population failure probability.

KM003C records absolute Pi DC input energy over the Mac ACK interval. FSSQL-R minus Safe-Greedy energy differences are +0.908, -1.209, and +0.645 J across B0/B1/B2; no consistent energy-saving claim is made. Each condition contains five matched seeds. Repeated windows and records are not independent statistical samples. The percentile bootstrap and exact sign-flip results are descriptive.

The official detector test was accessed once on 11 September 2026. The detector/preprocessor/threshold hashes are frozen; no test-driven tuning was performed. The detector-generalization result is separate from the scheduler-development online result.

## Scope

The release uses a zero-backlog synchronous workload. Its thermal cap is nonbinding and all four actions are admissible in every measured window. Background processes have PID/affinity/liveness evidence; an independent CPU-utilization time series is unavailable. Controller and E2E timers have explicitly documented exclusions; physical energy includes the full ACK interval and tail work.

## Verification

```bash
python -m unittest discover -s tests -v
python scripts/validate_public_release.py
# With the archived raw evidence present, use a new output directory:
python scripts/analyze_pi4b8_r6.py --output /tmp/tsusc-r6-check
```

The archived inputs, model files, Q tables, raw measurements and execution manifests are unchanged by the analysis/manuscript release. Hashes identify bytes; behavior tests and raw-data recomputation establish the checks they actually perform.

Do not commit private input packages, raw UNSW-NB15 files, credentials or local machine paths solely to make a public checkout self-contained. Follow the dataset's applicable distribution terms. The repository update does not run hardware or reopen the official test.
