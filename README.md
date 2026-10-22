# Calibrated IDS scheduling on Raspberry Pi

Research artifact for TSUSC-2026-07-0200, *Shielded Q-learning for Thermally-Safe, Energy-Aware Multi-Model Intrusion Detection Scheduling on Lightweight Edge Devices*.

## Current release

The current Pi 4B evidence is the corrected R6 campaign: 60 accepted policy/condition/seed cells, 36,000 true 16-record windows, and 576,000 record evaluations. The accepted set contains 58 original valid attempts and replacements for cells 21 and 45; all earlier attempts remain archived. The Pi 3B results and saved official detector test are separate evidence sources.

The earlier Pi 4B V4 campaign did not consistently execute the declared batch size, risk/controller rules, mask, reward, background load, or timing/energy accounting. Its scientific claims are withdrawn. Historical records remain for provenance and must not be merged with R6. The supplied manuscript has now been synchronized with R6; the superseded manuscript is under `history/manuscript_before_r6/`.

| Item | Location |
|---|---|
| Current manuscript, supplement and response | `paper/` |
| Corrected analysis (includes FPR) | `results/pi4b8_r6_revision/` |
| Original R6 analysis, retained unchanged | `results/pi4b8_r6/` |
| Accepted R6 selector | `evidence/pi4b8_corrected/v5_r6_execution/PI4B8_FORMAL_EVIDENCE_FINAL_VALIDATION.json` |
| Frozen R6 runtime | `evidence/pi4b8_corrected/pi4b8_formal_runtime_v5_r6/` |
| Source manifest | `protocols/PI4B8_CORRECTED_RUNTIME_SOURCE_MANIFEST_V5_R6.json` |
| Saved official test | `results/official_test/` |
| Reproduction instructions | `REPRODUCE.md` |
| Correction record | `AUDIT_AND_REPAIR_STATUS.md` |
| Release checksums | `PUBLIC_RELEASE_MANIFEST_SHA256.txt` |

## Findings

Safe-Greedy outperforms FSSQL-R in the Pi 3B stationary binding comparison. In corrected Pi 4B runs, their mean F1 is close under B0/B1; under B2 it is 0.8869 versus 0.9608, with FPR 0.5199 versus 0.1327. The learned policy reaches training-unvisited states in 1,322/3,000 B2 windows and follows its fixed zero-Q tie rule. This is a post hoc execution diagnostic, not a population failure probability.

KM003C records absolute Pi DC input energy over the Mac ACK interval. FSSQL-R minus Safe-Greedy energy differences are +0.908, -1.209, and +0.645 J across B0/B1/B2; no consistent energy-saving claim is made. Each condition contains five matched seeds. Repeated windows and records are not independent statistical samples. The percentile bootstrap and exact sign-flip results are descriptive.

The official detector test was accessed on 11 September 2026. October corrections and corrected Q training use frozen development inputs and validation quality statistics, not another test evaluation. The detector/preprocessor/threshold hashes are unchanged. Do not describe the test as still sealed or the later scheduler study as a fresh blinded test.

## Scope

R6 implements a zero-backlog synchronous workload. Its thermal cap is nonbinding and all four actions are admissible in every measured window. Background processes have PID/affinity/liveness evidence; an independent CPU-utilization time series is unavailable. Controller and E2E timers have explicitly documented exclusions; physical energy includes the full ACK interval and tail work. See the manuscript limitations for reconstruction and historical test-output provenance.

## Verification

```bash
python -m unittest discover -s tests -v
python scripts/validate_public_release.py
# With the archived raw evidence present, use a new output directory:
python scripts/analyze_pi4b8_r6.py --output /tmp/tsusc-r6-check
```

The archived inputs, model files, Q tables, raw measurements and execution manifests are unchanged by the current analysis/manuscript update. Hashes identify bytes; behavior tests and raw-data recomputation establish the more limited checks they actually perform.

Do not commit private input packages, raw UNSW-NB15 files, credentials or local machine paths solely to make a public checkout self-contained. Follow the dataset's applicable distribution terms. The repository update does not run hardware or reopen the official test.
