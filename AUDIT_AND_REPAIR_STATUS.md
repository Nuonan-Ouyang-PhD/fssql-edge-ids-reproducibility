# Release status

This public repository contains the submission-ready TSUSC Major Revision artifact. The current release is synchronized across the manuscript, response letter, supplementary material, analysis tables, figures, validation scripts and reproducibility instructions.

## Evidence scope

- Pi3B+ policy evidence and Pi4B8 online evidence are reported as separate device studies.
- Pi4B8 contains 60 accepted policy/condition/seed cells, 36,000 windows and 576,000 record evaluations.
- Physical power is measured at the Raspberry Pi DC input using the KM003C and integrated over the recorded Mac ACK interval.
- The official detector test was accessed once; its frozen outputs are independent detector-generalization evidence and are not used for scheduler decisions.

## Methodological limits

The online workload is a frozen scheduler-development workload. The Pi4B8 thermal constraint is nonbinding in the measured campaign, and the background-load evidence records process liveness and CPU affinity rather than an independent utilization time series. The paired inference uses five matched seeds per condition and is descriptive.

The release does not claim universal superiority of Q-learning or unconditional complete-pipeline latency guarantees. Safe-Greedy and Threshold remain explicit baselines, and adverse detector and policy results are retained in the current tables.

## Verification

The release validator, 118 unit tests, raw-evidence hash checks, energy recomputation and manuscript artifacts were checked before publication. No new training, hardware collection or official-test access is part of this release.
