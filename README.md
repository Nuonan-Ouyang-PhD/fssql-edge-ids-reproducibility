# Shielded Q-learning for Thermally-Safe, Energy-Aware Multi-Model Intrusion Detection Scheduling on Lightweight Edge Devices

### Reproducibility artifact for manuscript TSUSC-2026-07-0200

This repository provides the versioned research artifact for the above manuscript submitted to *IEEE Transactions on Sustainable Computing*. It exposes the experimental chain required to examine the paper's central question: whether a device-calibrated feasible-action layer can enforce latency and thermal operating constraints, and when adaptive detector scheduling adds value beyond deterministic controllers.

The repository is organized around executable methods and auditable evidence. It includes portable detector and preprocessing artifacts, Raspberry Pi online runners, device-specific latency and thermal calibration, shield and Q-table lineage, direct POWER-Z acquisition, matched-seed statistical analysis, and frozen result summaries. The manuscript files are retained under [`paper/`](paper/) as the scholarly record associated with the artifact.

| Record | Description |
|---|---|
| Manuscript | `TSUSC-2026-07-0200` |
| Venue | *IEEE Transactions on Sustainable Computing* |
| Artifact version | 15 September 2026 revision |
| Evaluation platforms | Raspberry Pi 3B+ and Raspberry Pi 4B 8GB |
| Primary dataset | UNSW-NB15 |
| Physical measurement | POWER-Z KM003C, Pi 4B DC input path |
| Reproduction guide | [`REPRODUCE.md`](REPRODUCE.md) |
| Integrity record | [`PUBLIC_RELEASE_MANIFEST_SHA256.txt`](PUBLIC_RELEASE_MANIFEST_SHA256.txt) |

## Scientific scope

FSSQL-R separates feasibility enforcement from utility optimization. A device-specific guard first constructs the admissible detector set using calibrated latency and temperature predictions. Masked tabular Q-learning then chooses only among admitted actions. This separation permits the guard, the learning component, and simpler controllers to be evaluated independently.

The evidence supports a deliberately conditional conclusion. The shield becomes active in the Pi 3B+ binding regime, but remains nonbinding in the evaluated Pi 4B conditions. Q-learning does not outperform Safe-Greedy on the stationary Pi 3B+ workload; on the Pi 4B it improves over a degenerate Safe-Greedy reference, while the no-training Threshold controller remains the strongest detector-selection baseline by mean F1. Direct power measurements quantify the operating trade-off but do not demonstrate a consistent FSSQL-R energy saving.

## Artifact principles

- Device-specific calibration: Pi 3B+ predictors and margins are never reused on the Pi 4B.
- Evidence separation: scheduler-development deployment results are not presented as independent detector-generalization results.
- Run-level inference: policy comparisons use matched seeds rather than pooled-window significance.
- Outcome-independent retention: technically valid adverse results, tail events, and seed-specific failures remain in the record.
- Physical energy boundary: reported joules are measured at the Pi 4B DC input path and are not inferred from utilization or temperature.

## Experimental platform

- Raspberry Pi 3B+ binding-resource campaign, with independent calibration and active cooling.
- Raspberry Pi 4B 8GB online campaign, with independent calibration and active cooling.
- Both campaigns use a 100-ms decision window, a 40-ms latency cap, and an 82°C thermal cap.
- Physical power is measured at the Pi 4B DC input path with a POWER-Z KM003C (serial `075356`). Reported energy is absolute DC input energy, not whole-site AC wall-plug energy.
- Formal meter captures include at least 5 s pre-roll and 5 s post-roll. Run energy is trapezoidally integrated over Mac monotonic start/end ACK boundaries using the recorded timestamps; the raw trace is retained unchanged.

## Data, detectors, and controllers

The primary benchmark is UNSW-NB15. The scheduler-development workload uses the frozen labeled development lineage for online deployment comparisons; it is not an independent generalization estimate. The official UNSW-NB15 test was opened once, after all development choices were frozen, and evaluated with fixed preprocessing and thresholds. Supplementary material documents TON-IoT, CICIoT2023, and N-BaIoT usage where applicable.

**Detector actions:** FISVDD, LUCID, TinyDL, and OI-SVDD+AS-ELM.

**Scheduling policies:** FSSQL-R, Safe-Greedy, Threshold, Static-Light, Round-Robin, and Unshielded-Q.

Labels are isolated from runtime decision inputs. Features and risk features drive preprocessing, state construction, guard evaluation, policy selection, and detector execution. Ground-truth labels are joined only after prediction for TP/FP/FN/TN scoring.

## Experimental evidence

### Pi 3B+ binding-resource regime

The accepted matrix contains 6 policies × 10 matched seeds × 600 windows (36,000 windows). The guard has exactly three admissible actions in every window and excludes LUCID.

| Policy | Mean F1 | Mean reward | Mean E2E (ms) | Switch cost | Violations |
|---|---:|---:|---:|---:|---:|
| FSSQL-R | 0.8523 | 0.5434 | 8.3149 | 5.348 | 0 |
| Safe-Greedy | **0.8708** | **0.5839** | **7.9775** | **0** | 0 |
| Threshold | 0.8704 | 0.5834 | 7.9704 | 0.012 | 1 |
| Unshielded-Q | 0.8523 | 0.5489 | 10.1141 | 4.438 | 0 |
| Round-Robin | 0.8148 | 0.4801 | 10.6571 | 11.980 | 0 |
| Static-Light | 0.7203 | 0.3389 | 8.2550 | 0 | 0 |

FSSQL-R therefore does not outperform Safe-Greedy in this stationary binding regime. The single retained Threshold complete-pipeline tail event was 52.522 ms and was dominated by controller/runtime overhead rather than the guarded detector prediction.

### Pi 4B 8GB online deployment and physical energy

The formal online matrix contains 4 policies × 3 conditions (`n16/B0`, `n16/B1`, `n16/B2`) × 5 matched seeds (`5101`–`5105`) × 600 windows = 60 cells and 36,000 windows. The guard is nonbinding in these conditions (`|A_safe|=4`), so this is a deployment/controller/energy comparison rather than a binding-shield stress test.

| Condition | Policy | F1 | FPR median [range] | Cumulative reward | P99 E2E (ms) | Controller (ms) | Energy (J) | Viol. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | FSSQL-R | 0.9235±0.0534 | 0.1917 [0.0674, 0.9144] | 333.8±52.2 | 5.386±0.151 | 3.909±0.164 | 173.93±12.46 | 0 |
| B0 | Safe-Greedy | 0.8025±0.0117 | **1.0000 [1.0000, 1.0000]** | 204.4±19.5 | 5.063±0.279 | 3.872±0.023 | 174.40±9.42 | 0 |
| B0 | Static-Light | 0.9332±0.0120 | 0.2190 [0.1865, 0.2767] | 346.2±17.9 | 5.727±0.936 | 4.114±0.170 | 175.47±19.98 | 0 |
| B0 | Threshold | **0.9491±0.0117** | **0.0714 [0.0622, 0.0874]** | **362.0±16.5** | 5.436±1.118 | 3.834±0.189 | 183.66±6.24 | 0 |
| B1 | FSSQL-R | 0.9271±0.0580 | 0.1869 [0.0444, 0.9780] | 341.6±58.3 | 5.021±0.328 | 3.897±0.156 | 164.50±0.58 | 0 |
| B1 | Safe-Greedy | 0.8099±0.0117 | **1.0000 [1.0000, 1.0000]** | 216.8±19.9 | 5.150±0.542 | 3.879±0.066 | 167.93±3.85 | 0 |
| B1 | Static-Light | 0.9387±0.0065 | 0.2278 [0.1832, 0.2602] | 356.2±9.5 | 5.481±0.223 | 4.075±0.019 | 168.80±5.08 | 0 |
| B1 | Threshold | **0.9533±0.0106** | **0.0444 [0.0354, 0.1173]** | **371.0±15.1** | 7.927±5.707 | 4.258±1.067 | 181.38±10.36 | 1 |
| B2 | FSSQL-R | 0.9222±0.0711 | 0.1882 [0.0378, 0.9476] | 336.0±83.5 | 5.352±0.576 | 3.951±0.119 | 171.32±8.16 | 0 |
| B2 | Safe-Greedy | 0.8081±0.0199 | **1.0000 [1.0000, 1.0000]** | 214.0±33.8 | 5.318±0.573 | 3.905±0.093 | 172.53±8.11 | 0 |
| B2 | Static-Light | 0.9374±0.0071 | 0.2270 [0.1832, 0.2381] | 354.2±21.1 | 8.919±7.298 | 4.643±1.193 | 175.17±21.45 | 1 |
| B2 | Threshold | **0.9627±0.0040** | **0.0471 [0.0378, 0.0647]** | **377.0±13.6** | **4.913±0.283** | **3.747±0.026** | 168.24±4.56 | 0 |

Safe-Greedy is retained because it isolates the one-step decision rule under the same guard; however, FPR=1.0000 and TN=0 in all 15 cells make it a degenerate detector-selection reference in this experiment. Threshold has the highest mean F1 and lowest mean FPR in every Pi 4B load. Accordingly, the FSSQL-R improvement over Safe-Greedy is interpreted as recovery from an all-positive policy, not as evidence of overall superiority.

The dispersion of FSSQL-R is driven by a reproducible seed-specific mode:

| Seed 5102 condition | FISVDD selections | F1 | FPR |
|---|---:|---:|---:|
| B0 | 582/600 | 0.8285 | 0.9144 |
| B1 | 592/600 | 0.8245 | 0.9780 |
| B2 | 588/600 | 0.7967 | 0.9476 |

Formal evaluation uses `epsilon=0` and disables Q/replay updates. The retained result is therefore described as a stable FISVDD-dominated evaluation policy under sparse visited-state support, rather than as an exploration event or complete-state convergence claim.

### Guard calibration and learning stability

- Pi 3B+ held-out joint coverage: `2879/2880 = 0.999653`.
- Pi 4B 8GB held-out joint coverage: `2877/2880 = 0.998958`; all three thermal misses are retained.
- Q-learning diagnostics support late-budget behavioral stability on visited states, not mathematical convergence over the complete state space. The 600-window to 6000-window checkpoint changes are documented in the supplement.

### One-shot official detector test

All 82,332 official UNSW-NB15 test rows were evaluated exactly once with frozen preprocessing and thresholds.

| Detector | F1 | Precision | Recall | FPR | AUROC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| FISVDD | 0.7102 | 0.5506 | 1.0000 | 1.0000 | 0.5351 | 0.6219 |
| LUCID | 0.8240 | 0.7020 | 0.9973 | 0.5186 | 0.8621 | 0.8469 |
| TinyDL | **0.8686** | **0.7802** | 0.9797 | **0.3382** | **0.9704** | **0.9784** |
| OI-SVDD+AS-ELM | 0.8521 | 0.7437 | **0.9975** | 0.4211 | 0.9513 | 0.9584 |

FISVDD predicts every test row positive under its frozen threshold, and its AUROC of 0.5351 is close to random. This is a ranking failure, not merely a threshold-calibration issue.

## Quick reproduction

Create the frozen Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

Place the official UNSW-NB15 training and testing CSVs under `data/raw/` using the exact filenames documented in [REPRODUCE.md](REPRODUCE.md), then verify the source hashes in `data/DATASET_SHA256.json`. The development pipeline is:

```bash
python scripts/split_development_data.py
python scripts/fit_preprocessor.py
python scripts/train_action_pool.py
python scripts/train_risk_proxy.py
python scripts/export_portable_preprocessor.py
python scripts/verify_portable_inference.py
```

The checked-in artifacts are frozen outputs of this pipeline. These scripts fail rather than overwrite existing frozen outputs, so reproduce in a clean checkout or an isolated output copy.

### Paper build

Install a LaTeX distribution with `IEEEtran`, `booktabs`, `amsmath`, `hyperref`, `siunitx`, and `graphicx`, then run:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
pdflatex supplement.tex
pdflatex response_to_reviewers.tex
pdflatex cover_letter.tex
```

### Evidence-aware experiment reproduction

The manuscript and supplement are the authoritative protocol-facing description. A faithful hardware reproduction requires:

1. Obtain UNSW-NB15 from its official source. Keep the official test sealed until all development/model/threshold/guard decisions are frozen; access it once for the independent evaluation.
2. Prepare Raspberry Pi 3B+ and Pi 4B 8GB systems with independent device calibration, active cooling, the 40-ms/82°C caps, and 100-ms decision windows.
3. Use the frozen preprocessing, detector artifacts, risk proxy, device-specific predictors/margins, policy definitions, and seed mapping. Never pass labels into runtime features, state, guard, action selection, Q updates, or reward.
4. For Pi 4B physical-energy runs, connect KM003C serial `075356`, capture at least 5 s before and after each run, retain raw timestamped CSV, and integrate only between the Mac start/end ACK boundaries with trapezoidal integration.
5. Preserve every per-window record, run manifest, hash, invalid attempt, and replacement lineage. Analyze matched seeds at run level; do not substitute pooled-window significance for paired inference.

The public package contains code, frozen runtime artifacts, calibration evidence, run-level/paired result tables, figures, and integrity records. The large raw hardware archive is not duplicated here. Do not infer physical W/J from CPU utilisation, temperature, or a normalized proxy.

## Repository contents

- `runtime/`: NumPy-only portable preprocessing, detector, and risk-proxy inference.
- `scripts/`: split, training, Q-learning, calibration preparation, validation, and statistics tools.
- `online/pi3b/`: Pi 3B+ binding-sweep and policy campaign runners.
- `online/pi4b8/`: Pi 4B worker, commissioning/formal orchestration, label-leakage gate, replacement runner, and direct KM003C HID capture.
- `device_calibration/`: fit/margin/coverage collection and guard fitting/evaluation code.
- `models/`, `preprocessing/`, `risk_proxy/`, `q_training/`: frozen portable runtime artifacts and lineage records.
- `configs/` and `protocols/`: scientific parameters, run matrices, preregistration, commissioning, POWER-Z, and final validation records.
- `calibration/`: accepted Pi3B+ and Pi4B device-specific calibration evidence.
- `results/pi3b/`: V3 campaign completion, aggregate results, validation, and violation forensic record.
- `results/pi4b8/`: 60-cell run-level metrics, paired-seed statistics, tables, figures, and accepted-cell integrity records.
- `results/official_test/`: one-shot official-test metrics, validation, decision, and hashes; no raw dataset rows.
- `data/`: dataset hashes, split method/manifest, leakage audit, and preprocessing provenance; `data/raw/` is intentionally user-supplied.
- `paper/`: revised manuscript, supplement, response letter, cover letter, PDFs, bibliography, and figures.

## Claim boundaries

The shield provides empirical, device/configuration-specific feasible-set enforcement, not an unconditional complete-pipeline 40-ms guarantee. Q-learning benefit is device- and regime-dependent: it is unfavorable to FSSQL-R on the stationary Pi 3B+ binding regime but positive relative to the degenerate Safe-Greedy control on Pi 4B, where Threshold remains strongest by mean F1. POWER-Z results quantify a measured DC-input operating trade-off and do not demonstrate a universal FSSQL-R energy saving. Pi 4B online F1 is a frozen scheduler-development deployment result; only the one-shot official UNSW-NB15 evaluation is independent detector-generalization evidence.
