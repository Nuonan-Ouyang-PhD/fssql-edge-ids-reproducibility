# Shielded Q-learning for Thermally-Safe, Energy-Aware Multi-Model Intrusion Detection Scheduling on Lightweight Edge Devices

### Reproducibility artifact for manuscript TSUSC-2026-07-0200

This repository provides the versioned research artifact for the above manuscript submitted to *IEEE Transactions on Sustainable Computing*. It exposes the experimental chain required to examine the paper's central question: whether a device-calibrated feasible-action layer can enforce latency and thermal operating constraints, and when adaptive detector scheduling adds value beyond deterministic controllers.

The repository is organized around executable methods and auditable evidence. It includes portable detector and preprocessing artifacts, Raspberry Pi online runners, device-specific latency and thermal calibration, shield and Q-table lineage, direct POWER-Z acquisition, matched-seed statistical analysis, and frozen result summaries. The manuscript files are retained under [`paper/`](paper/) as the scholarly record associated with the artifact.

| Record | Description |
|---|---|
| Manuscript | `TSUSC-2026-07-0200` |
| Venue | *IEEE Transactions on Sustainable Computing* |
| Artifact version | 21 October 2026 audit-repair branch |
| Evaluation platforms | Raspberry Pi 3B+ and Raspberry Pi 4B 8GB |
| Primary dataset | UNSW-NB15 |
| Physical measurement | POWER-Z KM003C, Pi 4B DC input path |
| Reproduction guide | [`REPRODUCE.md`](REPRODUCE.md) |
| Audit/repair status | [`AUDIT_AND_REPAIR_STATUS.md`](AUDIT_AND_REPAIR_STATUS.md) |
| Integrity record | [`PUBLIC_RELEASE_MANIFEST_SHA256.txt`](PUBLIC_RELEASE_MANIFEST_SHA256.txt) |

## Current audit status

An independent code-to-paper consistency audit on 21 October 2026 found that
the historical Pi 4B V4 online worker did not implement several declared
semantics: it processed one record rather than 16 per window, did not call the
frozen RiskProxy, did not implement the specified Threshold or Safe-Greedy
controllers, used an unmasked fallback after a global Q argmax, logged a
`TP-FP` score instead of the frozen reward, did not apply the thermal guard,
and has no contemporaneous evidence that B1/B2 CPU load was started. Its
timing fields also do not support the claimed decomposition.

The historical V4 files and results remain available as provenance, but the
Pi 4B policy rankings, F1/FPR, reward, B1/B2 robustness, controller/E2E
decomposition, and associated energy comparison are **withdrawn from current
scientific claims pending a corrected preregistered rerun**. They must not be
used as evidence that the declared Pi 4B experiment was completed correctly.

The repair branch makes thresholds/configuration authoritative, enforces true
16-record batches, uses RiskProxy, implements the declared controllers and
joint guard, masks Q selection correctly, separates label-only scoring from
the label-free reward, records independent timings, and fails closed without
an executable background-load specification. It also rejects the legacy
single-record Q tables for a corrected true-`n=16` run. Five seed-specific
corrected Q tables (`5101–5105`) have been trained from development-only inputs,
independently reproduced byte for byte, and hash-frozen. A corrected V5 runtime
R3 package passed the local package and remote inventory/hash gates, but its
remote smoke stopped before inference because the Q manifest bound the complete
pre-hardening worker file. The check was not bypassed. Re-executing the frozen
development-only training with the hardened worker reproduced all five Q tables
and the checkpoint CSV byte for byte; a new manifest changes only that worker
hash binding. A new package revision, remote smoke, corrected commissioning and
corrected formal matrix are still required; no corrected Pi 4B result has been
generated.

This audit does not by itself invalidate the Pi 3B campaign, Pi 4B independent
fit/margin/coverage observations, or the already saved one-shot official-test
aggregate. The official test must not be opened again.

## Scientific scope

FSSQL-R separates feasibility enforcement from utility optimization. A device-specific guard first constructs the admissible detector set using calibrated latency and temperature predictions. Masked tabular Q-learning then chooses only among admitted actions. This separation permits the guard, the learning component, and simpler controllers to be evaluated independently.

The currently usable policy evidence supports a deliberately conditional conclusion. The shield becomes active in the Pi 3B+ binding regime, where Q-learning does not outperform Safe-Greedy on the stationary workload. No current claim about comparative Pi 4B controller performance or energy follows from the historical V4 campaign; those questions require the corrected rerun described above.

## Artifact principles

- Device-specific calibration: Pi 3B+ predictors and margins are never reused on the Pi 4B.
- Evidence separation: scheduler-development deployment results are not presented as independent detector-generalization results.
- Run-level inference: policy comparisons use matched seeds rather than pooled-window significance.
- Outcome-independent retention: technically valid adverse results, tail events, and seed-specific failures remain in the record.
- Physical energy boundary: the historical V4 joules were measured at the Pi
  4B DC input path rather than inferred from utilization or temperature. The
  policy-conditioned comparison is withdrawn pending corrected execution.

## Experimental platform

- Raspberry Pi 3B+ binding-resource campaign, with independent calibration and active cooling.
- Raspberry Pi 4B 8GB independent calibration with active cooling; the historical V4 online campaign is retained as invalid implementation provenance pending a corrected rerun.
- Both campaigns use a 100-ms decision window, a 40-ms latency cap, and an 82°C thermal cap.
- The historical V4 campaign measured power at the Pi 4B DC input path with a
  POWER-Z KM003C (serial `075356`). This is absolute DC input energy, not
  whole-site AC wall-plug energy; its policy comparison is not currently an
  accepted result.
- Historical V4 meter captures used at least 5 s pre-roll and 5 s post-roll.
  Run energy was trapezoidally integrated over Mac monotonic start/end ACK
  boundaries using the recorded timestamps.

## Data, detectors, and controllers

The primary benchmark is UNSW-NB15. The scheduler-development workload uses the frozen labeled development lineage for online deployment comparisons; it is not an independent generalization estimate. The official UNSW-NB15 test was opened once, after all development choices were frozen, and evaluated with fixed preprocessing and threshold values. A later audit found that the historical evaluator embedded those four threshold constants rather than reading `models/MODEL_HYPERPARAMETERS.json`; the constants nevertheless match the frozen values exactly (`0.05`, `0.05`, `0.315`, and `0.065` in action order). The old output-generation/copy lineage is incomplete, so the saved aggregate is retained unchanged and must not be presented as having a fully replayable public provenance. It has not been regenerated. Supplementary material documents TON-IoT, CICIoT2023, and N-BaIoT usage where applicable.

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

### Pi 4B 8GB historical V4 campaign

The repository retains all 60 V4 cells, replacement-attempt lineage and raw
summaries so the failed implementation can be audited. These files are not a
valid realization of the declared `n16/B0`, `n16/B1`, `n16/B2` comparison and
are excluded from current policy, robustness and energy claims. In particular,
the apparent Safe-Greedy all-positive behavior and Threshold/FSSQL-R rankings
were produced by incorrect controller implementations; they are not valid
baseline comparisons. See
[`protocols/PI4B8_CORRECTED_TRUE_N16_PROTOCOL_V1.md`](protocols/PI4B8_CORRECTED_TRUE_N16_PROTOCOL_V1.md)
for the frozen corrected semantics and remaining execution gates.

### Guard calibration and learning stability

- Pi 3B+ held-out joint coverage: `2879/2880 = 0.999653`.
- Pi 4B 8GB held-out joint coverage: `2877/2880 = 0.998958`; all three thermal misses are retained.
- Pi 3B+ Q-learning diagnostics support late-budget behavioral stability on
  visited states, not mathematical convergence over the complete state space.
  The 600-window to 6000-window checkpoint changes are documented in the
  supplement.

### One-shot official detector test

The saved one-shot aggregate covers all 82,332 official UNSW-NB15 test rows.
The numerical threshold values used match the frozen validation thresholds, but
the historical evaluator hard-coded them and the path from its root-level
outputs to the checked-in result directory is not fully documented. The output
files below remain unchanged; the test was not reopened to repair provenance.

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

The frozen protocols/configurations and
[`AUDIT_AND_REPAIR_STATUS.md`](AUDIT_AND_REPAIR_STATUS.md) are the authoritative
protocol-facing sources for this repair branch. The files under `paper/` are a
pre-audit scholarly record and are not submission-ready. A faithful hardware
reproduction requires:

1. Obtain UNSW-NB15 from its official source. In a fresh reproduction, keep the
   official test sealed throughout development and open it once for the final
   independent evaluation. This repository's one authorized access has already
   occurred; do not run it again here.
2. Prepare Raspberry Pi 3B+ and Pi 4B 8GB systems with independent device calibration, active cooling, the 40-ms/82°C caps, and 100-ms decision windows.
3. Use the frozen preprocessing, detector artifacts, risk proxy, device-specific predictors/margins, policy definitions, and seed mapping. Never pass labels into runtime features, state, guard, action selection, Q updates, or reward.
4. For Pi 4B physical-energy runs, connect KM003C serial `075356`, capture at least 5 s before and after each run, retain raw timestamped CSV, and integrate only between the Mac start/end ACK boundaries with trapezoidal integration.
5. Preserve every per-window record, run manifest, hash, invalid attempt, and replacement lineage. Analyze matched seeds at run level; do not substitute pooled-window significance for paired inference.

The public package contains code, frozen runtime artifacts, calibration
evidence, accepted Pi 3B result tables, historical V4 arithmetic/provenance,
figures, and integrity records. The large raw hardware archive is not
duplicated here. Do not infer physical W/J from CPU utilisation, temperature,
or a normalized proxy.

## Repository contents

- `runtime/`: NumPy-only portable preprocessing, detector, and risk-proxy inference.
- `scripts/`: split, training, Q-learning, calibration preparation, validation, and statistics tools.
- `online/pi3b/`: Pi 3B+ binding-sweep and policy campaign runners.
- `online/pi4b8/`: corrected Pi 4B worker, historical disabled V4
  orchestration/replacement tooling, label-leakage utilities, and direct KM003C
  HID capture.
- `device_calibration/`: fit/margin/coverage collection and guard fitting/evaluation code.
- `models/`, `preprocessing/`, `risk_proxy/`, `q_training/`: frozen portable runtime artifacts, legacy lineage and corrected development-only true-`n=16` Q artifacts.
- `configs/` and `protocols/`: scientific parameters, run matrices, preregistration, commissioning, POWER-Z, and final validation records.
- `calibration/`: accepted Pi3B+ and Pi4B device-specific calibration evidence.
- `results/pi3b/`: V3 campaign completion, aggregate results, validation, and violation forensic record.
- `results/pi4b8/`: historical V4 provenance and arithmetic summaries; excluded from current scientific claims pending corrected execution.
- `results/official_test/`: one-shot official-test metrics, validation, decision, and hashes; no raw dataset rows.
- `data/`: dataset hashes, split method/manifest, leakage audit, and preprocessing provenance; `data/raw/` is intentionally user-supplied.
- `paper/`: pre-audit manuscript, supplement, response and cover-letter files
  retained for change tracking; these files are not submission-ready.

## Claim boundaries

The shield provides empirical, device/configuration-specific feasible-set enforcement, not an unconditional complete-pipeline 40-ms guarantee. On the currently usable stationary Pi 3B+ evidence, FSSQL-R does not outperform Safe-Greedy. The historical Pi 4B V4 online result is excluded pending corrected execution, so it cannot support Q-learning, robustness or energy-comparison claims. Only the saved one-shot official UNSW-NB15 evaluation is independent detector-generalization evidence; it must not be reopened or used to tune the repair.
