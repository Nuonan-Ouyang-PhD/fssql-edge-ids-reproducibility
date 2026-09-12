# Reproduction guide

This guide separates software verification, development reconstruction, Raspberry Pi calibration, online execution, physical-energy acquisition, and one-shot test evaluation. Run each stage in a fresh checkout or isolated output directory because frozen-output scripts deliberately refuse to overwrite existing artifacts.

## 1. Environment

Reference software versions are pinned in `requirements-lock.txt`.

```bash
git clone https://github.com/Nuonan-Ouyang-PhD/fssql-edge-ids-reproducibility.git
cd fssql-edge-ids-reproducibility
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

For KM003C access on macOS, install `hidapi` and ensure the current user can access the meter. Raspberry Pi execution additionally requires SSH connectivity and the packages listed in the same lock file.

## 2. Dataset placement and integrity

The public repository does not redistribute UNSW-NB15. Obtain the official training and testing CSVs from the dataset owner and place them as:

```text
data/raw/UNSW_NB15_training-set.csv
data/raw/UNSW_NB15_testing-set.csv
```

Verify both files against `data/DATASET_SHA256.json`. Do not substitute similarly named preprocessed mirrors. The test CSV must remain unopened during development reconstruction.

## 3. Development split, preprocessing, and detectors

The deterministic split is group-safe with respect to exact feature duplicates, uses seed `20260909`, and maps five stratified group folds to model training (0–2), validation (3), and scheduler development (4).

Run these in an isolated copy where their outputs do not already exist:

```bash
python scripts/split_development_data.py
python scripts/fit_preprocessor.py
python scripts/train_action_pool.py
python scripts/train_risk_proxy.py
python scripts/export_portable_preprocessor.py
python scripts/verify_portable_inference.py
```

Expected audit/artifact locations:

```text
data/SPLIT_MANIFEST.csv
data/LEAKAGE_AUDIT.json
data/PREPROCESSOR_PROVENANCE.json
preprocessing/preprocessor.joblib
preprocessing/preprocessor_portable.npz
models/artifacts/
models/MODEL_SHA256.json
models/PORTABLE_INFERENCE_PARITY.json
risk_proxy/risk_proxy.npz
```

The four action models are FISVDD, LUCID, TinyDL, and OI-SVDD+AS-ELM. Thresholds and hyperparameters are recorded in `models/MODEL_HYPERPARAMETERS.json`.

## 4. Q-learning reconstruction

The Q-learning protocol is frozen in `configs/q_learning_protocol.json`; Pi4B-specific lineage and its erratum are under `q_training/pi4b8_q_v1/`.

```bash
python scripts/train_q_tables.py \
  --config configs/q_learning_protocol.json \
  --output q_training_reproduced
```

The checked-in seed-specific tables have `q_values` shape `1280 × 4`. Evaluation uses epsilon 0 with no Q or replay updates. The reported convergence result is late-budget behavioral stability on visited states, not proof of full-state mathematical convergence.

## 5. Device-specific guard calibration

The required sequence is `preflight → fit → margin → freeze → held-out coverage`. Pi3B+ and Pi4B parameters must never be reused across devices.

Collection interface:

```bash
python device_calibration/collect_calibration_split.py \
  --config <device-calibration-config.json> \
  --split fit \
  --run-id <unique-run-id> \
  --output <new-output-directory> \
  --git-commit <source-commit>
```

Repeat with `--split margin` and `--split coverage`, preserving the frozen order and configuration. Fit the predictors only after fit and margin are complete:

```bash
python device_calibration/fit_guard_predictors.py \
  --collection-config <device-calibration-config.json> \
  --predictor-config <device-guard-config.json> \
  --fit <fit/raw_windows.csv> \
  --margin <margin/raw_windows.csv> \
  --output <new-frozen-predictor-directory>
```

Evaluate once on the held-out coverage split:

```bash
python device_calibration/evaluate_guard_coverage.py \
  --collection-config <device-calibration-config.json> \
  --predictor-config <device-guard-config.json> \
  --predictors <new-frozen-predictor-directory> \
  --coverage <coverage/raw_windows.csv> \
  --output <new-coverage-evaluation-directory>
```

Accepted reference outputs are in `calibration/pi3b/protocol_v2/` and `calibration/pi4b8/`.

## 6. Pi3B+ binding-resource campaign

The V3 protocol is frozen in `protocols/BINDING_STRESS_PROTOCOL_V3_FREEZE.md` and `configs/pi3b_stress_extension_v3.json`. The formal matrix is `configs/pi3b_stress_extension_v3_matrix.csv`.

```bash
python online/pi3b/run_stress_extension_v3_campaign.py \
  --config configs/pi3b_stress_extension_v3.json \
  --matrix configs/pi3b_stress_extension_v3_matrix.csv \
  --input <frozen-scheduler-development-input> \
  --output <new-campaign-directory> \
  --git-commit <source-commit>
```

The reference campaign contains 6 policies × 10 matched seeds × 600 windows. Do not use policy outcomes to alter the workload or guard.

## 7. Pi4B online runtime

The Pi worker is `online/pi4b8/pi4b8_online_worker.py`. Its decision path is:

```text
features/risk features
  → preprocessing/state/telemetry
  → Pi4B guard and admissible set
  → policy action
  → selected detector prediction
  → post-action label scoring
  → reward and per-window evidence
```

A direct worker invocation from a frozen Pi runtime package is:

```bash
python online/pi4b8/pi4b8_online_worker.py \
  --features <features.npy> \
  --risk-features <risk_features.npy> \
  --labels <scoring-only-labels.npy> \
  --policy FSSQL-R \
  --condition n16/B0 \
  --seed 5101 \
  --run-id <unique-run-id> \
  --q-table q_training/pi4b8_q_v1/tables/fssql_r_seed_5101.npz \
  --output <new-output-directory> \
  --windows 600
```

The label file is a scoring sidecar. The worker reads its scoring label only after prediction. Run `python online/pi4b8/test_label_leakage.py` before deployment.

The original formal Mac orchestrator and direct meter implementation are retained under `online/pi4b8/`. Hostnames and deployment directories in the frozen orchestrator are environment-specific and must be replaced only for an independent reproduction—not for reinterpretation of the published evidence.

## 8. POWER-Z KM003C acquisition

Measurement boundary: DC input-side power/energy measured inline on the Raspberry Pi 4B 8GB supply path.

A commissioning capture:

```bash
python online/pi4b8/powerz_hid_capture.py \
  --output commissioning_power.csv \
  --seconds 30 \
  --sps 10
```

Do not assume exact 10.000 SPS. Integrate power using recorded monotonic timestamps and the trapezoidal rule. For each online run:

1. Start meter capture at least 5 s before the Pi run.
2. Record Mac monotonic start-request and start-ACK timestamps.
3. Execute all 600 Pi windows.
4. Record Mac monotonic end-ACK.
5. Continue capture for at least 5 s.
6. Hash and bind the unchanged meter CSV to exactly one run ID.
7. Integrate only from the Mac start-ACK boundary to the end-ACK boundary.
8. Preserve invalid attempts; never combine Pi metrics from one attempt with energy from another.

The frozen specification is `protocols/POWER_Z_HID_ACQUISITION_V1.md`.

## 9. Statistical reproduction

The checked-in Pi4B run-level data are `results/pi4b8/RUN_LEVEL_METRICS.csv`; paired results are `results/pi4b8/PAIRED_SEED_RESULTS.json`.

The analysis implementation is `analyze_pi4b8_formal.py`. It expects the private raw evidence layout used by the original campaign. To recompute from a locally restored raw archive, place that archive under `evidence/pi4b8_formal_matrix/` with the accepted-cell manifest and run:

```bash
python analyze_pi4b8_formal.py
```

The statistical unit is the matched seed/run, never a pooled window. With five Pi4B pairs per condition, the minimum nonzero two-sided exact Wilcoxon p-value is 0.0625; paired t tests, exact Wilcoxon tests, bootstrap 95% confidence intervals, and paired effect sizes are all retained.

## 10. One-shot official-test evaluator

`run_official_test_once.py` is the exact evaluator retained for audit. It loads the frozen portable preprocessor and four detector artifacts, applies frozen thresholds, and writes confusion matrices plus AUROC/PR-AUC.

The historical official test has already been opened and the empirical phase is closed. A third-party reproduction may execute:

```bash
python run_official_test_once.py
```

only after reproducing/fixing all development artifacts. Do not tune any model or threshold from its output. Reference results and output hashes are under `results/official_test/`.

## 11. Integrity checks

```bash
python -m compileall -q runtime scripts online device_calibration
python scripts/validate_public_release.py
shasum -a 256 -c PUBLIC_RELEASE_MANIFEST_SHA256.txt
```

The checksum manifest covers the public technical release, excluding itself and Git metadata.

The original deployment label-leakage gate is retained as `online/pi4b8/test_label_leakage.py`; it requires the non-redistributed `.formal_inputs/pi4b8_scheduler_train_v1/` package and should be run after that package is reconstructed locally.
