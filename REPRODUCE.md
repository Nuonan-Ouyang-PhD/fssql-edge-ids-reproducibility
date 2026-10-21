# Reproduction guide

This guide separates software verification, development reconstruction, Raspberry Pi calibration, online execution, physical-energy acquisition, and one-shot test evaluation. Run each stage in a fresh checkout or isolated output directory because frozen-output scripts deliberately refuse to overwrite existing artifacts.

> **Audit notice (21 October 2026).** The historical Pi4B V4 online package
> does not implement the declared true-`n=16` policies and load conditions.
> Its results are provenance only. The corrected worker is intentionally
> fail-closed until a new repair protocol, true-`n=16` input package and
> compatible Q lineage are frozen. Do not use the historical command or data
> as a valid Pi4B reproduction.

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

Verify both files against `data/DATASET_SHA256.json`. Do not substitute
similarly named preprocessed mirrors. In a fresh reconstruction, keep the test
CSV unopened throughout development. In this repository's completed lineage,
the one authorized test access has already occurred; do not run it again.

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

The legacy Q-learning protocol is frozen in
`configs/q_learning_protocol.json`; Pi4B-specific lineage and its erratum are
under `q_training/pi4b8_q_v1/`.

```bash
python scripts/train_q_tables.py \
  --config configs/q_learning_protocol.json \
  --output q_training_reproduced
```

The checked-in seed-specific tables have `q_values` shape `1280 × 4`. They
were trained with one record per window, `n1/B0` predictor inputs and zero
backlog despite the old Pi4B config label. They may be inspected as legacy
single-record lineage, but the corrected worker rejects them for a true-`n=16`
FSSQL-R run. A new development-only Q protocol and tables are required if the
corrected experiment retains true-`n=16` or nonzero-backlog semantics.

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

## 7. Pi4B corrected online runtime (not yet authorized for execution)

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

The corrected interface additionally requires a hash-bound asset root,
runtime/Q configurations and manifest, the model-threshold source, 9,600 raw
and transformed development records, a separately frozen backlog source, and
the scoring-only label sidecar. It refuses the 600-row V4 inputs, missing B1/B2
load process definitions, and legacy single-record Q tables. Because the
scientific decisions listed in
`protocols/PI4B8_CORRECTED_RUNTIME_DECISION_DRAFT.md` are not yet frozen, this
guide deliberately provides no executable formal command.

The label file is opened only after every action and detector prediction has
been produced. B1/B2 require the exact pinned-CPU load lifecycle and
contemporaneous process evidence; a condition label alone is rejected. The
historical Mac orchestrator, V4 package builder and replacement runner remain
under `online/pi4b8/` for audit and are not compatible with the corrected
worker.

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

The checked-in Pi4B run-level data and paired results are historical V4
arithmetic summaries only. They are not valid corrected policy or energy
comparisons.

The analysis implementation is `analyze_pi4b8_formal.py`. It expects the private raw evidence layout used by the original campaign. To recompute from a locally restored raw archive, place that archive under `evidence/pi4b8_formal_matrix/` with the accepted-cell manifest and run:

```bash
python analyze_pi4b8_formal.py
```

The statistical unit is the matched seed/run, never a pooled window. With five Pi4B pairs per condition, the minimum nonzero two-sided exact Wilcoxon p-value is 0.0625; paired t tests, exact Wilcoxon tests, bootstrap 95% confidence intervals, and paired effect sizes are all retained.

## 10. One-shot official-test evaluator

`run_official_test_once.py` is retained for audit. It loads thresholds from
`models/MODEL_HYPERPARAMETERS.json`, writes only below
`results/official_test/`, and refuses to run while the immutable one-shot lock
exists.

The official test has already been opened once and the empirical phase is
closed. Do not execute it again in this artifact or use its saved results to
choose any repair parameter. Reference aggregate results and output hashes are
under `results/official_test/`.

## 11. Integrity checks

```bash
python -m compileall -q runtime scripts online device_calibration
python scripts/validate_public_release.py
shasum -a 256 -c PUBLIC_RELEASE_MANIFEST_SHA256.txt
```

The checksum manifest covers the public technical release, excluding itself and Git metadata.

The original deployment label-leakage gate is retained as `online/pi4b8/test_label_leakage.py`; it requires the non-redistributed `.formal_inputs/pi4b8_scheduler_train_v1/` package and should be run after that package is reconstructed locally.
