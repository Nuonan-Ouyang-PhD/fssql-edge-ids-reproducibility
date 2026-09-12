# PI4B8 Q-lineage audit

Status: **PASS — numeric Q artifacts retained; formal evaluation not started**

The accepted training process was `scripts/train_q_tables.py` (SHA-256
`7e058d7efd034ecac6cbcb1e54b7443c0b45bd56febbf6a864e32d6f3b53bd8e`) with
`configs/pi4b8_q_training_prereg_v1.json` (SHA-256
`87a05f8b3e81f0461cddb6b31ff7c24315bf8e86f45c77b2a5b506028f729f50`). The
script first opens and hash-checks every entry in `config.inputs`, then reads
only the files listed below for training calculations.

## Dependency audit

| Path | SHA-256 | Opened/read | Role | Device lineage |
|---|---|---|---|---|
| `configs/pi4b8_q_training_prereg_v1.json` | `87a05f8b3e81f0461cddb6b31ff7c24315bf8e86f45c77b2a5b506028f729f50` | yes | frozen training config | Pi4B8 prereg |
| `scripts/train_q_tables.py` | `7e058d7efd034ecac6cbcb1e54b7443c0b45bd56febbf6a864e32d6f3b53bd8e` | yes | trainer implementation | device-generic, Pi4 invocation |
| `device_calibration/inputs/scheduler_train_6000_risk_features.npy` | `fee72cdd9445e8ebd4ecb12e888848b52269bd0fe81ef259ecf25525bf711dcd` | yes | risk sequence | development, device-independent |
| `risk_proxy/risk_proxy.npz` | `eff41b7f4bd7b6303ba32d39b19be6e896d9b8c2de8d3a32f850000ea3c712c2` | yes | risk scoring | common frozen model |
| `calibration/pi4b8/frozen_guard_fit_margin_v1_run2/latency_predictor.csv` | `47f9126b40d49e2b085bab34ea595e3a1c339cb97f5d61b644863f8d78cf273c` | yes | n1/B0 latency estimates | Pi4B8 |
| `calibration/pi4b8/frozen_guard_fit_margin_v1_run2/thermal_predictor.json` | `ba65dfec91c76ee86ad18ebc6f6b99e9c4881c7b46e20cd9cebf086b37d2a6e5` | yes | thermal transition | Pi4B8 |
| `calibration/pi4b8/frozen_guard_fit_margin_v1_run2/guard_margins.json` | `bedc7b4ce2be5efce82dea220a9d311052aeea9a80d1022680c38c65c16c523a` | yes | guard margins | Pi4B8 |
| `calibration/pi4b8/formal_fit_20260910_v3/raw_windows.csv` | `2bdd3499aa9fd6e9702e01c1101f73f493614d37f1d49a3a5fc11ddac1350079` | yes | initial temperature estimate | Pi4B8 |
| `models/VALIDATION_METRICS.csv` | `7b2db09489ebc2c2e02c7604b0a7749a3a218d8d0b6abf547123f963affa1755` | yes | fixed action-quality terms | common frozen validation |
| `data/SPLIT_MANIFEST.csv` | `1c70959a6d12...` | yes | validation prevalence only | development split |
| `models/MODEL_SHA256.json` | `1c9c21560fc...` | yes (hash check only) | model-manifest integrity | common frozen models |
| `device_calibration/inputs/scheduler_train_6000_features.npy` | `4cbb6849d269129a01038693aa6a981017b55d8949da81074aece2e1100e823d` | yes (hash check only) | config-bound input integrity | development |
| `configs/near_cap_workload.json` | `57934ae14513abc4878313651fb8293610be088c1a30da1cfb73a0532c28fed1` | yes (hash check only) | inherited unused metadata | Pi3B+ (unused) |
| `workloads/pi3b/near_cap/freeze_v1/NEAR_CAP_FREEZE_SUMMARY.json` | `a939e0aa20b49c5e487e7d350e3505bba78deb2e3d0a9d92254135fbbd7070b0` | yes (hash check only) | inherited unused metadata | Pi3B+ (unused) |

The old `evaluation.training_to_evaluation_seed_map` is never referenced by
the trainer. The two Pi3B+ near-cap files are opened only by the generic
input hash-check loop; neither contributes to risk values, transitions,
allowed-action calculation, rewards, or Q updates. The actual five FSSQL-R
tables are trained with and bound to seeds 5101–5105 respectively.

The numeric hyperparameters remain a transparent Major-Revision
reimplementation because the historical numeric hyperparameters were not
retained; they are not claimed to be the exact submitted setting. Evaluation
is epsilon=0 with Q/replay updates disabled by the frozen protocol.

Ground-truth labels, labeled CSVs, and official-test files are not opened by
the trainer. The official UNSW-NB15 test remains sealed.
