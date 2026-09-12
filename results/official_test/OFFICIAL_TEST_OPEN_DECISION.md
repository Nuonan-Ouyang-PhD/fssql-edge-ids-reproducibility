# Official UNSW-NB15 test open decision

This is the one-shot final independent detector-generalization evaluation. All development experiments, models, preprocessing and thresholds are frozen. No retraining, threshold tuning, feature changes, Q retraining, guard recalibration, workload/policy changes, or further development experiments may use these outcomes. The Pi4B8 online evidence remains a separate scheduler-development deployment/energy result.

## Frozen hashes

- `data/raw/UNSW_NB15_testing-set.csv`: `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559`
- `preprocessing/preprocessor_portable.npz`: `77d7ed29880adf3e3e1bcbbb59d5d705653b7c07fdcd88690041a45534b897be`
- `models/artifacts/fisvdd_revision.npz`: `0dd092d97498fdf45713e4d14ac6cbdef1b1ca01da04133e7f4b891880c9dc0a`
- `models/artifacts/lucid_revision.npz`: `9d81b5cbd284bd25528bc6ecd8a88f80933f95c4ef5cbc06b9c15ffaa9e78eb9`
- `models/artifacts/tinydl_revision.npz`: `b97413e60db93333ee9b9bb88fe4257a3cf8d5d6a556a8f2f7372c36ddb2e047`
- `models/artifacts/oi_svdd_as_elm_revision.npz`: `8d8530e2ed82a0644b0e8fef577e5fdff2379d12f831aa32df4b8eeb5a253896`
- `models/MODEL_HYPERPARAMETERS.json`: `3e61a35b7273dc7aa98fed8eb4f8425f8f7c126983eac3f337240b6b3f8d133c`
- `models/VALIDATION_METRICS.csv`: `7b2db09489ebc2c2e02c7604b0a7749a3a218d8d0b6abf547123f963affa1755`
- source commit: `22a90196a9556ae8c6d138cf364c36ba656dfa02`
