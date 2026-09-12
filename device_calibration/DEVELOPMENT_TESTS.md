# Calibration Collector Development Tests

These device-local runs validate the collector code path only. They use
development overrides and are excluded from formal calibration evidence.

- `development_code_path_20260909T1035Z`: invalid. Collection reached final
  manifest construction, where the Python source used JSON-style `false` and
  raised `NameError`. Partial raw files are retained; no rows may be stitched
  into later runs.
- `development_code_path_20260909T1038Z`: pass for 3 cells x 2 observations,
  but the realized seeded prefix covered B1 only.
- `development_code_path_20260909T1040Z`: pass for 7 cells x 1 observation;
  the realized prefix covered B0, B1, and B2, including `n=1024` under B2.
- `development_code_path_20260909T1043Z`: final post-hardening pass for 7 cells
  x 1 observation after adding start-temperature enforcement and monitored-log
  closure/hashing; 7/7 windows and all three generated-log hashes passed.
- `development_code_path_20260909T1047Z`: pass for the final manifest additions;
  it verifies that the collector records its source hash, the frozen Git commit
  field, and the model-manifest hash after validating all four action artifacts.
- `development_frequency_telemetry_20260909T1118Z`: pass after adding four
  per-window CPU-frequency fields for Round 2; all four fields were populated
  and the one-window development run ended at `throttled=0x0`.
- `development_round2_invariant_20260909T1120Z`: pass under the Round 2
  performance-governor configuration. The three-cell prefix covered B0/B1/B2;
  every per-window start/end frequency was 1400 MHz and the run ended at
  `throttled=0x0`.

The corrected collector and frozen configuration must be committed before any
formal split is started.
