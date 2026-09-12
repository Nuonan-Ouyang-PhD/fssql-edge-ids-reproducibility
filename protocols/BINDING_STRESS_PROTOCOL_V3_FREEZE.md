# Binding Stress Protocol V3 Freeze

Status: **FROZEN BEFORE ANY V3 BINDING-LOAD SWEEP OR STRESS POLICY RUN**.

Protocol V3 is a post-completion extension of Formal Policy Matrix V1. It is
not a retry, replacement, relabelling, or tuning pass over the accepted
160-run/96,000-window matrix at commit
`6923d20e578497d88a399da243e30dc2db892348`.

## Sole objective

Identify the first externally applied workload/background point at which the
unchanged Protocol V2 guard is binding but non-degenerate. Workload selection
uses only scheduler-train/development inputs, guard admissibility, safe-set
cardinality, fallback requirement, and hardware-health telemetry. It must not
use any policy reward, F1, confusion count, ranking, official-test label, or
Formal Matrix V1 policy outcome.

The official test cannot be described as sealed: it was irrevocably opened for
Formal Matrix V1 at 2026-09-09T14:21:27Z. V3 nevertheless reads no official-test
artifact during workload selection. This explicit boundary replaces the
impossible instruction to keep an already-opened test sealed; it does not
authorize any post-opening tuning.

## Frozen candidate grid and order

The grid exactly reuses the values natively supported by the accepted V2
calibration generator:

- Workload outer order: `n=[1,16,64,256,512,1024]`
- Background inner order: `B0`, `B1`, `B2`
- B0: no experiment-created background load
- B1: one logical CPU, pinned to CPU 3, 50% duty in a 100 ms period
- B2: one logical CPU, pinned to CPU 3, 100% duty

Candidates are evaluated in that deterministic 18-point order. Evaluation
stops as soon as one candidate satisfies every frozen qualification rule. If
none qualifies, report that the V2 guard remained non-binding across the tested
development envelope and stop; do not alter the guard or scan a new grid.

## Frozen diagnostic schedule

- Seeds: `7801`, `7802`, `7803`
- Windows: 600 per seed and 1,800 per candidate
- Decision window: 100 ms
- Scheduler source: frozen 6,000-row `scheduler_train` development input
- Row order: `numpy.random.default_rng(seed).permutation(6000)`; consume
  `n_records` sequentially and cycle only when a candidate requires more than
  6,000 records
- Workload probe action: frozen action order modulo four, reset for each seed
- The probe action is not a scheduling policy and does not depend on the safe
  set, risk, label, reward, prediction quality, or any policy outcome
- Model outputs are checked only for finite execution and are not logged or
  used in qualification
- Realised diagnostic execution latency is logged for operational provenance
  but is expressly excluded from workload selection

## Frozen qualification rule

Across all 1,800 valid windows for a candidate, accept only if:

1. At least 10% of windows have `|A_safe| < 4`.
2. At least 5% of windows have `|A_safe| >= 2`.
3. At least two distinct actions are admissible somewhere in the diagnostic.
4. Empty-safe-set/fallback-required rate is at most 1%.
5. No current throttle, soft-temperature-limit, or undervoltage bit occurs.
6. Temperature remains below 58 C and sampled CPU frequency remains at least
   1350 MHz under the frozen `performance` governor.

The guard uses the unchanged 40 ms latency cap, 82 C thermal cap, V2 predictor,
per-action residual margins, shared thermal margin, and fixed 10 ms/0.6 C
buffers. No candidate may change any of these values.

## Physical-condition freeze and invalid attempts

The Pi 3B+ retains the V2 physical configuration: heatsink installed, fan
running continuously, case top open, and corrected adapter/cable on the stable
wall-outlet path. Each candidate begins with 30 one-second idle samples and may
start only at or below idle median plus 2 C. Any current fault bit, temperature
at or above 58 C, frequency below 1350 MHz, I/O/process failure, hash mismatch,
or physical-condition change invalidates and stops the entire sweep. Partial
evidence is retained; there is no stitching or silent retry.

## Freeze artifacts

- Machine config: `configs/pi3b_binding_stress_protocol_v3.json`
- Config SHA-256:
  `c17db21a8a224366c9845c6aec889286df1899e93045c5af8c2aae3529baa80d`
- Collector: `online/pi3b/collect_binding_stress_sweep.py`
- Collector SHA-256:
  `bb0d3525e3b37c5c93a288404ae56e7b03476e3654734c72c36a4e11b7f0e7b9`
- V2 latency predictor SHA-256:
  `9ec98d3cbce16657622905a12dc238aa10beb526a5304316a86c2841b2545edb`
- V2 thermal predictor SHA-256:
  `71ca6317ead2e59af43ad034634f6e5d115b437b81578aee55b5fcc1b64cc072`
- V2 guard margins SHA-256:
  `1a55ec7f33333b67456d0dddfafd08d46ea8b255f40ea5f56834f9029328e051`

If a candidate qualifies, its complete raw diagnostic and independent
validation must be committed before a separate Stress Extension Matrix is
preregistered. Formal Matrix V1 remains reported separately as the stable,
non-binding control regime.
