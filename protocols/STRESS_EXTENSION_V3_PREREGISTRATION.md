# Stress Extension Matrix V3 Preregistration

Status: **PREREGISTERED BEFORE ANY FORMAL STRESS EXTENSION POLICY RUN**.

This is a separate binding-resource extension of the accepted Formal Policy
Matrix V1. It is not a retry, replacement, relabelling, or tuning pass over the
160-run/96,000-window V1 evidence. V1 remains frozen at commit
`6923d20e578497d88a399da243e30dc2db892348`.

## Frozen workload and evidence boundary

The independently validated, outcome-blind Protocol V3 sweep selected the
first qualifying candidate, `n=16/B0`, at evidence commit
`aeccfe7ea1ae49265720568518b242d249c5dd43`. Across its 1,800 development
diagnostic windows, exactly three actions were safe in every window: FISVDD,
TinyDL, and OI-SVDD+AS-ELM. LUCID was excluded by the unchanged latency guard.
The binding fraction was 1.0 and the fallback rate was 0.0.

The official test was already irrevocably opened for Formal Matrix V1 before
this extension was conceived. This preregistration reuses the immutable matched
trace and verifies its accepted manifest and five artifact hashes. Workload,
policies, guard, Q tables, order, and analysis rules are frozen here before any
Stress Extension V3 policy outcome is generated.

## Matrix

- Device: Raspberry Pi 3B+
- Regime label: `binding_resource`
- Policies: Static-Light, Round-Robin, Safe-Greedy, Threshold, Unshielded-Q,
  and FSSQL-R
- Matched evaluation seeds: 4101 through 4110
- Runs: 60, exactly one run per policy/seed pair
- Run length: 600 decision windows at 100 ms per window
- Workload: 16 records per window, B0 background
- Records evaluated per run: 9,600
- Total planned windows/record evaluations: 36,000/576,000

Within each seed, policy order is the deterministic permutation from
`numpy.random.default_rng(20260909 + evaluation_seed)`. Seed blocks execute in
ascending order. The generated 60-row matrix is immutable.

## Frozen 16-record trace rule

For decision window `w`, use matched-trace positions `(16*w+j) mod 600` for
`j=0..15` in that seed's existing frozen order. Consequently every one of the
600 matched trace entries is processed exactly 16 times per run and every
policy sees the same record sequence within a seed.

Before action selection, the window-level risk state is the arithmetic mean of
the 16 frozen per-record risk-proxy scores. The selected detector processes all
16 records in one batch. Labels remain in their separate frozen array and are
read only after action selection and inference; window TP, FP, TN, and FN are
counts over the 16 records. Each window logs its exact 16 array indices.

## Frozen controllers and guard

All six policy definitions, detector thresholds, model artifacts, V2 guard,
40 ms latency cap, 82 C thermal cap, residual margins, fixed buffers, and
conventional Threshold settings are unchanged from V1. Safe-Greedy and
FSSQL-R use the identical V2 safe set. Under `n=16/B0`, the guard uses the
frozen workload-specific V2 latency and thermal predictions.

Unshielded-Q and FSSQL-R use the previously frozen mapped Q tables, epsilon 0,
and no evaluation updates or replay updates. There is no stress-specific
retraining. State bins and the learned reward remain unchanged. In particular,
the energy-proxy state transition and reward cost retain the original frozen
`n=1/B0` Q-training definition; `n=16/B0` changes guard admissibility and the
measured execution workload, not the learned objective. This boundary must be
reported when interpreting out-of-training-regime Q behavior.

## Hardware and invalidation

The frozen physical configuration remains: heatsink installed, fan running
continuously, case top open, and corrected adapter/cable on the stable wall
outlet. Each run begins with 30 one-second idle samples and may start only at
or below idle median plus 2 C. The governor must be `performance`, sampled
frequency must remain at least 1,350 MHz, current throttle/soft-temperature/
undervoltage low-nibble bits must remain zero, and temperature must remain below
58 C.

Any hardware fault, frequency/temperature breach, hash mismatch, incomplete or
non-finite required field, process/I/O failure, or physical-condition change
invalidates the attempt and stops the entire campaign. Retain the partial
attempt; do not stitch or retry silently.

## Frozen artifacts

- Machine config: `configs/pi3b_stress_extension_v3.json`
- Config SHA-256:
  `94ef273682b9343bb52d6acba479f4045ea3011eeda0ef90b1facc7e29bbc86a`
- Matrix: `configs/pi3b_stress_extension_v3_matrix.csv`
- Matrix SHA-256:
  `9a841e4b0fd5bb9fed9ebe571cfa2c5fe49437a59cb5695c8c937a5ee902bf5f`
- Policy collector SHA-256:
  `ab6dafce99c6d7dd8dcd935d9812637532bbc2ca9645a194a2bbb3f7da13323b`
- Campaign runner SHA-256:
  `6f6852cfde7e0ad187a6bbf03ab7c3bca38aea16c32fd57fb5f31cde675ad29a`
- Freeze validation: `provenance/STRESS_EXTENSION_V3_FREEZE_VALIDATION.json`

Formal Matrix V1 and Stress Extension V3 must be analyzed and reported as
separate matrices. No Stress Extension V3 result may be used to change any
frozen input, controller, workload, seed, order, or interpretation boundary.
