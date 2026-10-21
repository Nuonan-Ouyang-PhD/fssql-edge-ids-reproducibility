# Pi4B8 Corrected Runtime Decision Draft

Status: **DRAFT / NO EXECUTION AUTHORIZATION**

This document records confirmed lineage facts and decisions that remain open
after the Pi4B8 code-to-protocol audit. It is not a preregistration, does not
freeze a scientific choice, and does not authorize Q retraining, commissioning,
or corrected formal runs. The historical V4 package and evidence remain
unchanged.

## Confirmed legacy semantics

- `configs/pi4b8_q_training_prereg_v1.json` labels the training workload as
  `n16/B0`, but `scripts/train_q_tables.py` actually consumes one scheduler
  record and one RiskProxy probability per training window.
- The trainer evaluates latency and energy from the Pi4B8 `n1/B0` lookup and
  calls the thermal predictor with `log2_n_records=0` and
  `background_duty=0`.
- Training initializes arrival and service to one record per window, so backlog
  remains zero. Direct inspection of all five FSSQL-R tables found that every
  visited state has backlog bin 0 and temperature bin 1 (35--45 C). Only
  35--42 of 1,280 states were visited per seed.
- The frozen training reward is label-free:
  `risk*validation_TPR - (1-risk)*validation_FPR`, minus the frozen synthetic
  energy, backlog, and switching terms. Energy starts at 1.0 and decreases by
  the action-specific `n1/B0` latency-derived cost.
- The V4 online inputs contain 600 individual feature rows, 600 individual risk
  rows, and 600 labels per cell. They do not contain 600 batches of 16 records.
- The V4 worker used the arithmetic mean of the eight unscaled risk-feature
  values, fixed energy to 1.0 and backlog to zero, and inferred one record per
  window. These are historical implementation facts, not corrected semantics.
- The five Q-table hashes, action order, 1,280-state encoding, bin boundaries,
  identity seed mapping `5101->5101` through `5105->5105`, and evaluation rules
  `epsilon=0`, no Q update, and no replay update remain reproducible.

## Existing Q tables: permitted interpretations requiring a decision

Two distinct paths are possible; neither is selected by this draft.

1. **Legacy transfer baseline without retraining.** Retain the existing Q
   tables and describe them explicitly as policies trained in the single-record
   `n1/B0`, zero-backlog synthetic MDP, then evaluated out of distribution.
   This path must preserve the trained state/reward semantics and report
   visited-state coverage. It cannot describe the tables as true-n16-trained or
   backlog-aware.
2. **Corrected true-n16 FSSQL-R.** Freeze a new development-only Q protocol
   covering 16-record batching, risk aggregation, Pi4B8 n16 thermal/latency
   inputs, backlog dynamics, energy state, and reward before training new Q
   tables. This path is required for any claim that learning was optimized for
   true n16 or nonzero backlog.

Existing Q tables must not be queried with nonzero-backlog states and presented
as learned behavior: those Q rows were never visited and remain at their zero
initialization.

## Threshold quantities fixed by existing rules

The following calculations use only the frozen Pi4B8 calibration artifacts and
the rule in `configs/major_revision_rebuild_protocol.json`:

- `protocols/POLICY_EXPERIMENT_PREREGISTRATION.md` and the frozen policy
  configuration define Threshold as an **unshielded** conventional baseline.
  A corrected implementation must still log the guard outputs for comparison,
  but must not silently mask Threshold through the safe set.

- `T_hi = 82 - epsilon_T = 80.14945503488127 C`
- `T_lo = T_hi - 2 = 78.14945503488127 C`
- Conditional on retaining TinyDL as the heavy action, its B0 conservative
  latency upper bound is 34.944391 ms at n512 and 56.327255 ms at n1024.
  Therefore `q_hi=512` and the previous calibrated workload bin gives
  `q_lo=256`.

The temperature calculation follows the already documented interpretation of
`epsilon_T`; it does not additionally subtract the fixed 0.6 C guard buffer.
The n512/n1024 cutoff is unchanged under B1 and B2.

## Threshold quantities not fixed for Pi4B8

- No Pi4B8 file freezes `p_hi` or a validation-only selection rule for it. The
  value `0.70` exists only as the earlier Pi3B method-level default. Adopting it
  for corrected Pi4B8 work would require an explicit additive decision; it
  cannot be described as uniquely derived from Pi4B8 development evidence.
- The Pi4B8 online configuration does not explicitly freeze the Threshold
  light/heavy action mapping. TinyDL is the unique highest-validation-F1 action
  and is consistent with the earlier heavy role. Reusing FISVDD as the
  method-level light action also requires an explicit statement. A new
  device-specific latency-median derivation is not equivalent: on Pi4B8 the
  TinyDL median calibration latency is slightly lower than FISVDD's.

## Decisions required before corrected Pi4B8 execution

1. Select and document either the legacy-transfer path or new true-n16 Q
   training path above.
2. Freeze the deterministic 16-record scheduler-development batching and all
   input hashes; do not reuse the 600-row V4 inputs as if they contained 9,600
   records.
3. Freeze the risk aggregation rule and its ordering relative to action
   selection. Labels must remain inaccessible until after prediction.
4. Freeze backlog provenance and dynamics. If backlog may exceed zero, new Q
   training is required.
5. Freeze whether n1/B0 synthetic energy/reward semantics remain in force or
   are replaced. Any replacement requires new Q training; physical POWER-Z
   energy must not be silently substituted into the Q state or reward.
6. Freeze Threshold `p_hi`, light/heavy roles, and their development-only
   rationale before execution.
7. Freeze exact B1/B2 load generation, affinity, duty cycle, lifetime, and
   cleanup, then validate it independently.
8. Build a new hash-bound runtime/input package and pass unit, local, remote,
   label-isolation, workload, guard, and background-load commissioning gates.
9. Obtain a separate execution authorization only after the selected repair
   protocol and all hashes are frozen.

The official test has already been accessed once. It must not be reopened or
used to choose any repair parameter, threshold, Q protocol, or workload.
