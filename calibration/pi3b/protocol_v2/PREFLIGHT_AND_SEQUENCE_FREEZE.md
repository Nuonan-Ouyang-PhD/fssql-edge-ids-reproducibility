# Pi 3B+ Protocol V2 Preflight and Sequence Freeze

Protocol V2 is a newly authorized protocol under a newly attested physical
cooling configuration. It is not Round 4, a retry, or a continuation of any
pre-V2 calibration data.

Before formal collection, a targeted preflight runs 800 LUCID windows at
1024 records/window with B2 full-duty background load under the `performance`
governor. It must complete 800/800 windows, retain at least 1350 MHz in every
per-window frequency sample, remain below 55 C, and show no current low-nibble
throttle/undervoltage bit. Preflight rows are excluded from all predictors,
margins, coverage, feasibility diagnostics, and policy outcomes.

If preflight passes, V2 collects entirely new fit, margin, and coverage splits
with seeds 7501, 7502, and 7503. Fit and margin artifacts must be hashed,
validated, frozen, and committed before coverage starts. Coverage is evaluated
once without tuning. The official test remains sealed.

The heatsink, continuously running fan, case body, open/removed top cover, and
wall-outlet power path are frozen throughout V2. Software can validate thermal,
frequency, and fault telemetry but cannot electronically prove fan rotation or
lid position; those remain direct user attestation.

The first preflight failed on current undervoltage/throttling and remains
immutable. After the user confirmed replacement or correction of the power
adapter/cable and rebooted to a clean `0x0`, the formal configuration was
re-frozen before any formal V2 row existed. The second preflight uses seed 7699
and the power-corrected hardware manifest. Formal seeds remain 7501/7502/7503
because they have not been used by any formal V2 run.
