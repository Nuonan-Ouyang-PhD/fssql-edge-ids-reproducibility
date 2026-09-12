# Protocol V2 Power-Corrected Preflight Decision

Status: preflight pass; formal V2 fit may begin under the unchanged frozen
physical and governor configuration.

Run `pi3bplus_v2_power_corrected_preflight_20260909T123851Z` completed all
800/800 targeted LUCID 1024-record B2 windows under the `performance` governor.
It ran from 2026-09-09T12:39:00Z to 12:48:36Z. All 572 independent 1 Hz power
samples and the initial/final checks were `0x0`; every per-window frequency
sample was 1400 MHz; the maximum independent temperature was 40.780 C, below
the preregistered 55 C preflight limit.

All three raw hashes reproduce the run manifest:

- `raw_windows.csv`: `b038c675a6eb3ca798a06e258921f2216728123d0d21e0deb29bb524acc1fefd`
- `blocks.csv`: `52d6ea3e97e4afa7229019989ced65a4f1b6f9da0a3c432fee4d859276aa2c64`
- `power_telemetry_1hz.csv`: `6897aeaabb5bc17c9e2fb45cfed636f26188ef99316a45122f07b1934417f48b`

The preflight is code-path, thermal, frequency, and power-path evidence only. It
is excluded from V2 predictor fitting, margins, held-out coverage, feasibility
diagnostics, and policy results. Cooling, power path, and governor must remain
unchanged through the V2 sequence.
