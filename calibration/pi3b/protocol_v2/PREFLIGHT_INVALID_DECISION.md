# Protocol V2 Preflight Invalid Decision

Status: preflight invalid; do not start V2 formal fit.

The targeted run `pi3bplus_v2_preflight_20260909T123046Z` was configured for
800 LUCID windows at 1024 records/window with B2 full-duty background load and
the `performance` governor. The independent monitor observed
`throttled=0xd0005` after 115 recorded windows. Low nibble `0x5` indicates a
current undervoltage condition together with current throttling, so the
collector invalidated the complete preflight and exited without retry.

The event occurred at only 37.552 C; the maximum independent monitor
temperature was 38.628 C. Therefore active cooling appears thermally effective
under the observed interval, but the V2 power path is not accepted under the
target load. This preflight cannot establish formal calibration evidence.

All retained raw hashes reproduce the run manifest:

- `raw_windows.csv`: `955708fcb4f549a152f9cc9c934f5ad9d3112a707cee6ac97615d66bf1b50b31`
- `blocks.csv`: `3fce99cf7b7355f9db216acfe6c6be0adc8a9901d423f95ca78a68e5156424a0`
- `power_telemetry_1hz.csv`: `757750b022a2e6448c45ea0bfe8aa7e7c28c1c35129fa4361d701722f5620604`

The governor was restored to `ondemand`. Formal V2 fit, margin, and coverage
were not started. Before a new preflight, the Pi power adapter/cable path must
be corrected and the device rebooted so the new power condition can be checked
from a clean throttle register. Any changed power or fan-power wiring must be
recorded in a new hardware snapshot; the invalid attempt must not be deleted,
continued, or stitched into later evidence.
