# Raspberry Pi 4B 8GB second-device phase

Status: **preflight reachable; calibration not yet executed**.

The device-specific phase is independent of the accepted Pi 3B+ V1/V3 evidence. The Pi 4B 8GB (`pi4b8g`, serial `100000005368e39d`) is reachable over Wi-Fi, reports 8,007,464 KiB memory, `throttled=0x0`, and 28.2 C at the 2026-09-10 snapshot. Its current governor is `ondemand`; no experiment parameters or guard values have been copied from Pi 3B+.

Required next execution sequence:

1. Freeze a Pi 4B 8GB hardware manifest and cooling/power attestation.
2. Run independent latency and thermal calibration (fit → margin → coverage) on development data only, producing a new Pi 4B 8GB guard.
3. Run online closed-loop development confirmation for FSSQL-R, Safe-Greedy, Threshold, and Static-Light.
4. Run preregistered natural/near-cap evaluation and background robustness only after the Pi 4B guard and workload decisions are frozen.
5. Record whole-device POWER-Z KM002C/KM003C input power at the protocol-defined rate and window, with physical W/J claims limited to those recordings.

The official test remains sealed. No Pi 4B formal result is claimed by this status file. POWER-Z live CSV export and the device-specific calibration runner are still required before formal execution.
