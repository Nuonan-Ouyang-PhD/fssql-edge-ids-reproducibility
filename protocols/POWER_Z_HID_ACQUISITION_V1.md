# POWER_Z_HID_ACQUISITION_V1

Status: **FROZEN BEFORE PI4B8 FORMAL ONLINE RUNS**

## Meter identity

- Device: POWER-Z KM003C (ChargerLAB)
- Serial: `075356`
- macOS USB node: `/dev/cu.usbmodem0753562`
- Acquisition path: direct HID, VID `0x5FC9`, PID `0x0063`
- Capture implementation: `online/pi4b8/powerz_hid_capture.py`
- SHA-256: `b4e7a5c00803171d867009780ddfe2bfb5bc441d5a80930e2789e184e9e30721`

## Measurement boundary

Power and energy are **DC input-side measurements taken inline on the
Raspberry Pi 4B 8GB power-supply path using the POWER-Z KM003C**. They are not
AC wall-plug or whole-site measurements.

## Raw CSV schema

Each unchanged raw CSV contains one header and rows with:

`timestamp_utc,sample_monotonic,voltage_V,current_A,power_W,raw_hex`

`timestamp_utc` is UTC ISO-8601; `sample_monotonic` is the Mac monotonic clock
at HID request start. `raw_hex` is retained for decoding audit.

## Capture and binding convention

For run `<run_id>`, retain the raw file as
`POWERZ_<run_id>_raw.csv` and bind it one-to-one in the run manifest using its
SHA-256. Capture starts at least 5 seconds before the Pi run, retains the exact
capture-start timestamp, continues through the run, and ends at least 5 seconds
after Pi completion. The commissioning file
`POWERZ_test_20260910.csv` is commissioning evidence only and is excluded from
all formal statistics.

## Integration rule

Primary run energy is the timestamp-based trapezoidal integral of `power_W`
over the exact Pi run start/end interval. Use recorded monotonic timestamps;
do not assume exactly 10 SPS, resample, smooth, or subtract idle power. If a
boundary falls between samples, linearly interpolate only that boundary value.
Retain all pre/post samples for audit. Report absolute J, Wh, mean/peak W,
voltage/current distributions, J/window, J/1000 processed records, and
J/correctly detected attack.

## Alignment and invalidation

Before formal execution, record Mac/Pi UTC handshake offset and uncertainty.
Each run records Pi acknowledged UTC and monotonic start/end. Invalidate the
energy record if the meter identity changes, HID reply is malformed, samples
are missing or non-monotonic, voltage/current/power is non-finite or outside
the accepted device envelope, pre/post coverage is under 5 seconds, run_id
binding/hash is absent, or clock alignment is insufficient for the declared
metric. Per-window physical energy is not claimed unless 100-ms alignment is
demonstrated; per-run energy is primary.
