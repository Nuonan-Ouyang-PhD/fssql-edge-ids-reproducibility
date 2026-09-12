#!/usr/bin/env python3
"""Read one KM003C ADC frame to verify programmatic meter access."""

from __future__ import annotations

import argparse
import json
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import hid


VENDOR_ID = 0x5FC9
PRODUCT_ID = 0x0063
SERIAL = "075356"

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)

sequence = 249
request = bytes(1) + struct.pack(
    "<I", 12 | ((sequence % 256) << 8) | (1 << 17)
) + bytes(60)

device = hid.device()
try:
    device.open(VENDOR_ID, PRODUCT_ID, SERIAL)
    monotonic_start = time.monotonic_ns()
    device.write(request)
    raw = bytes(device.read(64, 1000))
    monotonic_end = time.monotonic_ns()
finally:
    device.close()

if len(raw) != 64:
    raise RuntimeError(f"expected 64-byte ADC reply, received {len(raw)}")

header, attribute = struct.unpack_from("<II", raw)
if header & 127 != 65 or (header >> 8) & 255 != sequence or attribute & 32767 != 1:
    raise RuntimeError("unexpected KM003C reply identity")

microvolts, microamps = struct.unpack_from("<ii", raw, 16)
voltage_v = microvolts / 1e6
signed_current_a = microamps / 1e6
load_current_a = -signed_current_a
if not 4.5 <= voltage_v <= 5.6 or not 0 < load_current_a <= 5:
    raise RuntimeError("meter values outside conservative accessibility-probe range")

record = {
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "vendor_id": f"0x{VENDOR_ID:04x}",
    "product_id": f"0x{PRODUCT_ID:04x}",
    "serial": SERIAL,
    "monotonic_start_ns": monotonic_start,
    "monotonic_end_ns": monotonic_end,
    "roundtrip_ms": (monotonic_end - monotonic_start) / 1e6,
    "voltage_v": voltage_v,
    "signed_current_a": signed_current_a,
    "load_current_a": load_current_a,
    "power_w": voltage_v * load_current_a,
    "raw_hex": raw.hex(),
    "status": "PASS",
    "scope": "single accessibility frame; not a formal experiment measurement",
}

with output.open("x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")

print(output)
