#!/usr/bin/env python3
"""Direct HID KM003C capture used by the accepted TNSM measurement path."""
from __future__ import annotations
import argparse, csv, datetime, struct, time
from pathlib import Path
import hid

def request(sequence: int) -> bytes:
    return bytes(1) + struct.pack('<I', 12 | ((sequence % 256) << 8) | (1 << 17)) + bytes(60)

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); p.add_argument('--seconds', type=float, default=20.0); p.add_argument('--sps', type=float, default=10.0); a = p.parse_args()
    d = hid.device(); d.open(0x5FC9, 0x0063, '075356'); rows = []; period = 1.0 / a.sps; end = time.monotonic() + a.seconds; seq = 0
    try:
        while time.monotonic() < end:
            t0 = time.monotonic(); utc = datetime.datetime.now(datetime.timezone.utc).isoformat(); d.write(request(seq)); raw = bytes(d.read(64, 1000));
            if len(raw) != 64: raise RuntimeError('KM003C HID reply length')
            uv, ua = struct.unpack_from('<ii', raw, 16); volts = uv / 1e6; amps = -ua / 1e6; rows.append([utc, t0, volts, amps, volts * amps, raw.hex()]); seq += 1
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    finally: d.close()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['timestamp_utc','sample_monotonic','voltage_V','current_A','power_W','raw_hex']); w.writerows(rows)
    print(f'captured_rows={len(rows)} output={a.output}')
if __name__ == '__main__': main()
