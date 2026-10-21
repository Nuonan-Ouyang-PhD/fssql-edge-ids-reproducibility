#!/usr/bin/env python3
"""Stream timestamped POWER-Z KM003C samples to an audit-preserving CSV."""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import signal
import struct
import time
from pathlib import Path
from typing import Callable, Protocol


VID = 0x5FC9
PID = 0x0063
SERIAL = "075356"
SCHEMA = (
    "timestamp_utc",
    "sample_monotonic",
    "voltage_V",
    "current_A",
    "power_W",
    "raw_hex",
)


def shared_monotonic() -> float:
    """Return the system-wide monotonic clock shared by Mac processes."""
    return time.clock_gettime(time.CLOCK_MONOTONIC)


class HidDevice(Protocol):
    def open(self, vendor_id: int, product_id: int, serial: str) -> None: ...

    def write(self, data: bytes) -> int: ...

    def read(self, size: int, timeout_ms: int) -> list[int] | bytes: ...

    def close(self) -> None: ...


def request(sequence: int) -> bytes:
    return (
        bytes(1)
        + struct.pack("<I", 12 | ((sequence % 256) << 8) | (1 << 17))
        + bytes(60)
    )


def _default_device_factory() -> HidDevice:
    import hid

    return hid.device()


def capture(
    output: Path,
    *,
    seconds: float = 20.0,
    sps: float = 10.0,
    device_factory: Callable[[], HidDevice] = _default_device_factory,
    monotonic: Callable[[], float] = shared_monotonic,
    utc_now: Callable[[], datetime.datetime] = lambda: datetime.datetime.now(
        datetime.timezone.utc
    ),
    sleep: Callable[[float], None] = time.sleep,
    stop_requested: Callable[[], bool] = lambda: False,
    emit: Callable[[str], None] = lambda line: print(line, flush=True),
) -> int:
    """Capture samples, flushing every row so interruption preserves raw evidence."""
    if seconds <= 0 or sps <= 0:
        raise ValueError("seconds and sps must be positive")

    output.parent.mkdir(parents=True, exist_ok=True)
    device = device_factory()
    device.open(VID, PID, SERIAL)
    period = 1.0 / sps
    end = monotonic() + seconds
    rows = 0

    try:
        with output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(SCHEMA)
            stream.flush()
            emit(
                json.dumps(
                    {
                        "event": "POWERZ_READY",
                        "serial": SERIAL,
                        "output": str(output),
                    },
                    sort_keys=True,
                )
            )

            try:
                while not stop_requested() and monotonic() < end:
                    sample_time = monotonic()
                    timestamp_utc = utc_now().astimezone(
                        datetime.timezone.utc
                    ).isoformat()
                    device.write(request(rows))
                    raw = bytes(device.read(64, 1000))
                    if len(raw) != 64:
                        raise RuntimeError("KM003C HID reply length")
                    microvolts, microamps = struct.unpack_from("<ii", raw, 16)
                    voltage = microvolts / 1e6
                    current = -microamps / 1e6
                    writer.writerow(
                        (
                            timestamp_utc,
                            sample_time,
                            voltage,
                            current,
                            voltage * current,
                            raw.hex(),
                        )
                    )
                    stream.flush()
                    rows += 1
                    sleep(max(0.0, period - (monotonic() - sample_time)))
            except KeyboardInterrupt:
                pass
            finally:
                stream.flush()
    finally:
        device.close()

    emit(
        json.dumps(
            {
                "event": "POWERZ_CAPTURE_END",
                "serial": SERIAL,
                "captured_rows": rows,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--sps", type=float, default=10.0)
    args = parser.parse_args()

    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    previous = signal.signal(signal.SIGTERM, request_stop)
    try:
        capture(
            args.output,
            seconds=args.seconds,
            sps=args.sps,
            stop_requested=lambda: stopping,
        )
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
