import csv
import datetime
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from online.pi4b8 import powerz_hid_capture as capture
from online.pi4b8.powerz_evidence_validator import (
    PowerEvidenceError,
    SCHEMA,
    validate_powerz_evidence,
)


class FakeClock:
    def __init__(self, step=0.1):
        self.value = -step
        self.step = step

    def __call__(self):
        self.value += self.step
        return self.value


class FakeDevice:
    def __init__(self, interrupt_after=None):
        self.interrupt_after = interrupt_after
        self.reads = 0
        self.opened = None
        self.closed = False

    def open(self, vendor_id, product_id, serial):
        self.opened = (vendor_id, product_id, serial)

    def write(self, _data):
        return 65

    def read(self, _size, _timeout):
        if self.interrupt_after is not None and self.reads >= self.interrupt_after:
            raise KeyboardInterrupt
        self.reads += 1
        raw = bytearray(64)
        struct.pack_into("<ii", raw, 16, 5_000_000, -400_000)
        return raw

    def close(self):
        self.closed = True


def handshake(run_id="run-1"):
    return {
        "run_id": run_id,
        "meter_capture_start_mac_monotonic": -1.0,
        "start_request_mac_monotonic": 4.0,
        "start_ack_receive_mac_monotonic": 5.0,
        "end_ack_receive_mac_monotonic": 15.0,
        "meter_capture_end_mac_monotonic": 21.0,
        "pi_start_ack": {"event": "RUN_START_ACK", "run_id": run_id},
        "pi_end_ack": {"event": "RUN_END_ACK", "run_id": run_id},
    }


def write_meter(path, *, mutate=None):
    start = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    packet = bytearray(64)
    struct.pack_into("<ii", packet, 16, 5_000_000, -400_000)
    rows = []
    for second in range(21):
        rows.append(
            {
                "timestamp_utc": (start + datetime.timedelta(seconds=second)).isoformat(),
                "sample_monotonic": str(second),
                "voltage_V": "5.0",
                "current_A": "0.4",
                "power_W": "2.0",
                "raw_hex": bytes(packet).hex(),
            }
        )
    if mutate:
        mutate(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCHEMA)
        writer.writeheader()
        writer.writerows(rows)


class CaptureTests(unittest.TestCase):
    def test_default_monotonic_clock_is_shared_across_processes(self):
        child = float(
            subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    "import time; print(time.clock_gettime(time.CLOCK_MONOTONIC))",
                ],
                text=True,
            ).strip()
        )
        self.assertLess(abs(capture.shared_monotonic() - child), 1.0)
        self.assertIs(
            capture.capture.__kwdefaults__["monotonic"], capture.shared_monotonic
        )

    def test_ready_identity_and_streamed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "power.csv"
            device = FakeDevice()
            emitted = []
            rows = capture.capture(
                path,
                seconds=0.35,
                sps=10,
                device_factory=lambda: device,
                monotonic=FakeClock(),
                utc_now=lambda: datetime.datetime(
                    2026, 1, 1, tzinfo=datetime.timezone.utc
                ),
                sleep=lambda _seconds: None,
                emit=emitted.append,
            )
            self.assertEqual(rows, 1)
            ready = json.loads(emitted[0])
            self.assertEqual(ready["event"], "POWERZ_READY")
            self.assertEqual(ready["serial"], "075356")
            self.assertEqual(device.opened, (0x5FC9, 0x0063, "075356"))
            self.assertTrue(device.closed)
            with path.open(newline="", encoding="utf-8") as stream:
                captured = list(csv.DictReader(stream))
            self.assertEqual(len(captured), 1)
            self.assertEqual(len(captured[0]["raw_hex"]), 128)

    def test_keyboard_interrupt_preserves_completed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.csv"
            device = FakeDevice(interrupt_after=1)
            rows = capture.capture(
                path,
                seconds=10,
                sps=10,
                device_factory=lambda: device,
                monotonic=FakeClock(),
                utc_now=lambda: datetime.datetime(
                    2026, 1, 1, tzinfo=datetime.timezone.utc
                ),
                sleep=lambda _seconds: None,
                emit=lambda _line: None,
            )
            self.assertEqual(rows, 1)
            with path.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)


class EvidenceValidatorTests(unittest.TestCase):
    def test_independent_ack_boundary_integration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "power.csv"
            write_meter(path)
            result = validate_powerz_evidence(path, handshake(), run_id="run-1")
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["samples"], 21)
            self.assertEqual(result["energy_J"], 20.0)
            self.assertEqual(result["mean_power_W"], 2.0)
            self.assertFalse(result["endpoint_extension"])

    def test_rejects_invalid_raw_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "power.csv"
            write_meter(path, mutate=lambda rows: rows[5].update(raw_hex="00"))
            with self.assertRaisesRegex(PowerEvidenceError, "raw_hex length"):
                validate_powerz_evidence(path, handshake(), run_id="run-1")

    def test_rejects_inconsistent_power(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "power.csv"
            write_meter(path, mutate=lambda rows: rows[5].update(power_W="3.0"))
            with self.assertRaisesRegex(PowerEvidenceError, "inconsistent"):
                validate_powerz_evidence(path, handshake(), run_id="run-1")

    def test_rejects_columns_inconsistent_with_raw_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "power.csv"
            write_meter(
                path,
                mutate=lambda rows: rows[5].update(
                    voltage_V="4.9", power_W="1.96"
                ),
            )
            with self.assertRaisesRegex(PowerEvidenceError, "raw HID packet"):
                validate_powerz_evidence(path, handshake(), run_id="run-1")

    def test_rejects_nonincreasing_utc(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "power.csv"
            write_meter(
                path,
                mutate=lambda rows: rows[5].update(
                    timestamp_utc=rows[4]["timestamp_utc"]
                ),
            )
            with self.assertRaisesRegex(PowerEvidenceError, "timestamp_utc"):
                validate_powerz_evidence(path, handshake(), run_id="run-1")

    def test_rejects_ack_boundary_outside_samples_without_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "power.csv"
            write_meter(path)
            bad = handshake()
            bad["start_ack_receive_mac_monotonic"] = -0.5
            bad["start_request_mac_monotonic"] = -0.5
            with self.assertRaisesRegex(PowerEvidenceError, "outside"):
                validate_powerz_evidence(path, bad, run_id="run-1")


if __name__ == "__main__":
    unittest.main()
