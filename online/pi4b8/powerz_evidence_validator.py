#!/usr/bin/env python3
"""Independent schema, provenance, and ACK-boundary checks for POWER-Z CSVs."""
from __future__ import annotations

import bisect
import csv
import datetime
import hashlib
import math
import struct
from pathlib import Path
from typing import Any, Mapping


SCHEMA = (
    "timestamp_utc",
    "sample_monotonic",
    "voltage_V",
    "current_A",
    "power_W",
    "raw_hex",
)


class PowerEvidenceError(ValueError):
    """Raised when raw POWER-Z evidence is not independently valid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_seconds(value: str, row_number: int) -> float:
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PowerEvidenceError(
            f"row {row_number}: timestamp_utc is not ISO-8601"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() != datetime.timedelta(0):
        raise PowerEvidenceError(f"row {row_number}: timestamp_utc is not UTC")
    seconds = parsed.timestamp()
    if not math.isfinite(seconds):
        raise PowerEvidenceError(f"row {row_number}: timestamp_utc is non-finite")
    return seconds


def _interpolate(times: list[float], values: list[float], target: float) -> float:
    """Interpolate only inside the measured interval; never extend endpoints."""
    if target < times[0] or target > times[-1]:
        raise PowerEvidenceError("ACK boundary is outside measured POWER-Z samples")
    index = bisect.bisect_left(times, target)
    if index < len(times) and times[index] == target:
        return values[index]
    if index == 0 or index == len(times):
        raise PowerEvidenceError("ACK boundary would require endpoint extension")
    left, right = index - 1, index
    fraction = (target - times[left]) / (times[right] - times[left])
    return values[left] + fraction * (values[right] - values[left])


def _integrate(
    times: list[float], powers: list[float], start: float, end: float
) -> tuple[float, float]:
    start_power = _interpolate(times, powers, start)
    end_power = _interpolate(times, powers, end)
    interior = [index for index, value in enumerate(times) if start < value < end]
    integration_times = [start] + [times[index] for index in interior] + [end]
    integration_power = [start_power] + [powers[index] for index in interior] + [end_power]
    energy = sum(
        (integration_times[index + 1] - integration_times[index])
        * (integration_power[index + 1] + integration_power[index])
        / 2.0
        for index in range(len(integration_times) - 1)
    )
    return energy, max(integration_power)


def validate_powerz_evidence(
    meter_path: Path,
    handshake: Mapping[str, Any],
    *,
    run_id: str,
    minimum_roll_s: float = 5.0,
    power_relative_tolerance: float = 1e-4,
    power_absolute_tolerance_w: float = 1e-6,
) -> dict[str, Any]:
    """Validate raw samples and independently integrate between Mac ACK receipts."""
    if handshake.get("run_id") != run_id:
        raise PowerEvidenceError("handshake run_id mismatch")
    start_ack = handshake.get("pi_start_ack")
    end_ack = handshake.get("pi_end_ack")
    if not isinstance(start_ack, Mapping) or not isinstance(end_ack, Mapping):
        raise PowerEvidenceError("missing Pi start/end ACK payload")
    if (
        start_ack.get("event") != "RUN_START_ACK"
        or end_ack.get("event") != "RUN_END_ACK"
        or start_ack.get("run_id") != run_id
        or end_ack.get("run_id") != run_id
    ):
        raise PowerEvidenceError("Pi ACK identity mismatch")

    names = (
        "meter_capture_start_mac_monotonic",
        "start_request_mac_monotonic",
        "start_ack_receive_mac_monotonic",
        "end_ack_receive_mac_monotonic",
        "meter_capture_end_mac_monotonic",
    )
    try:
        order = [float(handshake[name]) for name in names]
    except (KeyError, TypeError, ValueError) as error:
        raise PowerEvidenceError("missing or malformed Mac monotonic boundary") from error
    if not all(math.isfinite(value) for value in order):
        raise PowerEvidenceError("non-finite Mac monotonic boundary")
    if not (order[0] < order[1] <= order[2] < order[3] < order[4]):
        raise PowerEvidenceError("invalid Mac monotonic ACK ordering")

    with meter_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != SCHEMA:
            raise PowerEvidenceError("POWER-Z CSV schema mismatch")
        rows = list(reader)
    if len(rows) < 2:
        raise PowerEvidenceError("insufficient POWER-Z samples")

    utc_times: list[float] = []
    sample_times: list[float] = []
    voltages: list[float] = []
    currents: list[float] = []
    powers: list[float] = []
    for row_number, row in enumerate(rows, start=2):
        utc_times.append(_utc_seconds(row["timestamp_utc"], row_number))
        try:
            sample_time = float(row["sample_monotonic"])
            voltage = float(row["voltage_V"])
            current = float(row["current_A"])
            power = float(row["power_W"])
        except ValueError as error:
            raise PowerEvidenceError(f"row {row_number}: malformed numeric field") from error
        if not all(math.isfinite(value) for value in (sample_time, voltage, current, power)):
            raise PowerEvidenceError(f"row {row_number}: non-finite numeric field")
        if voltage < 0 or current < 0 or power < 0:
            raise PowerEvidenceError(f"row {row_number}: negative V/I/P value")
        if not math.isclose(
            power,
            voltage * current,
            rel_tol=power_relative_tolerance,
            abs_tol=power_absolute_tolerance_w,
        ):
            raise PowerEvidenceError(f"row {row_number}: power is inconsistent with V*I")
        raw_hex = row["raw_hex"]
        if len(raw_hex) != 128:
            raise PowerEvidenceError(f"row {row_number}: raw_hex length is not 128")
        try:
            raw = bytes.fromhex(raw_hex)
        except ValueError as error:
            raise PowerEvidenceError(f"row {row_number}: raw_hex is not decodable") from error
        if len(raw) != 64:
            raise PowerEvidenceError(f"row {row_number}: raw_hex is not 64 bytes")
        raw_microvolts, raw_microamps = struct.unpack_from("<ii", raw, 16)
        raw_voltage = raw_microvolts / 1e6
        raw_current = -raw_microamps / 1e6
        if not math.isclose(voltage, raw_voltage, rel_tol=0.0, abs_tol=1e-9):
            raise PowerEvidenceError(
                f"row {row_number}: voltage is inconsistent with raw HID packet"
            )
        if not math.isclose(current, raw_current, rel_tol=0.0, abs_tol=1e-9):
            raise PowerEvidenceError(
                f"row {row_number}: current is inconsistent with raw HID packet"
            )
        sample_times.append(sample_time)
        voltages.append(voltage)
        currents.append(current)
        powers.append(power)

    if any(right <= left for left, right in zip(utc_times, utc_times[1:])):
        raise PowerEvidenceError("timestamp_utc values are not strictly increasing")
    if any(right <= left for left, right in zip(sample_times, sample_times[1:])):
        raise PowerEvidenceError("sample_monotonic values are not strictly increasing")

    start, end = order[2], order[3]
    if not sample_times[0] <= start < end <= sample_times[-1]:
        raise PowerEvidenceError("Mac ACK boundaries fall outside POWER-Z samples")
    pre_roll = start - sample_times[0]
    post_roll = sample_times[-1] - end
    if pre_roll < minimum_roll_s or post_roll < minimum_roll_s:
        raise PowerEvidenceError(
            f"insufficient POWER-Z roll: pre={pre_roll:.6f}, post={post_roll:.6f}"
        )

    energy_j, peak_power = _integrate(sample_times, powers, start, end)
    duration = end - start
    if not math.isfinite(energy_j) or energy_j < 0:
        raise PowerEvidenceError("invalid POWER-Z energy integral")
    return {
        "status": "PASS",
        "run_id": run_id,
        "samples": len(rows),
        "sha256": _sha256(meter_path),
        "schema": list(SCHEMA),
        "raw_packets_64_bytes": True,
        "raw_packet_measurements_consistent": True,
        "utc_strictly_increasing": True,
        "sample_monotonic_strictly_increasing": True,
        "voltage_current_power_consistent": True,
        "boundary": "start_ack_receive_mac_monotonic_to_end_ack_receive_mac_monotonic",
        "endpoint_extension": False,
        "start_request_to_ack_ms": (order[2] - order[1]) * 1000.0,
        "pre_roll_s": pre_roll,
        "post_roll_s": post_roll,
        "duration_s": duration,
        "energy_J": energy_j,
        "mean_power_W": energy_j / duration,
        "peak_observed_with_interpolated_boundaries_W": peak_power,
        "voltage_V_min": min(voltages),
        "voltage_V_max": max(voltages),
        "current_A_min": min(currents),
        "current_A_max": max(currents),
    }
