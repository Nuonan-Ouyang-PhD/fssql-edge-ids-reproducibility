#!/usr/bin/env python3
"""Execute the frozen Pi4B8 formal matrix behind a fail-closed gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
POLICIES = ["FSSQL-R", "Safe-Greedy", "Threshold", "Static-Light"]
CONDITIONS = [
    ("n16/B0", "natural"),
    ("n16/B1", "robustness_B1"),
    ("n16/B2", "robustness_B2"),
]
SEEDS = [5101, 5102, 5103, 5104, 5105]
TIMING_FIELDS = (
    "preprocessing_ms",
    "telemetry_ms",
    "state_construction_ms",
    "guard_ms",
    "admissible_set_ms",
    "policy_selection_ms",
    "q_lookup_ms",
    "detector_inference_ms",
    "switching_overhead_ms",
    "controller_ms",
    "e2e_ms",
)


class RunValidationError(RuntimeError):
    """A formal run failed a frozen technical validity condition."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RunValidationError(f"missing CSV: {path}")
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def parse_ack(line: str, event: str, run_id: str) -> dict:
    try:
        ack = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RunValidationError(f"malformed {event}: {line!r}") from exc
    if ack.get("event") != event or ack.get("run_id") != run_id:
        raise RunValidationError(f"{event} run binding mismatch: {ack}")
    for field in ("pi_utc", "pi_monotonic"):
        if not math.isfinite(float(ack.get(field, float("nan")))):
            raise RunValidationError(f"non-finite {event} {field}")
    return ack


def validate_windows(
    path: Path,
    *,
    run_id: str,
    policy: str,
    condition: str,
    seed: int,
    windows: int,
    min_frequency_mhz: float,
    end_ack: dict,
) -> dict:
    records = read_csv(path)
    if len(records) != windows:
        raise RunValidationError(f"{run_id}: expected {windows} windows, found {len(records)}")
    if [int(row["window_id"]) for row in records] != list(range(windows)):
        raise RunValidationError(f"{run_id}: window IDs are incomplete or out of order")
    if any(row["policy"] != policy for row in records):
        raise RunValidationError(f"{run_id}: policy mismatch in windows.csv")
    if any(row["condition"] != condition for row in records):
        raise RunValidationError(f"{run_id}: condition mismatch in windows.csv")
    if any(int(row["seed"]) != seed for row in records):
        raise RunValidationError(f"{run_id}: seed mismatch in windows.csv")

    numeric_fields = TIMING_FIELDS + (
        "predicted_latency_ms",
        "latency_headroom_ms",
        "temperature_C",
        "cpu_frequency_mhz_min",
    )
    for row in records:
        for field in numeric_fields:
            try:
                value = float(row[field])
            except (KeyError, ValueError) as exc:
                raise RunValidationError(f"{run_id}: invalid numeric field {field}") from exc
            if not math.isfinite(value) or (field in TIMING_FIELDS and value < 0):
                raise RunValidationError(f"{run_id}: invalid value for {field}: {value}")
        if float(row["cpu_frequency_mhz_min"]) < min_frequency_mhz:
            raise RunValidationError(f"{run_id}: frequency invariant failed")
        try:
            throttle = int(row["throttled"].split("=", 1)[1], 16)
        except (KeyError, IndexError, ValueError) as exc:
            raise RunValidationError(f"{run_id}: malformed throttle status") from exc
        if throttle != 0:
            raise RunValidationError(f"{run_id}: throttle/undervoltage status {throttle:#x}")
        prediction = int(row["prediction"])
        label = int(row["label_scoring_only"])
        expected = (
            int(prediction == 1 and label == 1),
            int(prediction == 1 and label == 0),
            int(prediction == 0 and label == 1),
            int(prediction == 0 and label == 0),
        )
        observed = tuple(int(row[field]) for field in ("tp", "fp", "fn", "tn"))
        if observed != expected:
            raise RunValidationError(f"{run_id}: confusion contribution is not reproducible")

    observed_hash = sha256(path)
    if end_ack.get("windows_completed") != windows:
        raise RunValidationError(f"{run_id}: RUN_END_ACK window count mismatch")
    if end_ack.get("raw_artifact_sha256") != observed_hash:
        raise RunValidationError(f"{run_id}: Pi raw hash mismatch")
    return {"rows": len(records), "sha256": observed_hash}


def validate_power_capture(
    path: Path, handshake: dict, *, run_id: str, minimum_roll_s: float = 5.0
) -> dict:
    if handshake.get("run_id") != run_id:
        raise RunValidationError(f"{run_id}: handshake run_id mismatch")
    start_ack = parse_ack(json.dumps(handshake["pi_start_ack"]), "RUN_START_ACK", run_id)
    end_ack = parse_ack(json.dumps(handshake["pi_end_ack"]), "RUN_END_ACK", run_id)

    order = [
        float(handshake["meter_capture_start_mac_monotonic"]),
        float(handshake["start_request_mac_monotonic"]),
        float(handshake["start_ack_receive_mac_monotonic"]),
        float(handshake["end_ack_receive_mac_monotonic"]),
        float(handshake["meter_capture_end_mac_monotonic"]),
    ]
    if not all(math.isfinite(value) for value in order):
        raise RunValidationError(f"{run_id}: non-finite Mac handshake timestamp")
    if not (order[0] < order[1] <= order[2] < order[3] < order[4]):
        raise RunValidationError(f"{run_id}: invalid Mac handshake ordering")
    if float(start_ack["pi_monotonic"]) >= float(end_ack["pi_monotonic"]):
        raise RunValidationError(f"{run_id}: invalid Pi ACK ordering")

    records = read_csv(path)
    if len(records) < 2:
        raise RunValidationError(f"{run_id}: insufficient POWER-Z samples")
    required = ("sample_monotonic", "voltage_V", "current_A", "power_W")
    arrays = {}
    for field in required:
        try:
            values = np.array([float(row[field]) for row in records], dtype=float)
        except (KeyError, ValueError) as exc:
            raise RunValidationError(f"{run_id}: invalid POWER-Z field {field}") from exc
        if not np.all(np.isfinite(values)):
            raise RunValidationError(f"{run_id}: non-finite POWER-Z field {field}")
        arrays[field] = values

    sample_times = arrays["sample_monotonic"]
    if not np.all(np.diff(sample_times) > 0):
        raise RunValidationError(f"{run_id}: POWER-Z timestamps are not strictly monotonic")
    start = order[2]
    end = order[3]
    if not (sample_times[0] <= start < end <= sample_times[-1]):
        raise RunValidationError(f"{run_id}: ACK boundaries fall outside POWER-Z samples")
    pre_roll = start - sample_times[0]
    post_roll = sample_times[-1] - end
    if pre_roll < minimum_roll_s or post_roll < minimum_roll_s:
        raise RunValidationError(
            f"{run_id}: insufficient POWER-Z roll: pre={pre_roll:.6f}, post={post_roll:.6f}"
        )

    interior = (sample_times > start) & (sample_times < end)
    integration_times = np.r_[start, sample_times[interior], end]
    integration_power = np.interp(
        integration_times, sample_times, arrays["power_W"]
    )
    energy = float(np.trapezoid(integration_power, integration_times))
    if not math.isfinite(energy) or energy < 0:
        raise RunValidationError(f"{run_id}: invalid integrated energy")
    return {
        "samples": len(records),
        "sha256": sha256(path),
        "pre_roll_s": pre_roll,
        "post_roll_s": post_roll,
        "energy_J": energy,
        "mean_power_W": energy / (end - start),
        "peak_power_W": float(integration_power.max()),
    }


def run_formal_gate(python: str) -> dict:
    command = [
        python,
        str(ROOT / "online/pi4b8/pi4b8_formal_runner.py"),
        "--formal",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        raise RunValidationError(f"cannot execute formal gate: {exc}") from exc
    if result.returncode:
        raise RunValidationError(f"formal gate failed: {result.stdout}{result.stderr}".strip())
    try:
        gate = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RunValidationError("formal gate returned malformed JSON") from exc
    if gate.get("status") != "PI4B8_FINAL_FORMAL_GATE_PASS":
        raise RunValidationError(f"formal gate did not authorize execution: {gate}")
    return gate


def build_order() -> list[tuple[int, str, str, str]]:
    order = []
    for seed in SEEDS:
        rng = np.random.default_rng(20260910 + seed)
        block = [(policy, condition) for condition, _ in CONDITIONS for policy in POLICIES]
        rng.shuffle(block)
        for policy, condition in block:
            condition_dir = next(name for value, name in CONDITIONS if value == condition)
            order.append((seed, policy, condition, condition_dir))
    return order


def worker_command(
    run_id: str, seed: int, policy: str, condition: str, condition_dir: str
) -> str:
    command = [
        "python3",
        "pi4b8_online_worker.py",
        "--features",
        f"inputs/{condition_dir}/{seed}/features.npy",
        "--risk-features",
        f"inputs/{condition_dir}/{seed}/risk_features.npy",
        "--labels",
        f"scoring/{condition_dir}/{seed}/labels.npy",
        "--policy",
        policy,
        "--condition",
        condition,
        "--seed",
        str(seed),
        "--run-id",
        run_id,
        "--output",
        run_id,
        "--windows",
        "600",
    ]
    if policy == "FSSQL-R":
        command.extend(["--q-table", f"q_tables/fssql_r_seed_{seed}.npz"])
    return "cd /home/pi/tsusc_major_revision/pi4b8_formal_runtime_v4 && " + shlex.join(
        command
    )


def execute_cell(
    output_root: Path,
    python: str,
    index: int,
    seed: int,
    policy: str,
    condition: str,
    condition_dir: str,
    min_frequency_mhz: float,
) -> None:
    run_id = (
        f"pi4b8_formal_{index:02d}_{policy.lower().replace('-', '_')}_"
        f"{condition.replace('/', '_')}_s{seed}"
    )
    output = output_root / run_id
    output.mkdir(exist_ok=False)
    meter_file = output / "POWERZ_raw.csv"
    meter_command = [
        python,
        str(ROOT / "online/pi4b8/powerz_hid_capture.py"),
        "--output",
        str(meter_file),
        "--seconds",
        "80",
        "--sps",
        "10",
    ]
    meter_start = time.monotonic()
    meter = subprocess.Popen(
        meter_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    worker = None
    handshake = {"run_id": run_id, "status": "INVALID_PENDING_VALIDATION"}
    try:
        time.sleep(5)
        request = time.monotonic()
        worker = subprocess.Popen(
            ["ssh", "pi@pi4b8g.local", worker_command(run_id, seed, policy, condition, condition_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        start_line = worker.stdout.readline()
        start_ack_time = time.monotonic()
        start_ack = parse_ack(start_line, "RUN_START_ACK", run_id)
        lines = [start_line]
        end_line = ""
        end_ack_time = None
        for line in worker.stdout:
            lines.append(line)
            if '"event": "RUN_END_ACK"' in line:
                end_line = line
                end_ack_time = time.monotonic()
                break
        worker.wait()
        worker_stderr = worker.stderr.read()
        (output / "worker.stdout").write_text("".join(lines))
        (output / "worker.stderr").write_text(worker_stderr)
        if worker.returncode:
            raise RunValidationError(f"{run_id}: worker exited {worker.returncode}")
        if end_ack_time is None:
            raise RunValidationError(f"{run_id}: missing RUN_END_ACK")
        end_ack = parse_ack(end_line, "RUN_END_ACK", run_id)

        transfer = subprocess.run(
            [
                "scp",
                "-r",
                f"pi@pi4b8g.local:/home/pi/tsusc_major_revision/"
                f"pi4b8_formal_runtime_v4/{run_id}",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        (output / "scp.stdout").write_text(transfer.stdout)
        (output / "scp.stderr").write_text(transfer.stderr)
        if transfer.returncode:
            raise RunValidationError(f"{run_id}: Pi evidence retrieval failed")

        time.sleep(5)
        try:
            meter_stdout, meter_stderr = meter.communicate(timeout=15)
        except subprocess.TimeoutExpired as exc:
            meter.terminate()
            meter.communicate()
            raise RunValidationError(f"{run_id}: POWER-Z capture did not terminate") from exc
        meter_end = time.monotonic()
        (output / "meter.stdout").write_text(meter_stdout)
        (output / "meter.stderr").write_text(meter_stderr)
        if meter.returncode:
            raise RunValidationError(f"{run_id}: POWER-Z capture exited {meter.returncode}")

        handshake = {
            "run_id": run_id,
            "seed": seed,
            "policy": policy,
            "condition": condition,
            "meter_capture_start_mac_monotonic": meter_start,
            "start_request_mac_monotonic": request,
            "start_ack_receive_mac_monotonic": start_ack_time,
            "end_ack_receive_mac_monotonic": end_ack_time,
            "meter_capture_end_mac_monotonic": meter_end,
            "pi_start_ack": start_ack,
            "pi_end_ack": end_ack,
            "status": "PENDING_VALIDATION",
        }
        handshake_path = output / "HANDSHAKE.json"
        handshake_path.write_text(json.dumps(handshake, indent=2) + "\n")

        window_files = list(output.glob("*/windows.csv"))
        if len(window_files) != 1:
            raise RunValidationError(
                f"{run_id}: expected one retrieved windows.csv, found {len(window_files)}"
            )
        pi_result = validate_windows(
            window_files[0],
            run_id=run_id,
            policy=policy,
            condition=condition,
            seed=seed,
            windows=600,
            min_frequency_mhz=min_frequency_mhz,
            end_ack=end_ack,
        )
        power_result = validate_power_capture(meter_file, handshake, run_id=run_id)
        handshake["status"] = "VALID_COMPLETE"
        handshake_path.write_text(json.dumps(handshake, indent=2) + "\n")
        validation = {
            "status": "VALID_COMPLETE",
            "run_id": run_id,
            "pi": pi_result,
            "powerz": power_result,
            "handshake_sha256": sha256(handshake_path),
        }
        (output / "RUN_VALIDATION.json").write_text(
            json.dumps(validation, indent=2) + "\n"
        )
        manifest = {
            "run_id": run_id,
            "policy": policy,
            "condition": condition,
            "seed": seed,
            "pi_raw": str(window_files[0].relative_to(output)),
            "pi_raw_sha256": pi_result["sha256"],
            "powerz_raw": meter_file.name,
            "powerz_raw_sha256": power_result["sha256"],
            "handshake": handshake_path.name,
            "handshake_sha256": sha256(handshake_path),
            "validity_status": "VALID_COMPLETE",
        }
        (output / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    except Exception as exc:
        handshake["status"] = "INVALID_TECHNICAL"
        handshake["reason"] = str(exc)
        (output / "HANDSHAKE.json").write_text(json.dumps(handshake, indent=2) + "\n")
        (output / "RUN_VALIDATION.json").write_text(
            json.dumps(
                {"status": "INVALID_TECHNICAL", "run_id": run_id, "reason": str(exc)},
                indent=2,
            )
            + "\n"
        )
        raise
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            worker.wait()
        if meter.poll() is None:
            meter.terminate()
            meter.wait()


def main() -> None:
    raise RunValidationError(
        "historical V4 launcher disabled after the 2026-10-21 audit; "
        "the corrected worker/package/schema contract is not yet frozen"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, choices=range(1, 61))
    args = parser.parse_args()
    gate = run_formal_gate(sys.executable)
    python_path = ROOT / ".venv/bin/python"
    if not python_path.is_file():
        raise RunValidationError(f"missing frozen Mac acquisition runtime: {python_path}")
    python = str(python_path)
    config = json.loads((ROOT / "configs/pi4b8_online_experiment_v1.json").read_text())
    output_root = ROOT / "evidence/pi4b8_formal_matrix"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "FORMAL_GATE_RECEIPT.json").write_text(
        json.dumps(gate, indent=2) + "\n"
    )

    for index, (seed, policy, condition, condition_dir) in enumerate(build_order(), 1):
        if index < args.start:
            continue
        execute_cell(
            output_root,
            python,
            index,
            seed,
            policy,
            condition,
            condition_dir,
            float(config["cooldown"]["min_frequency_mhz"]),
        )
        print(f"{index}/60 VALID_COMPLETE", flush=True)
    print("60-RUN MATRIX EXECUTION COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except RunValidationError as exc:
        raise SystemExit(f"Pi4B8 formal execution stopped: {exc}") from exc
