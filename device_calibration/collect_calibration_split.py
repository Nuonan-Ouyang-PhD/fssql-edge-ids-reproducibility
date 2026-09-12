#!/usr/bin/env python3
"""Collect one independent Pi-local calibration split.

This collector produces raw timing and thermal transitions only. Predictor
fitting and guard-margin selection are intentionally separate from collection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parent
REPOSITORY_RUNTIME = ROOT.parent / "runtime"
if REPOSITORY_RUNTIME.is_dir():
    sys.path.insert(0, str(REPOSITORY_RUNTIME))

from action_pool_numpy import NumpyActionModel
from preprocessor_numpy import PortablePreprocessor
from risk_proxy_numpy import RiskProxy


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def command(args: list[str]) -> str:
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def temperature_c() -> float:
    value = int(Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()) / 1000.0
    if not math.isfinite(value):
        raise RuntimeError("non-finite temperature")
    return value


def throttled() -> tuple[str, int]:
    text = command(["vcgencmd", "get_throttled"])
    return text, int(text.split("=")[-1], 16)


def governors() -> list[str]:
    paths = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor"))
    return [path.read_text().strip() for path in paths]


def cpu_frequencies_mhz() -> list[float]:
    paths = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"))
    if not paths:
        raise RuntimeError("CPU frequency telemetry is unavailable")
    return [int(path.read_text().strip()) / 1000.0 for path in paths]


def busy_worker(stop: mp.synchronize.Event, duty_cycle: float, period_s: float, cpu: int) -> None:
    os.sched_setaffinity(0, {cpu})
    busy_s = period_s * duty_cycle
    idle_s = period_s - busy_s
    while not stop.is_set():
        start = time.perf_counter()
        while time.perf_counter() - start < busy_s:
            pass
        if idle_s > 0:
            stop.wait(idle_s)


class BackgroundLoad:
    def __init__(self, condition: dict[str, object]):
        self.condition = condition
        self.stop: mp.synchronize.Event | None = None
        self.process: mp.Process | None = None

    def __enter__(self) -> "BackgroundLoad":
        if int(self.condition["busy_cpu_count"]) == 0:
            return self
        context = mp.get_context("spawn")
        self.stop = context.Event()
        self.process = context.Process(
            target=busy_worker,
            args=(
                self.stop,
                float(self.condition["duty_cycle"]),
                float(self.condition["period_ms"]) / 1000.0,
                int(self.condition["pinned_cpu"]),
            ),
        )
        self.process.start()
        return self

    def __exit__(self, *_: object) -> None:
        if self.stop is not None:
            self.stop.set()
        if self.process is not None:
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)


class PowerMonitor:
    def __init__(self, output: Path, interval_s: float):
        self.output = output
        self.interval_s = interval_s
        self.stop = threading.Event()
        self.current_fault = threading.Event()
        self.samples = 0
        self.failure_reason = ""
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            with self.output.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["timestamp_utc", "temperature_c", "throttled"])
                while not self.stop.is_set():
                    text, value = throttled()
                    writer.writerow([utc_now(), temperature_c(), text.split("=")[-1]])
                    stream.flush()
                    self.samples += 1
                    if value & 0xF:
                        self.current_fault.set()
                    self.stop.wait(self.interval_s)
        except BaseException as exc:
            self.failure_reason = f"{type(exc).__name__}: {exc}"

    def __enter__(self) -> "PowerMonitor":
        self.thread.start()
        return self

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=self.interval_s + 3)
        if self.thread.is_alive() and not self.failure_reason:
            self.failure_reason = "power monitor did not stop"

    def __exit__(self, *_: object) -> None:
        self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("fit", "margin", "coverage"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--development-limit-cells", type=int, default=None,
                        help="Non-formal code-path test only; recorded as development evidence.")
    parser.add_argument("--development-observations", type=int, default=None,
                        help="Non-formal code-path test only; recorded as development evidence.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    if socket.gethostname() != config["device_id"]:
        raise RuntimeError(f"wrong host: {socket.gethostname()}")
    if config["evidence_source"] != "scheduler_train only":
        raise RuntimeError("unexpected evidence source")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")

    formal = args.development_limit_cells is None and args.development_observations is None
    if formal and config["status"] != "FROZEN_BEFORE_FORMAL_CALIBRATION":
        raise RuntimeError("formal collection requires frozen config")
    if formal and (len(args.git_commit) != 40 or any(c not in "0123456789abcdef" for c in args.git_commit)):
        raise RuntimeError("formal collection requires a full lowercase git commit hash")
    args.output.mkdir(parents=True)

    run_start = utc_now()
    initial_throttled_text, initial_throttled_value = throttled()
    if initial_throttled_value & 0xF:
        raise RuntimeError(f"current throttle fault before start: {initial_throttled_text}")
    observed_governors = governors()
    if set(observed_governors) != {config["expected_governor"]}:
        raise RuntimeError(f"unexpected governors: {observed_governors}")

    idle_rows: list[list[object]] = []
    idle = config["idle_baseline"]
    idle_samples = int(idle["samples"]) if formal else 3
    idle_interval = float(idle["interval_seconds"]) if formal else 0.1
    for sample in range(idle_samples):
        text, value = throttled()
        temp = temperature_c()
        idle_rows.append([sample, utc_now(), temp, text.split("=")[-1], os.getloadavg()[0]])
        if value & 0xF:
            raise RuntimeError(f"current throttle fault during idle baseline: {text}")
        if sample + 1 < idle_samples:
            time.sleep(idle_interval)
    with (args.output / "idle_baseline.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["sample", "timestamp_utc", "temperature_c", "throttled", "load1"])
        writer.writerows(idle_rows)
    idle_baseline_c = float(np.median([row[2] for row in idle_rows]))

    with (ROOT / "scheduler_train_6000_raw_features.csv").open(newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream))
    features = np.load(ROOT / "scheduler_train_6000_features.npy", allow_pickle=False)
    risk_features = np.load(ROOT / "scheduler_train_6000_risk_features.npy", allow_pickle=False)
    preprocessor = PortablePreprocessor(ROOT / "preprocessor_portable.npz")
    risk_proxy = RiskProxy(ROOT / "risk_proxy.npz")
    actions = list(config["actions"])
    model_manifest_path = ROOT / "MODEL_SHA256.json"
    model_manifest = json.loads(model_manifest_path.read_text())
    for action in actions:
        relative_path = f"models/artifacts/{action}.npz"
        expected = model_manifest["files"][relative_path]["sha256"]
        actual = sha256(ROOT / f"{action}.npz")
        if actual != expected:
            raise RuntimeError(f"model artifact hash mismatch: {action}.npz")
    models = {action: NumpyActionModel(ROOT / f"{action}.npz") for action in actions}
    for model in models.values():
        for _ in range(10):
            model.predict_proba(features[:16])
    start_temperature_c = temperature_c()
    start_limit_c = idle_baseline_c + float(idle["start_limit_c_above_baseline"])
    if start_temperature_c > start_limit_c:
        raise RuntimeError(
            f"start temperature {start_temperature_c:.3f} C exceeds "
            f"idle baseline limit {start_limit_c:.3f} C"
        )

    split_seed = int(config["split_seeds"][args.split])
    rng = np.random.default_rng(split_seed)
    row_order = rng.permutation(len(features))
    row_cursor = 0
    conditions = list(product(actions, config["workload_records_per_window"], config["background_conditions"]))
    rng.shuffle(conditions)
    if args.development_limit_cells is not None:
        conditions = conditions[:args.development_limit_cells]
    observations = (
        int(args.development_observations)
        if args.development_observations is not None
        else int(config["observations_per_action_workload_background_cell"])
    )

    raw_path = args.output / "raw_windows.csv"
    block_path = args.output / "blocks.csv"
    raw_fields = [
        "run_id", "device_id", "protocol_version", "round", "split", "formal", "block_index",
        "window_in_block", "obs_id", "wall_start_utc", "mono_start_ns",
        "action", "workload_bin", "n_records", "background", "background_duty",
        "T_start_C", "T_end_C", "cpu_freq_start_min_mhz", "cpu_freq_start_max_mhz",
        "cpu_freq_end_min_mhz", "cpu_freq_end_max_mhz", "telemetry_ms", "feature_ms", "risk_ms",
        "inference_ms", "controller_basic_ms", "end_to_end_ms", "window_elapsed_ms",
        "deadline_overrun_ms", "mean_risk", "mean_action_probability", "valid",
    ]
    block_fields = [
        "block_index", "action", "workload_bin", "background", "start_utc", "end_utc",
        "throttled_before", "throttled_after", "temperature_before_c", "temperature_after_c",
        "windows_expected", "windows_observed", "valid", "invalidation_reason",
    ]
    status = "RUNNING"
    failure_reason = ""
    observed_windows = 0
    emergency_stop_c = float(config["emergency_stop_c"])
    decision_window_s = float(config["decision_window_ms"]) / 1000.0

    with raw_path.open("x", newline="", encoding="utf-8", buffering=1) as raw_stream, \
            block_path.open("x", newline="", encoding="utf-8", buffering=1) as block_stream, \
            PowerMonitor(args.output / "power_telemetry_1hz.csv", float(config["power_monitor_interval_seconds"])) as monitor:
        raw_writer = csv.DictWriter(raw_stream, fieldnames=raw_fields, lineterminator="\n")
        block_writer = csv.DictWriter(block_stream, fieldnames=block_fields, lineterminator="\n")
        raw_writer.writeheader()
        block_writer.writeheader()
        try:
            for block_index, (action, workload, background) in enumerate(conditions):
                if monitor.failure_reason:
                    raise RuntimeError(f"power monitor failed: {monitor.failure_reason}")
                if monitor.current_fault.is_set():
                    raise RuntimeError("current throttle fault observed by independent monitor")
                before_text, before_value = throttled()
                before_temp = temperature_c()
                if before_value & 0xF:
                    raise RuntimeError(f"current throttle fault before block: {before_text}")
                block_start = utc_now()
                block_observed = 0
                block_valid = True
                block_reason = ""
                with BackgroundLoad(background):
                    time.sleep(float(config["block_stabilization_seconds"]) if formal else 0.05)
                    for window_in_block in range(observations):
                        if monitor.failure_reason:
                            raise RuntimeError(f"power monitor failed: {monitor.failure_reason}")
                        if monitor.current_fault.is_set():
                            raise RuntimeError("current throttle fault observed by independent monitor")
                        take = int(workload)
                        positions = np.arange(row_cursor, row_cursor + take) % len(row_order)
                        indices = row_order[positions]
                        row_cursor = (row_cursor + take) % len(row_order)
                        selected_raw = [raw_rows[int(index)] for index in indices]

                        mono_start = time.perf_counter_ns()
                        wall_start = utc_now()
                        start = time.perf_counter_ns()
                        start_temp = temperature_c()
                        start_frequencies = cpu_frequencies_mhz()
                        telemetry_ns = time.perf_counter_ns() - start

                        start = time.perf_counter_ns()
                        transformed = preprocessor.transform(selected_raw)
                        feature_ns = time.perf_counter_ns() - start
                        if not np.array_equal(transformed, features[indices]):
                            raise RuntimeError("portable preprocessing parity failure")

                        start = time.perf_counter_ns()
                        risks = risk_proxy.predict(risk_features[indices])
                        risk_ns = time.perf_counter_ns() - start
                        start = time.perf_counter_ns()
                        probabilities = models[action].predict_proba(transformed)
                        inference_ns = time.perf_counter_ns() - start
                        if not np.isfinite(risks).all() or not np.isfinite(probabilities).all():
                            raise RuntimeError("non-finite probability")

                        start = time.perf_counter_ns()
                        mean_risk = float(np.mean(risks))
                        basic_scores = [mean_risk * (index + 1) - 0.001 * take for index in range(len(actions))]
                        _ = max(range(len(actions)), key=basic_scores.__getitem__)
                        controller_ns = time.perf_counter_ns() - start
                        processing_end = time.perf_counter_ns()
                        end_to_end_ms = (processing_end - mono_start) / 1e6
                        remaining = decision_window_s - (processing_end - mono_start) / 1e9
                        if remaining > 0:
                            time.sleep(remaining)
                        end_temp = temperature_c()
                        end_frequencies = cpu_frequencies_mhz()
                        elapsed_ms = (time.perf_counter_ns() - mono_start) / 1e6
                        if max(start_temp, end_temp) >= emergency_stop_c:
                            raise RuntimeError(f"emergency temperature stop: {max(start_temp, end_temp):.3f} C")
                        frequency_invariant = config.get("frequency_invariant")
                        if frequency_invariant is not None:
                            minimum_mhz = float(frequency_invariant["minimum_observed_mhz"])
                            if min(start_frequencies + end_frequencies) < minimum_mhz:
                                raise RuntimeError(
                                    "CPU frequency invariant failed: "
                                    f"{min(start_frequencies + end_frequencies):.3f} < {minimum_mhz:.3f} MHz"
                                )

                        raw_writer.writerow({
                            "run_id": args.run_id,
                            "device_id": config["device_id"],
                            "protocol_version": config.get("protocol_version", 1),
                            "round": config["calibration_round"],
                            "split": args.split,
                            "formal": str(formal).lower(),
                            "block_index": block_index,
                            "window_in_block": window_in_block,
                            "obs_id": observed_windows,
                            "wall_start_utc": wall_start,
                            "mono_start_ns": mono_start,
                            "action": action,
                            "workload_bin": f"n{take}",
                            "n_records": take,
                            "background": background["id"],
                            "background_duty": background["duty_cycle"],
                            "T_start_C": f"{start_temp:.3f}",
                            "T_end_C": f"{end_temp:.3f}",
                            "cpu_freq_start_min_mhz": f"{min(start_frequencies):.3f}",
                            "cpu_freq_start_max_mhz": f"{max(start_frequencies):.3f}",
                            "cpu_freq_end_min_mhz": f"{min(end_frequencies):.3f}",
                            "cpu_freq_end_max_mhz": f"{max(end_frequencies):.3f}",
                            "telemetry_ms": f"{telemetry_ns / 1e6:.6f}",
                            "feature_ms": f"{feature_ns / 1e6:.6f}",
                            "risk_ms": f"{risk_ns / 1e6:.6f}",
                            "inference_ms": f"{inference_ns / 1e6:.6f}",
                            "controller_basic_ms": f"{controller_ns / 1e6:.6f}",
                            "end_to_end_ms": f"{end_to_end_ms:.6f}",
                            "window_elapsed_ms": f"{elapsed_ms:.6f}",
                            "deadline_overrun_ms": f"{max(0.0, end_to_end_ms - float(config['decision_window_ms'])):.6f}",
                            "mean_risk": f"{mean_risk:.9f}",
                            "mean_action_probability": f"{float(np.mean(probabilities)):.9f}",
                            "valid": 1,
                        })
                        observed_windows += 1
                        block_observed += 1
                after_text, after_value = throttled()
                after_temp = temperature_c()
                if after_value & 0xF:
                    block_valid = False
                    block_reason = f"current throttle fault after block: {after_text}"
                block_writer.writerow({
                    "block_index": block_index,
                    "action": action,
                    "workload_bin": f"n{int(workload)}",
                    "background": background["id"],
                    "start_utc": block_start,
                    "end_utc": utc_now(),
                    "throttled_before": before_text.split("=")[-1],
                    "throttled_after": after_text.split("=")[-1],
                    "temperature_before_c": f"{before_temp:.3f}",
                    "temperature_after_c": f"{after_temp:.3f}",
                    "windows_expected": observations,
                    "windows_observed": block_observed,
                    "valid": int(block_valid),
                    "invalidation_reason": block_reason,
                })
                if not block_valid:
                    raise RuntimeError(block_reason)
            status = "FORMAL_CALIBRATION_SPLIT_PASS" if formal else "DEVELOPMENT_CODE_PATH_PASS"
        except BaseException as exc:
            status = "FORMAL_CALIBRATION_SPLIT_INVALID" if formal else "DEVELOPMENT_CODE_PATH_INVALID"
            failure_reason = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            monitor.close()
            final_text, final_value = throttled()
            if monitor.failure_reason and not failure_reason:
                status = "FORMAL_CALIBRATION_SPLIT_INVALID" if formal else "DEVELOPMENT_CODE_PATH_INVALID"
                failure_reason = f"power monitor failed: {monitor.failure_reason}"
            elif monitor.current_fault.is_set() and not failure_reason:
                status = "FORMAL_CALIBRATION_SPLIT_INVALID" if formal else "DEVELOPMENT_CODE_PATH_INVALID"
                failure_reason = "current throttle fault observed by independent monitor"
            elif final_value & 0xF and not failure_reason:
                status = "FORMAL_CALIBRATION_SPLIT_INVALID" if formal else "DEVELOPMENT_CODE_PATH_INVALID"
                failure_reason = f"current throttle fault at end: {final_text}"
            manifest = {
                "run_id": args.run_id,
                "status": status,
                "failure_reason": failure_reason,
                "formal": formal,
                "device_id": config["device_id"],
                "protocol_version": config.get("protocol_version", 1),
                "hostname": platform.node(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "split": args.split,
                "split_seed": split_seed,
                "git_commit": args.git_commit,
                "run_start_utc": run_start,
                "run_end_utc": utc_now(),
                "config_sha256": sha256(args.config),
                "collector_sha256": sha256(Path(__file__)),
                "model_manifest_sha256": sha256(model_manifest_path),
                "input_manifest_sha256": sha256(ROOT / "MANIFEST.json"),
                "physical_condition_attestation_sha256": config.get("physical_condition_attestation_sha256"),
                "power_condition_attestation_sha256": config.get("power_condition_attestation_sha256"),
                "hardware_manifest_sha256": config.get("hardware_manifest_sha256"),
                "idle_baseline_c": idle_baseline_c,
                "start_temperature_c": start_temperature_c,
                "start_limit_c": start_limit_c,
                "idle_samples": idle_samples,
                "governors": observed_governors,
                "frequency_invariant": config.get("frequency_invariant"),
                "initial_throttled": initial_throttled_text,
                "final_throttled": final_text,
                "conditions_expected": len(conditions),
                "observations_per_condition": observations,
                "windows_expected": len(conditions) * observations,
                "windows_observed": observed_windows,
                "automatic_retry": False,
                "power_monitor_samples": monitor.samples,
                "raw_windows_sha256": sha256(raw_path),
                "blocks_sha256": sha256(block_path),
                "power_telemetry_sha256": sha256(args.output / "power_telemetry_1hz.csv"),
            }
            with (args.output / "RUN_MANIFEST.json").open("x", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2)
                stream.write("\n")
    if status.endswith("INVALID"):
        raise RuntimeError(failure_reason)


if __name__ == "__main__":
    main()
