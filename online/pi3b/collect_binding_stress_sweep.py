#!/usr/bin/env python3
"""Run the outcome-blind Protocol V3 binding-load diagnostic on Pi 3B+."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runtime"))

from action_pool_numpy import NumpyActionModel
from preprocessor_numpy import PortablePreprocessor
from device_calibration.collect_calibration_split import (
    BackgroundLoad,
    PowerMonitor,
    cpu_frequencies_mhz,
    governors,
    temperature_c,
    throttled,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def validate_inputs(config: dict[str, object]) -> None:
    for name, record in config["inputs"].items():
        path = ROOT / record["path"]
        if digest(path) != record["sha256"]:
            raise RuntimeError(f"frozen input hash mismatch: {name}")


def guarded_temperature(
    thermal: dict[str, object], margins: dict[str, object], action: str,
    temperature: float, n_records: int, background_duty: float, latency_hat: float,
) -> float:
    numeric = np.asarray(
        [temperature, math.log2(n_records), background_duty, latency_hat],
        dtype=np.float64,
    )
    mean = np.asarray(thermal["continuous_mean"], dtype=np.float64)
    scale = np.asarray(thermal["continuous_scale"], dtype=np.float64)
    indicators = np.asarray([
        float(action == "lucid_revision"),
        float(action == "tinydl_revision"),
        float(action == "oi_svdd_as_elm_revision"),
    ])
    design = np.concatenate(([1.0], (numeric - mean) / scale, indicators))
    prediction = float(design @ np.asarray(thermal["coefficients"], dtype=np.float64))
    return prediction + float(margins["epsilon_T_C_shared"]) + float(margins["fixed_thermal_buffer_c"])


def run_candidate(
    config: dict[str, object], candidate: dict[str, object], output: Path,
    raw_rows: list[dict[str, str]], features: np.ndarray,
    preprocessor: PortablePreprocessor, models: dict[str, NumpyActionModel],
    latency: dict[tuple[str, str, str], float], thermal: dict[str, object],
    margins: dict[str, object], formal_diagnostic: bool,
) -> dict[str, object]:
    output.mkdir()
    background = next(row for row in config["background_conditions"] if row["id"] == candidate["background"])
    n_records = int(candidate["n_records"])
    actions = list(config["actions"])
    idle_rows: list[list[object]] = []
    for sample in range(int(config["idle_baseline"]["samples"])):
        throttle_text, throttle_value = throttled()
        temp = temperature_c()
        idle_rows.append([sample, utc_now(), f"{temp:.3f}", throttle_text, f"{os.getloadavg()[0]:.6f}"])
        if throttle_value & 0xF:
            raise RuntimeError(f"current hardware fault during candidate idle baseline: {throttle_text}")
        if sample + 1 < int(config["idle_baseline"]["samples"]):
            time.sleep(float(config["idle_baseline"]["interval_seconds"]))
    idle_path = output / "idle_baseline.csv"
    with idle_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["sample", "timestamp_utc", "temperature_C", "throttled", "load1"])
        writer.writerows(idle_rows)
    idle_median = float(np.median([float(row[2]) for row in idle_rows]))
    start_temperature = temperature_c()
    if start_temperature > idle_median + float(config["idle_baseline"]["start_limit_c_above_baseline"]):
        raise RuntimeError("candidate start temperature exceeds frozen cooldown rule")

    window_path = output / "diagnostic_windows.csv"
    progress_path = output / "seed_progress.csv"
    power_path = output / "fault_telemetry_1hz.csv"
    fields = [
        "candidate_index", "seed", "window_id", "n_records", "background",
        "probe_action", "T_start_C", "T_end_C", "safe_set_size", "safe_set_mask",
        "fallback_required", "predicted_latency_by_action_ms",
        "guarded_latency_by_action_ms", "latency_headroom_by_action_ms",
        "guarded_temperature_by_action_C", "thermal_headroom_by_action_C",
        "cpu_frequency_min_mhz", "throttled_start", "throttled_end",
        "feature_ms", "inference_ms", "workload_execution_ms", "window_elapsed_ms", "valid",
    ]
    size_counts: Counter[int] = Counter()
    action_admissible_counts: Counter[str] = Counter()
    fallback_count = 0
    observed = 0
    max_temperature = start_temperature
    min_frequency = math.inf
    status = "RUNNING"
    failure_reason = ""
    start_utc = utc_now()
    monitor = PowerMonitor(power_path, float(config["fault_monitor_interval_seconds"]))
    try:
        with window_path.open("x", newline="", encoding="utf-8", buffering=1) as window_stream, \
                progress_path.open("x", newline="", encoding="utf-8", buffering=1) as progress_stream, \
                monitor, BackgroundLoad(background):
            writer = csv.DictWriter(window_stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            progress_writer = csv.writer(progress_stream, lineterminator="\n")
            progress_writer.writerow(["seed", "windows_expected", "windows_observed", "end_utc"])
            time.sleep(float(config["background_stabilization_seconds"]))
            for seed in config["diagnostic_seeds"]:
                order = np.random.default_rng(int(seed)).permutation(len(features))
                cursor = 0
                seed_observed = 0
                for window_id in range(int(config["windows_per_seed"])):
                    if monitor.failure_reason:
                        raise RuntimeError(f"fault monitor failure: {monitor.failure_reason}")
                    if monitor.current_fault.is_set():
                        raise RuntimeError("current hardware fault observed by independent monitor")
                    mono_start = time.perf_counter_ns()
                    T_start = temperature_c()
                    throttle_start, throttle_start_value = throttled()
                    frequency_start = cpu_frequencies_mhz()
                    if throttle_start_value & 0xF:
                        raise RuntimeError(f"current hardware fault: {throttle_start}")

                    latency_hat = {
                        action: latency[(action, f"n{n_records}", str(candidate["background"]))]
                        for action in actions
                    }
                    latency_upper = {
                        action: latency_hat[action]
                        + float(margins["epsilon_tau_ms_by_action"][action])
                        + float(margins["fixed_latency_buffer_ms"])
                        for action in actions
                    }
                    temperature_upper = {
                        action: guarded_temperature(
                            thermal, margins, action, T_start, n_records,
                            float(background["duty_cycle"]), latency_hat[action],
                        )
                        for action in actions
                    }
                    allowed = [
                        action for action in actions
                        if latency_upper[action] <= float(config["constraints"]["latency_cap_ms"])
                        and temperature_upper[action] <= float(config["constraints"]["thermal_cap_c"])
                    ]

                    positions = np.arange(cursor, cursor + n_records) % len(order)
                    indices = order[positions]
                    cursor = (cursor + n_records) % len(order)
                    selected_raw = [raw_rows[int(index)] for index in indices]
                    feature_start = time.perf_counter_ns()
                    transformed = preprocessor.transform(selected_raw)
                    feature_ms = (time.perf_counter_ns() - feature_start) / 1e6
                    if not np.array_equal(transformed, features[indices]):
                        raise RuntimeError("portable preprocessing parity failure")
                    probe_action = actions[window_id % len(actions)]
                    inference_start = time.perf_counter_ns()
                    probabilities = models[probe_action].predict_proba(transformed)
                    inference_ms = (time.perf_counter_ns() - inference_start) / 1e6
                    if not np.isfinite(probabilities).all():
                        raise RuntimeError("non-finite diagnostic inference output")
                    execution_ms = (time.perf_counter_ns() - mono_start) / 1e6

                    T_end = temperature_c()
                    throttle_end, throttle_end_value = throttled()
                    frequency_end = cpu_frequencies_mhz()
                    minimum_frequency = min(frequency_start + frequency_end)
                    maximum_temperature = max(T_start, T_end)
                    if throttle_end_value & 0xF:
                        raise RuntimeError(f"current hardware fault after diagnostic action: {throttle_end}")
                    if minimum_frequency < float(config["minimum_cpu_frequency_mhz"]):
                        raise RuntimeError("CPU frequency below frozen minimum")
                    if maximum_temperature >= float(config["constraints"]["emergency_stop_c"]):
                        raise RuntimeError("emergency temperature stop")
                    remaining = float(config["decision_window_ms"]) / 1000.0 - (time.perf_counter_ns() - mono_start) / 1e9
                    if remaining > 0:
                        time.sleep(remaining)
                    elapsed_ms = (time.perf_counter_ns() - mono_start) / 1e6

                    writer.writerow({
                        "candidate_index": candidate["candidate_index"], "seed": seed,
                        "window_id": window_id, "n_records": n_records,
                        "background": candidate["background"], "probe_action": probe_action,
                        "T_start_C": f"{T_start:.3f}", "T_end_C": f"{T_end:.3f}",
                        "safe_set_size": len(allowed), "safe_set_mask": "|".join(allowed),
                        "fallback_required": int(not allowed),
                        "predicted_latency_by_action_ms": json.dumps(latency_hat, sort_keys=True, separators=(",", ":")),
                        "guarded_latency_by_action_ms": json.dumps(latency_upper, sort_keys=True, separators=(",", ":")),
                        "latency_headroom_by_action_ms": json.dumps({action: float(config["constraints"]["latency_cap_ms"]) - latency_upper[action] for action in actions}, sort_keys=True, separators=(",", ":")),
                        "guarded_temperature_by_action_C": json.dumps(temperature_upper, sort_keys=True, separators=(",", ":")),
                        "thermal_headroom_by_action_C": json.dumps({action: float(config["constraints"]["thermal_cap_c"]) - temperature_upper[action] for action in actions}, sort_keys=True, separators=(",", ":")),
                        "cpu_frequency_min_mhz": f"{minimum_frequency:.3f}",
                        "throttled_start": throttle_start, "throttled_end": throttle_end,
                        "feature_ms": f"{feature_ms:.6f}", "inference_ms": f"{inference_ms:.6f}",
                        "workload_execution_ms": f"{execution_ms:.6f}",
                        "window_elapsed_ms": f"{elapsed_ms:.6f}", "valid": 1,
                    })
                    size_counts[len(allowed)] += 1
                    action_admissible_counts.update(allowed)
                    fallback_count += int(not allowed)
                    observed += 1
                    seed_observed += 1
                    max_temperature = max(max_temperature, maximum_temperature)
                    min_frequency = min(min_frequency, minimum_frequency)
                progress_writer.writerow([seed, config["windows_per_seed"], seed_observed, utc_now()])
            if monitor.failure_reason:
                raise RuntimeError(f"fault monitor failure: {monitor.failure_reason}")
            if monitor.current_fault.is_set():
                raise RuntimeError("current hardware fault observed by independent monitor")
        final_throttle, final_value = throttled()
        if final_value & 0xF:
            raise RuntimeError(f"current hardware fault at candidate end: {final_throttle}")
        status = "BINDING_STRESS_CANDIDATE_VALID" if formal_diagnostic else "DEVELOPMENT_BINDING_STRESS_CODE_PATH_PASS"
    except BaseException as exc:
        status = "BINDING_STRESS_CANDIDATE_INVALID" if formal_diagnostic else "DEVELOPMENT_BINDING_STRESS_CODE_PATH_INVALID"
        failure_reason = f"{type(exc).__name__}: {exc}"

    total_expected = len(config["diagnostic_seeds"]) * int(config["windows_per_seed"])
    binding_fraction = size_counts[0] / total_expected + size_counts[1] / total_expected + size_counts[2] / total_expected + size_counts[3] / total_expected
    at_least_two_fraction = sum(count for size, count in size_counts.items() if size >= 2) / total_expected
    fallback_rate = fallback_count / total_expected
    distinct_actions = [action for action in actions if action_admissible_counts[action] > 0]
    qualified = (
        status == "BINDING_STRESS_CANDIDATE_VALID"
        and observed == total_expected
        and binding_fraction >= float(config["qualification"]["minimum_binding_fraction"])
        and at_least_two_fraction >= float(config["qualification"]["minimum_at_least_two_safe_fraction"])
        and len(distinct_actions) >= int(config["qualification"]["minimum_distinct_admissible_actions"])
        and fallback_rate <= float(config["qualification"]["maximum_fallback_rate"])
    )
    manifest = {
        "status": status, "failure_reason": failure_reason, "formal_diagnostic": formal_diagnostic,
        "candidate_index": candidate["candidate_index"], "n_records": n_records,
        "background": candidate["background"], "start_utc": start_utc, "end_utc": utc_now(),
        "windows_expected": total_expected, "windows_observed": observed,
        "safe_set_size_counts": {str(size): size_counts[size] for size in range(len(actions) + 1)},
        "binding_fraction": binding_fraction, "at_least_two_safe_fraction": at_least_two_fraction,
        "fallback_rate": fallback_rate, "distinct_admissible_actions": distinct_actions,
        "maximum_temperature_C": max_temperature, "minimum_cpu_frequency_mhz": min_frequency,
        "qualified": qualified,
        "selection_inputs": "guard admissibility, safe-set cardinality, fallback requirement, and hardware health only",
        "excluded_from_selection": "model probabilities, labels, confusion counts, rewards, F1, policy outcomes, and realised execution latency",
        "artifact_sha256": {
            path.name: digest(path) for path in (idle_path, window_path, progress_path, power_path) if path.exists()
        },
    }
    (output / "CANDIDATE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if status.endswith("INVALID"):
        raise RuntimeError(failure_reason)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--development-smoke-windows", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    config = json.loads(args.config.read_text())
    if config["status"] != "FROZEN_BEFORE_BINDING_STRESS_SWEEP":
        raise RuntimeError("Protocol V3 config is not frozen")
    if socket.gethostname() != config["device_id"]:
        raise RuntimeError("wrong device")
    formal_diagnostic = args.development_smoke_windows is None
    if formal_diagnostic and len(args.git_commit) != 40:
        raise RuntimeError("full source commit is required")
    if not formal_diagnostic and args.git_commit != "UNCOMMITTED_DEVELOPMENT_SMOKE":
        raise RuntimeError("development smoke must use its explicit non-formal source marker")
    if set(governors()) != {config["required_governor"]}:
        raise RuntimeError("governor mismatch")
    validate_inputs(config)
    if not formal_diagnostic:
        if args.development_smoke_windows < 1:
            raise RuntimeError("development smoke window count must be positive")
        config["diagnostic_seeds"] = config["diagnostic_seeds"][:1]
        config["windows_per_seed"] = args.development_smoke_windows
        config["candidate_grid"] = config["candidate_grid"][:1]
    args.output.mkdir(parents=True)

    raw_path = ROOT / config["inputs"]["scheduler_raw"]["path"]
    with raw_path.open(newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream))
    features = np.load(ROOT / config["inputs"]["scheduler_features"]["path"], allow_pickle=False)
    preprocessor = PortablePreprocessor(ROOT / config["inputs"]["preprocessor"]["path"])
    models = {
        action: NumpyActionModel(ROOT / config["inputs"][f"model_{action}"]["path"])
        for action in config["actions"]
    }
    for model in models.values():
        model.predict_proba(features[:16])
    latency = {
        (row["action"], row["workload_bin"], row["background"]): float(row["tau_hat_ms"])
        for row in csv.DictReader((ROOT / config["inputs"]["latency_predictor"]["path"]).open())
    }
    thermal = json.loads((ROOT / config["inputs"]["thermal_predictor"]["path"]).read_text())
    margins = json.loads((ROOT / config["inputs"]["guard_margins"]["path"]).read_text())

    progress_path = args.output / "SWEEP_PROGRESS.csv"
    summaries = []
    selected = None
    try:
        with progress_path.open("x", newline="", encoding="utf-8", buffering=1) as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["candidate_index", "n_records", "background", "status", "qualified", "output"])
            for candidate in config["candidate_grid"]:
                candidate_output = args.output / f"candidate_{int(candidate['candidate_index']):02d}_n{candidate['n_records']}_{candidate['background']}"
                summary = run_candidate(
                    config, candidate, candidate_output, raw_rows, features,
                    preprocessor, models, latency, thermal, margins, formal_diagnostic,
                )
                summaries.append(summary)
                writer.writerow([
                    candidate["candidate_index"], candidate["n_records"], candidate["background"],
                    summary["status"], int(summary["qualified"]), candidate_output.name,
                ])
                if formal_diagnostic and summary["qualified"]:
                    selected = summary
                    break
    except BaseException as exc:
        stop = {
            "status": "BINDING_STRESS_SWEEP_STOPPED_INVALID",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "git_commit": args.git_commit,
            "config_sha256": digest(args.config),
            "candidates_completed": len(summaries),
        }
        (args.output / "SWEEP_STOP.json").write_text(json.dumps(stop, indent=2) + "\n")
        raise

    result = {
        "status": (
            "BINDING_STRESS_POINT_SELECTED" if selected
            else "NO_BINDING_STRESS_POINT_IN_FROZEN_GRID" if formal_diagnostic
            else "DEVELOPMENT_BINDING_STRESS_CODE_PATH_PASS"
        ),
        "git_commit": args.git_commit,
        "config_sha256": digest(args.config),
        "candidates_evaluated": len(summaries),
        "selected_candidate": None if selected is None else {
            "candidate_index": selected["candidate_index"],
            "n_records": selected["n_records"],
            "background": selected["background"],
        },
        "candidate_manifest_sha256": {
            f"candidate_{int(row['candidate_index']):02d}_n{row['n_records']}_{row['background']}":
                digest(args.output / f"candidate_{int(row['candidate_index']):02d}_n{row['n_records']}_{row['background']}" / "CANDIDATE_MANIFEST.json")
            for row in summaries
        },
        "completed_utc": utc_now(),
    }
    (args.output / "SWEEP_SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
