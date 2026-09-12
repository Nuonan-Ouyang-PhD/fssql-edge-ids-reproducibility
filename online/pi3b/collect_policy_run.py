#!/usr/bin/env python3
"""Collect one Pi-local formal closed-loop policy run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime"))
from action_pool_numpy import NumpyActionModel  # noqa: E402
from preprocessor_numpy import PortablePreprocessor  # noqa: E402
from risk_proxy_numpy import RiskProxy  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def command(arguments: list[str]) -> str:
    result = subprocess.run(arguments, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed: {' '.join(arguments)}: {result.stderr.strip()}")
    return result.stdout.strip()


def temperature_c() -> float:
    return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()) / 1000.0


def throttled() -> tuple[str, int]:
    text = command(["vcgencmd", "get_throttled"])
    return text.split("=")[-1], int(text.split("=")[-1], 16)


def frequencies_mhz() -> list[float]:
    paths = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"))
    if not paths:
        raise RuntimeError("CPU frequency telemetry unavailable")
    return [int(path.read_text().strip()) / 1000.0 for path in paths]


def governors() -> list[str]:
    return [
        path.read_text().strip()
        for path in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor"))
    ]


class FaultMonitor:
    def __init__(self, path: Path):
        self.path = path
        self.stop = threading.Event()
        self.fault = threading.Event()
        self.failure = ""
        self.samples = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            with self.path.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["timestamp_utc", "temperature_C", "throttled"])
                while not self.stop.is_set():
                    text, value = throttled()
                    writer.writerow([utc_now(), f"{temperature_c():.3f}", text])
                    stream.flush()
                    self.samples += 1
                    if value & 0xF:
                        self.fault.set()
                    self.stop.wait(1.0)
        except BaseException as exc:
            self.failure = f"{type(exc).__name__}: {exc}"

    def __enter__(self) -> "FaultMonitor":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join(timeout=4)
        if self.thread.is_alive() and not self.failure:
            self.failure = "fault monitor did not stop"


def state_id(config: dict[str, object], risk: float, temperature: float, energy: float,
             backlog: float, previous_action: int) -> int:
    state = config["state"]
    bins = [
        int(np.digitize(risk, state["risk_bin_upper_edges"])),
        int(np.digitize(temperature, state["temperature_C_upper_edges"])),
        int(np.digitize(energy, state["energy_residual_upper_edges"])),
        int(np.digitize(backlog, state["backlog_upper_edges"])),
        previous_action,
    ]
    result = 0
    for value, cardinality in zip(bins, [4, 4, 4, 4, 5]):
        result = result * cardinality + value
    return result


def thermal_prediction(thermal: dict[str, object], action: str, temperature: float,
                       latency_hat: float, n_records: int = 1,
                       background_duty: float = 0.0) -> float:
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
    return float(design @ np.asarray(thermal["coefficients"], dtype=np.float64))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--development-windows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)
    config = json.loads(args.config.read_text())
    if "runtime" in config:
        for name in ("collector", "campaign_runner"):
            if sha256(ROOT / config["runtime"][name]) != config["runtime"][f"{name}_sha256"]:
                raise RuntimeError(f"runtime hash mismatch: {name}")
        for name, record in config.get("frozen_inputs", {}).items():
            if sha256(ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen input hash mismatch: {name}")
    formal = args.development_windows is None
    expected_windows = int(config["windows_per_run"] if formal else args.development_windows)
    if socket.gethostname() != config["device_id"]:
        raise RuntimeError(f"wrong host: {socket.gethostname()}")
    if args.policy not in config["policies"] or args.regime not in config["regimes"]:
        raise RuntimeError("policy or regime is outside preregistration")
    if args.seed not in config["evaluation_seeds"]:
        raise RuntimeError("seed is outside preregistration")
    if formal and (len(args.git_commit) != 40 or any(c not in "0123456789abcdef" for c in args.git_commit)):
        raise RuntimeError("formal run requires full lowercase Git commit")

    input_manifest_path = args.input / "OFFICIAL_TEST_INPUT_MANIFEST.json"
    input_manifest = json.loads(input_manifest_path.read_text())
    if formal and input_manifest["status"] != "OFFICIAL_TEST_OPENED_AND_MATCHED_TRACES_FROZEN":
        raise RuntimeError("formal run requires official-test input manifest")
    accepted_input_manifest_sha256 = config.get("official_test", {}).get(
        "accepted_input_manifest_sha256"
    )
    if formal and accepted_input_manifest_sha256:
        if sha256(input_manifest_path) != accepted_input_manifest_sha256:
            raise RuntimeError("accepted official-test input manifest hash mismatch")
        if input_manifest["policy_config_sha256"] != config["official_test"]["formal_matrix_v1_config_sha256"]:
            raise RuntimeError("accepted official-test trace lineage mismatch")
    elif formal and input_manifest["policy_config_sha256"] != sha256(args.config):
        raise RuntimeError("official-test input was not frozen for this policy config")
    for name, expected in input_manifest["artifact_sha256"].items():
        if sha256(args.input / name) != expected:
            raise RuntimeError(f"input hash mismatch: {name}")
    q_config_path = ROOT / config["q_learning"]["config"]
    if sha256(q_config_path) != config["q_learning"]["config_sha256"]:
        raise RuntimeError("Q config hash mismatch")
    q_config = json.loads(q_config_path.read_text())
    for name in ("risk_proxy", "latency_predictor", "thermal_predictor", "guard_margins", "validation_metrics", "model_manifest"):
        record = q_config["inputs"][name]
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"runtime input hash mismatch: {name}")
    q_manifest_path = ROOT / "q_training/Q_TABLE_MANIFEST.json"
    q_manifest = json.loads(q_manifest_path.read_text())
    if q_manifest["status"] != "Q_TRAINING_AND_STABILITY_DIAGNOSTIC_PASS":
        raise RuntimeError("Q tables are not frozen")
    for relative, expected in q_manifest["artifact_sha256"].items():
        if sha256(ROOT / "q_training" / relative) != expected:
            raise RuntimeError(f"Q artifact hash mismatch: {relative}")
    model_manifest = json.loads((ROOT / "models/MODEL_SHA256.json").read_text())
    actions = config["action_order"]
    action_names = {
        "fisvdd_revision": "FISVDD", "lucid_revision": "LUCID",
        "tinydl_revision": "TinyDL", "oi_svdd_as_elm_revision": "OI-SVDD+AS-ELM",
    }
    validation_metrics = {
        row["action"]: row
        for row in csv.DictReader((ROOT / "models/VALIDATION_METRICS.csv").open())
    }
    decision_thresholds = {
        action: float(validation_metrics[action_names[action]]["threshold"])
        for action in actions
    }
    for action in actions:
        relative = f"models/artifacts/{action}.npz"
        if sha256(ROOT / relative) != model_manifest["files"][relative]["sha256"]:
            raise RuntimeError(f"model hash mismatch: {action}")
    parity = json.loads((ROOT / "preprocessing/PREPROCESSOR_PORTABLE_PARITY.json").read_text())
    if parity["status"] != "PASS" or sha256(ROOT / parity["artifact"]) != parity["artifact_sha256"]:
        raise RuntimeError("portable preprocessor parity artifact mismatch")

    workload = config["workloads"][args.regime]
    n_records = int(workload["n_records"])
    background = str(workload["background"])
    background_duty = float(workload.get("background_duty", 0.0))
    index_rows = [row for row in csv.DictReader((args.input / "trace_index.csv").open()) if int(row["seed"]) == args.seed]
    if len(index_rows) < expected_windows:
        raise RuntimeError("incomplete matched trace")
    all_raw = list(csv.DictReader((args.input / "trace_raw_features.csv").open()))
    all_features = np.load(args.input / "trace_features.npy", allow_pickle=False)
    all_risk = np.load(args.input / "trace_risk_features.npy", allow_pickle=False)
    all_labels = np.load(args.input / "trace_labels.npy", allow_pickle=False)
    indices = [int(row["array_index"]) for row in index_rows[:expected_windows]]
    batches = [
        [indices[(window_id * n_records + offset) % len(indices)] for offset in range(n_records)]
        for window_id in range(expected_windows)
    ]

    preprocessor = PortablePreprocessor(ROOT / "preprocessing/preprocessor_portable.npz")
    risk_proxy = RiskProxy(ROOT / "risk_proxy/risk_proxy.npz")
    models = {action: NumpyActionModel(ROOT / f"models/artifacts/{action}.npz") for action in actions}
    for model in models.values():
        for _ in range(10):
            model.predict_proba(all_features[batches[0]])

    latency = {
        (row["action"], row["workload_bin"], row["background"]): float(row["tau_hat_ms"])
        for row in csv.DictReader((ROOT / "calibration/pi3b/protocol_v2/frozen_guard_fit_margin_v1/latency_predictor.csv").open())
    }
    latency_hat = np.asarray([latency[action, f"n{n_records}", background] for action in actions])
    margins = json.loads((ROOT / "calibration/pi3b/protocol_v2/frozen_guard_fit_margin_v1/guard_margins.json").read_text())
    thermal = json.loads((ROOT / "calibration/pi3b/protocol_v2/frozen_guard_fit_margin_v1/thermal_predictor.json").read_text())
    latency_upper = latency_hat + np.asarray([margins["epsilon_tau_ms_by_action"][a] for a in actions]) + margins["fixed_latency_buffer_ms"]
    thermal_margin = margins["epsilon_T_C_shared"] + margins["fixed_thermal_buffer_c"]
    quality = q_manifest["validation_action_quality"]
    frozen_q_latency_hat = np.asarray([latency[action, "n1", "B0"] for action in actions])
    energy_cost = 0.001 * frozen_q_latency_hat / frozen_q_latency_hat.max()

    q_values = None
    training_seed = config["q_learning"]["evaluation_seed_to_training_seed"][str(args.seed)]
    if args.policy in ("Unshielded-Q", "FSSQL-R"):
        prefix = "unshielded_q" if args.policy == "Unshielded-Q" else "fssql_r"
        q_values = np.load(ROOT / f"q_training/tables/{prefix}_seed_{training_seed}.npz", allow_pickle=False)["q_values"]

    initial_text, initial_value = throttled()
    if initial_value & 0xF:
        raise RuntimeError(f"current throttle fault before idle baseline: {initial_text}")
    if set(governors()) != {config["start_and_cooldown"]["required_governor"]}:
        raise RuntimeError("governor mismatch")
    idle_rows = []
    idle_samples = config["start_and_cooldown"]["idle_samples"] if formal else 3
    idle_interval = config["start_and_cooldown"]["idle_interval_seconds"] if formal else 0.1
    for sample in range(idle_samples):
        text, value = throttled()
        temp = temperature_c()
        idle_rows.append([sample, utc_now(), f"{temp:.3f}", text, f"{os.getloadavg()[0]:.6f}"])
        if value & 0xF:
            raise RuntimeError(f"current throttle fault in idle baseline: {text}")
        if sample + 1 < idle_samples:
            time.sleep(idle_interval)
    with (args.output / "idle_baseline.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["sample", "timestamp_utc", "temperature_C", "throttled", "load1"])
        writer.writerows(idle_rows)
    idle_median = float(np.median([float(row[2]) for row in idle_rows]))
    start_temperature = temperature_c()
    if start_temperature > idle_median + 2.0:
        raise RuntimeError("start temperature exceeds cooldown rule")

    fields = config["required_window_fields"]
    raw_path = args.output / "windows.csv"
    status = "RUNNING"
    failure = ""
    observed = 0
    start_utc = utc_now()
    previous_action = 4
    energy = 1.0
    backlog = 0.0
    rows = []
    minimum_frequency = math.inf
    maximum_temperature = start_temperature
    monitor = FaultMonitor(args.output / "fault_telemetry_1hz.csv")
    try:
        with monitor:
            for window_id, batch_indices in enumerate(batches):
                if monitor.failure or monitor.fault.is_set():
                    raise RuntimeError(monitor.failure or "current fault observed by independent monitor")
                mono_start = time.perf_counter_ns()
                wall = utc_now()
                step = time.perf_counter_ns()
                T_start = temperature_c()
                text, value = throttled()
                freq_start = frequencies_mhz()
                telemetry_ms = (time.perf_counter_ns() - step) / 1e6
                if value & 0xF:
                    raise RuntimeError(f"current throttle fault: {text}")
                minimum_frequency = min(minimum_frequency, *freq_start)
                if minimum_frequency < config["start_and_cooldown"]["minimum_frequency_mhz"]:
                    raise RuntimeError("CPU frequency below frozen minimum")

                step = time.perf_counter_ns()
                transformed = preprocessor.transform([all_raw[array_index] for array_index in batch_indices])
                feature_ms = (time.perf_counter_ns() - step) / 1e6
                if not np.array_equal(transformed, all_features[batch_indices]):
                    raise RuntimeError("portable preprocessing parity failure")
                step = time.perf_counter_ns()
                p_risk = float(np.mean(risk_proxy.predict(all_risk[batch_indices])))
                state = state_id(q_config, p_risk, T_start, energy, backlog, previous_action)
                state_build_ms = (time.perf_counter_ns() - step) / 1e6
                step = time.perf_counter_ns()
                temp_hat = np.asarray([
                    thermal_prediction(
                        thermal, action, T_start, latency_hat[i], n_records, background_duty
                    )
                    for i, action in enumerate(actions)
                ])
                temp_upper = temp_hat + thermal_margin
                allowed = [i for i in range(len(actions)) if latency_upper[i] <= config["constraints"]["latency_cap_ms"] and temp_upper[i] <= config["constraints"]["thermal_cap_c"]]
                guard_ms = (time.perf_counter_ns() - step) / 1e6

                def immediate(index: int) -> float:
                    action = actions[index]
                    security = p_risk * quality[action]["tpr"] - (1.0 - p_risk) * quality[action]["fpr"]
                    switch = -q_config["reward"]["switch_weight"] * float(previous_action != 4 and previous_action != index)
                    return security - q_config["reward"]["energy_weight"] * float(energy_cost[index] / 0.001) + switch

                step = time.perf_counter_ns()
                fallback = False
                if args.policy == "Static-Heavy":
                    action_index = actions.index(config["action_roles"]["heavy"])
                elif args.policy == "Static-Light":
                    action_index = actions.index(config["action_roles"]["light"])
                elif args.policy == "Round-Robin":
                    action_index = window_id % len(actions)
                elif args.policy == "Greedy-Risk":
                    action_index = max(range(len(actions)), key=lambda i: (immediate(i), -i))
                elif args.policy == "Safe-Greedy":
                    candidates = allowed or [0]
                    fallback = not bool(allowed)
                    action_index = max(candidates, key=lambda i: (immediate(i), -i))
                elif args.policy == "Threshold":
                    threshold = config["threshold_controller"]
                    if T_start >= threshold["T_hi_C"] or backlog >= threshold["q_hi_records"]:
                        action_index = actions.index(threshold["light_action"])
                    elif T_start <= threshold["T_lo_C"] and backlog <= threshold["q_lo_records"] and p_risk >= threshold["p_hi"]:
                        action_index = actions.index(threshold["heavy_action"])
                    else:
                        action_index = previous_action if previous_action != 4 else actions.index(threshold["light_action"])
                elif args.policy == "Unshielded-Q":
                    action_index = max(range(len(actions)), key=lambda i: (q_values[state, i], -i))
                elif args.policy == "FSSQL-R":
                    candidates = allowed or [0]
                    fallback = not bool(allowed)
                    action_index = max(candidates, key=lambda i: (q_values[state, i], -i))
                else:
                    raise RuntimeError("unimplemented policy")
                selection_ms = (time.perf_counter_ns() - step) / 1e6
                step = time.perf_counter_ns()
                selected_action = actions[action_index]
                switched = previous_action != 4 and previous_action != action_index
                selected_model = models[selected_action]
                switch_ms = (time.perf_counter_ns() - step) / 1e6
                step = time.perf_counter_ns()
                probabilities = selected_model.predict_proba(transformed)
                predictions = probabilities >= decision_thresholds[selected_action]
                inference_ms = (time.perf_counter_ns() - step) / 1e6
                mono_end = time.perf_counter_ns()
                T_end = temperature_c()
                freq_end = frequencies_mhz()
                minimum_frequency = min(minimum_frequency, *freq_end)
                maximum_temperature = max(maximum_temperature, T_start, T_end)
                end_to_end_ms = (mono_end - mono_start) / 1e6
                if minimum_frequency < config["start_and_cooldown"]["minimum_frequency_mhz"]:
                    raise RuntimeError("CPU frequency below frozen minimum")
                if maximum_temperature >= config["constraints"]["emergency_stop_c"]:
                    raise RuntimeError("emergency temperature stop")

                labels = all_labels[batch_indices]
                tp = int(np.sum(predictions & (labels == 1)))
                fp = int(np.sum(predictions & (labels == 0)))
                tn = int(np.sum(~predictions & (labels == 0)))
                fn = int(np.sum(~predictions & (labels == 1)))
                q_security = p_risk * quality[selected_action]["tpr"] - (1.0 - p_risk) * quality[selected_action]["fpr"]
                reward_energy = -q_config["reward"]["energy_weight"] * float(energy_cost[action_index] / 0.001)
                reward_backlog = -q_config["reward"]["backlog_weight"] * min(backlog / q_config["reward"]["backlog_normalizer"], 1.0)
                reward_switch = -q_config["reward"]["switch_weight"] * float(switched)
                reward_total = q_security + reward_energy + reward_backlog + reward_switch
                next_energy = max(0.0, energy - float(energy_cost[action_index]))
                next_backlog = max(0.0, backlog + 1.0 - 1.0)
                selected_tau_upper = float(latency_upper[action_index])
                selected_temp_upper = float(temp_upper[action_index])
                record = {
                    "run_id": args.run_id, "device_id": config["device_id"], "policy": args.policy,
                    "regime": args.regime, "seed": args.seed, "window_id": window_id,
                    "wall_timestamp": wall, "mono_start_ns": mono_start, "mono_end_ns": mono_end,
                    "p_risk": f"{p_risk:.9f}", "T_start_C": f"{T_start:.3f}", "T_end_C": f"{T_end:.3f}",
                    "backlog_start": f"{backlog:.6f}", "backlog_end": f"{next_backlog:.6f}",
                    "energy_proxy_start": f"{energy:.9f}",
                    "prev_action": "" if previous_action == 4 else actions[previous_action],
                    "selected_action": selected_action, "safe_set_size": len(allowed),
                    "safe_set_mask": "|".join(actions[i] for i in allowed),
                    "predicted_latency_by_action_ms": json.dumps({a: float(latency_hat[i]) for i, a in enumerate(actions)}, sort_keys=True, separators=(",", ":")),
                    "latency_guard_headroom_by_action_ms": json.dumps({a: config["constraints"]["latency_cap_ms"] - float(latency_upper[i]) for i, a in enumerate(actions)}, sort_keys=True, separators=(",", ":")),
                    "predicted_temperature_by_action_C": json.dumps({a: float(temp_hat[i]) for i, a in enumerate(actions)}, sort_keys=True, separators=(",", ":")),
                    "thermal_headroom_by_action_C": json.dumps({a: config["constraints"]["thermal_cap_c"] - float(temp_upper[i]) for i, a in enumerate(actions)}, sort_keys=True, separators=(",", ":")),
                    "tau_hat_ms": f"{latency_hat[action_index]:.9f}", "tau_upper_ms": f"{selected_tau_upper:.9f}",
                    "temp_hat_next_C": f"{temp_hat[action_index]:.9f}", "temp_upper_C": f"{selected_temp_upper:.9f}",
                    "latency_slack_ms": f"{config['constraints']['latency_cap_ms'] - selected_tau_upper:.9f}",
                    "thermal_slack_C": f"{config['constraints']['thermal_cap_c'] - selected_temp_upper:.9f}",
                    "fallback": int(fallback), "telemetry_ms": f"{telemetry_ms:.6f}",
                    "state_build_ms": f"{state_build_ms:.6f}", "guard_ms": f"{guard_ms:.6f}",
                    "selection_ms": f"{selection_ms:.6f}", "switch_ms": f"{switch_ms:.6f}",
                    "feature_ms": f"{feature_ms:.6f}", "inference_ms": f"{inference_ms:.6f}",
                    "q_update_ms": "", "replay_update_ms": "", "logging_ms": "",
                    "total_controller_ms": f"{telemetry_ms + state_build_ms + guard_ms + selection_ms + switch_ms:.6f}",
                    "end_to_end_ms": f"{end_to_end_ms:.6f}", "throttled": text,
                    "undervoltage": int(bool(value & 1)), "cpu_frequency_min_mhz": f"{min(freq_start + freq_end):.3f}",
                    "n_records": n_records, "background": background,
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    "reward_security": f"{q_security:.9f}", "reward_energy": f"{reward_energy:.9f}",
                    "reward_backlog": f"{reward_backlog:.9f}", "reward_switch": f"{reward_switch:.9f}",
                    "reward_total": f"{reward_total:.9f}",
                    "latency_violation": int(end_to_end_ms > config["constraints"]["latency_cap_ms"]),
                    "thermal_violation": int(T_end > config["constraints"]["thermal_cap_c"]),
                    "any_violation": int(end_to_end_ms > config["constraints"]["latency_cap_ms"] or T_end > config["constraints"]["thermal_cap_c"]),
                    "valid": 1,
                }
                if "batch_array_indices" in fields:
                    record["batch_array_indices"] = json.dumps(batch_indices, separators=(",", ":"))
                step = time.perf_counter_ns()
                sink = io.StringIO()
                probe = csv.DictWriter(sink, fieldnames=fields, lineterminator="\n")
                probe.writerow(record)
                logging_ms = (time.perf_counter_ns() - step) / 1e6
                record["logging_ms"] = f"{logging_ms:.6f}"
                rows.append(record)
                observed += 1
                previous_action, energy, backlog = action_index, next_energy, next_backlog
                remaining = config["decision_window_ms"] / 1000.0 - (time.perf_counter_ns() - mono_start) / 1e9
                if remaining > 0:
                    time.sleep(remaining)
        if monitor.failure or monitor.fault.is_set():
            raise RuntimeError(monitor.failure or "current fault observed by independent monitor")
        final_text, final_value = throttled()
        if final_value & 0xF:
            raise RuntimeError(f"current throttle fault at end: {final_text}")
        status = "FORMAL_POLICY_RUN_PASS" if formal else "DEVELOPMENT_POLICY_CODE_PATH_PASS"
    except BaseException as exc:
        status = "FORMAL_POLICY_RUN_INVALID" if formal else "DEVELOPMENT_POLICY_CODE_PATH_INVALID"
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if rows:
            with raw_path.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        end_utc = utc_now()
        artifacts = [path for path in (raw_path, args.output / "idle_baseline.csv", args.output / "fault_telemetry_1hz.csv") if path.exists()]
        manifest = {
            "run_id": args.run_id, "status": status, "failure_reason": failure, "formal": formal,
            "device_id": config["device_id"], "hostname": platform.node(), "platform": platform.platform(),
            "python": platform.python_version(), "numpy": np.__version__, "policy": args.policy,
            "regime": args.regime, "seed": args.seed, "training_seed": training_seed,
            "git_commit": args.git_commit, "config_sha256": sha256(args.config),
            "model_manifest_sha256": sha256(ROOT / "models/MODEL_SHA256.json"),
            "q_manifest_sha256": sha256(q_manifest_path), "input_manifest_sha256": sha256(input_manifest_path),
            "start_time": start_utc, "end_time": end_utc, "windows_expected": expected_windows,
            "windows_observed": observed, "records_evaluated": observed * n_records,
            "valid": status.endswith("PASS"),
            "idle_baseline_C": idle_median, "start_temperature_C": start_temperature,
            "maximum_temperature_C": maximum_temperature, "minimum_cpu_frequency_mhz": minimum_frequency,
            "fault_monitor_samples": monitor.samples,
            "artifact_sha256": {path.name: sha256(path) for path in artifacts},
        }
        (args.output / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
