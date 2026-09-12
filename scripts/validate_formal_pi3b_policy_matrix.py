#!/usr/bin/env python3
"""Independently validate and summarize the frozen Pi 3B+ policy matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "online/pi3b/formal/formal_pi3b_policy_matrix_20260909T142400Z"
CONFIG_PATH = ROOT / "configs/pi3b_policy_experiment.json"
MATRIX_PATH = ROOT / "configs/pi3b_policy_experiment_matrix.csv"
INPUT_MANIFEST_PATH = ROOT / ".formal_inputs/pi3b_official_test_v1/OFFICIAL_TEST_INPUT_MANIFEST.json"
VALIDATION_PATH = CAMPAIGN / "INDEPENDENT_VALIDATION.json"
AGGREGATE_PATH = CAMPAIGN / "POLICY_REGIME_AGGREGATE.csv"
FROZEN_COMMIT = "50b2539a12a63059645537aed5f807c160f3af4b"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(left: float, right: float, tolerance: float = 2e-6) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


config = json.loads(CONFIG_PATH.read_text())
input_manifest = json.loads(INPUT_MANIFEST_PATH.read_text())
matrix_rows = list(csv.DictReader(MATRIX_PATH.open(newline="", encoding="utf-8")))
progress_path = CAMPAIGN / "CAMPAIGN_PROGRESS.csv"
progress_rows = list(csv.DictReader(progress_path.open(newline="", encoding="utf-8")))
complete_path = CAMPAIGN / "CAMPAIGN_COMPLETE.json"
complete = json.loads(complete_path.read_text())

require(digest(MATRIX_PATH) == config["execution_order"]["expected_matrix_sha256"], "matrix hash mismatch")
require(digest(CONFIG_PATH) == input_manifest["policy_config_sha256"], "policy/input hash mismatch")
require(len(matrix_rows) == len(progress_rows) == 160, "matrix/progress row count is not 160")
require(complete == {"status": "FORMAL_PI3B_POLICY_MATRIX_COMPLETE", "runs": 160, "completed_utc": progress_rows[-1]["end_utc"]}, "completion marker mismatch")
require(not (CAMPAIGN / "CAMPAIGN_STOP.json").exists(), "campaign stop marker exists")

expected_combinations = Counter(
    (str(seed), regime, policy)
    for seed in config["evaluation_seeds"]
    for regime in config["regimes"]
    for policy in config["policies"]
)
observed_combinations: Counter[tuple[str, str, str]] = Counter()
run_dirs = sorted(path for path in CAMPAIGN.iterdir() if path.is_dir() and path.name[:3].isdigit())
require(len(run_dirs) == 160, "formal run-directory count is not 160")

aggregate: dict[tuple[str, str], dict[str, object]] = defaultdict(lambda: {
    "runs": 0, "windows": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
    "latency_violations": 0, "thermal_violations": 0, "any_violations": 0,
    "fallbacks": 0, "reward_sum": 0.0, "max_end_to_end_ms": 0.0,
    "max_temperature_C": -math.inf, "min_cpu_frequency_mhz": math.inf,
    "safe_set_sizes": Counter(), "actions": Counter(),
})
canonical_risk_by_seed: dict[str, list[str]] = {}
shared_guard_rows: dict[tuple[str, str, str], list[tuple[str, ...]]] = {}
run_manifest_hashes: dict[str, str] = {}
total_windows = 0
total_idle_samples = 0
total_fault_samples = 0
global_fault_max_temperature = -math.inf
global_max_temperature = -math.inf
global_min_frequency = math.inf
global_max_end_to_end = 0.0

for index, (planned, progress, run_dir) in enumerate(zip(matrix_rows, progress_rows, run_dirs), start=1):
    expected_output = f"{index:03d}_{planned['planned_run_id']}"
    require(int(planned["global_order"]) == index, f"matrix order mismatch at {index}")
    require(int(progress["global_order"]) == index, f"progress order mismatch at {index}")
    require(progress["run_id"] == planned["planned_run_id"], f"run id mismatch at {index}")
    require(progress["status"] == "FORMAL_POLICY_RUN_PASS" and not progress["failure_reason"], f"non-pass progress at {index}")
    require(progress["output"] == expected_output == run_dir.name, f"output-directory mismatch at {index}")
    require(index == 1 or progress_rows[index - 2]["end_utc"] == progress["start_utc"], f"campaign timeline gap at {index}")

    manifest_path = run_dir / "RUN_MANIFEST.json"
    require({path.name for path in run_dir.iterdir()} == {"RUN_MANIFEST.json", "windows.csv", "idle_baseline.csv", "fault_telemetry_1hz.csv"}, f"unexpected run artifacts: {run_dir.name}")
    manifest = json.loads(manifest_path.read_text())
    run_manifest_hashes[run_dir.name] = digest(manifest_path)
    require(manifest["status"] == "FORMAL_POLICY_RUN_PASS" and manifest["valid"] is True and manifest["formal"] is True, f"manifest non-pass: {run_dir.name}")
    require(manifest["run_id"] == planned["planned_run_id"], f"manifest run id mismatch: {run_dir.name}")
    require(manifest["device_id"] == manifest["hostname"] == config["device_id"], f"device mismatch: {run_dir.name}")
    require(manifest["policy"] == planned["policy"] and manifest["regime"] == planned["regime"] and manifest["seed"] == int(planned["seed"]), f"manifest matrix mismatch: {run_dir.name}")
    require(manifest["git_commit"] == FROZEN_COMMIT, f"source commit mismatch: {run_dir.name}")
    require(manifest["config_sha256"] == digest(CONFIG_PATH), f"config hash mismatch: {run_dir.name}")
    require(manifest["model_manifest_sha256"] == digest(ROOT / "models/MODEL_SHA256.json"), f"model hash mismatch: {run_dir.name}")
    require(manifest["q_manifest_sha256"] == digest(ROOT / "q_training/Q_TABLE_MANIFEST.json"), f"Q hash mismatch: {run_dir.name}")
    require(manifest["input_manifest_sha256"] == digest(INPUT_MANIFEST_PATH), f"input hash mismatch: {run_dir.name}")
    require(manifest["windows_expected"] == manifest["windows_observed"] == 600, f"manifest window count mismatch: {run_dir.name}")

    for artifact, expected_hash in manifest["artifact_sha256"].items():
        require(digest(run_dir / artifact) == expected_hash, f"artifact hash mismatch: {run_dir.name}/{artifact}")

    idle_rows = list(csv.DictReader((run_dir / "idle_baseline.csv").open(newline="", encoding="utf-8")))
    require(len(idle_rows) == config["start_and_cooldown"]["idle_samples"] == 30, f"idle sample count mismatch: {run_dir.name}")
    idle_temperatures = [float(row["temperature_C"]) for row in idle_rows]
    require(all(int(row["throttled"], 16) & 0xF == 0 for row in idle_rows), f"idle throttle fault: {run_dir.name}")
    require(close(statistics.median(idle_temperatures), manifest["idle_baseline_C"], 0.001), f"idle median mismatch: {run_dir.name}")
    require(manifest["start_temperature_C"] <= manifest["idle_baseline_C"] + 2.0, f"cooldown violation: {run_dir.name}")
    total_idle_samples += len(idle_rows)

    fault_rows = list(csv.DictReader((run_dir / "fault_telemetry_1hz.csv").open(newline="", encoding="utf-8")))
    require(len(fault_rows) == manifest["fault_monitor_samples"], f"fault monitor count mismatch: {run_dir.name}")
    require(all(int(row["throttled"], 16) & 0xF == 0 for row in fault_rows), f"fault monitor throttle bit: {run_dir.name}")
    require(all(float(row["temperature_C"]) < config["constraints"]["emergency_stop_c"] for row in fault_rows), f"fault monitor temperature stop: {run_dir.name}")
    global_fault_max_temperature = max(global_fault_max_temperature, *(float(row["temperature_C"]) for row in fault_rows))
    total_fault_samples += len(fault_rows)

    with (run_dir / "windows.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == config["required_window_fields"], f"window schema mismatch: {run_dir.name}")
        rows = list(reader)
    require(len(rows) == 600, f"window row count mismatch: {run_dir.name}")

    key = (planned["policy"], planned["regime"])
    stats = aggregate[key]
    stats["runs"] = int(stats["runs"]) + 1
    risk_sequence: list[str] = []
    guard_sequence: list[tuple[str, ...]] = []
    observed_max_temperature = -math.inf
    observed_min_frequency = math.inf
    for window_id, row in enumerate(rows):
        require(row["run_id"] == planned["planned_run_id"] and row["device_id"] == config["device_id"], f"window identity mismatch: {run_dir.name}/{window_id}")
        require(row["policy"] == planned["policy"] and row["regime"] == planned["regime"] and row["seed"] == planned["seed"], f"window matrix mismatch: {run_dir.name}/{window_id}")
        require(int(row["window_id"]) == window_id, f"window order mismatch: {run_dir.name}/{window_id}")
        require(row["n_records"] == planned["n_records"] == "1" and row["background"] == planned["background"] == "B0", f"workload mismatch: {run_dir.name}/{window_id}")
        require(row["q_update_ms"] == row["replay_update_ms"] == "", f"evaluation update logged: {run_dir.name}/{window_id}")
        require(row["valid"] == "1" and row["undervoltage"] == "0" and int(row["throttled"], 16) & 0xF == 0, f"invalid/fault window: {run_dir.name}/{window_id}")
        frequency = float(row["cpu_frequency_min_mhz"])
        t_start, t_end = float(row["T_start_C"]), float(row["T_end_C"])
        end_to_end = float(row["end_to_end_ms"])
        require(frequency >= config["start_and_cooldown"]["minimum_frequency_mhz"], f"frequency violation: {run_dir.name}/{window_id}")
        require(max(t_start, t_end) < config["constraints"]["emergency_stop_c"], f"temperature stop: {run_dir.name}/{window_id}")
        latency_violation = int(end_to_end > config["constraints"]["latency_cap_ms"])
        thermal_violation = int(t_end > config["constraints"]["thermal_cap_c"])
        require(int(row["latency_violation"]) == latency_violation and int(row["thermal_violation"]) == thermal_violation and int(row["any_violation"]) == int(latency_violation or thermal_violation), f"violation flag mismatch: {run_dir.name}/{window_id}")

        latency_predictions = json.loads(row["predicted_latency_by_action_ms"])
        latency_headroom = json.loads(row["latency_guard_headroom_by_action_ms"])
        temperature_predictions = json.loads(row["predicted_temperature_by_action_C"])
        thermal_headroom = json.loads(row["thermal_headroom_by_action_C"])
        selected = row["selected_action"]
        recomputed_safe = [action for action in config["action_order"] if latency_headroom[action] >= 0.0 and thermal_headroom[action] >= 0.0]
        require(row["safe_set_mask"] == "|".join(recomputed_safe) and int(row["safe_set_size"]) == len(recomputed_safe), f"safe-set mismatch: {run_dir.name}/{window_id}")
        require(close(float(row["tau_hat_ms"]), latency_predictions[selected]) and close(float(row["latency_slack_ms"]), latency_headroom[selected]), f"latency guard field mismatch: {run_dir.name}/{window_id}")
        require(close(float(row["temp_hat_next_C"]), temperature_predictions[selected]) and close(float(row["thermal_slack_C"]), thermal_headroom[selected]), f"thermal guard field mismatch: {run_dir.name}/{window_id}")
        require(int(row["fallback"]) == int(not recomputed_safe and planned["policy"] in {"Safe-Greedy", "FSSQL-R"}), f"fallback mismatch: {run_dir.name}/{window_id}")

        stats["windows"] = int(stats["windows"]) + 1
        for field in ("tp", "fp", "tn", "fn", "latency_violation", "thermal_violation", "any_violation", "fallback"):
            target = "fallbacks" if field == "fallback" else f"{field}s" if field.endswith("violation") else field
            stats[target] = int(stats[target]) + int(row[field])
        stats["reward_sum"] = float(stats["reward_sum"]) + float(row["reward_total"])
        stats["max_end_to_end_ms"] = max(float(stats["max_end_to_end_ms"]), end_to_end)
        stats["max_temperature_C"] = max(float(stats["max_temperature_C"]), t_start, t_end)
        stats["min_cpu_frequency_mhz"] = min(float(stats["min_cpu_frequency_mhz"]), frequency)
        stats["safe_set_sizes"][row["safe_set_size"]] += 1
        stats["actions"][selected] += 1
        risk_sequence.append(row["p_risk"])
        guard_sequence.append((row["p_risk"], row["safe_set_size"], row["safe_set_mask"], row["predicted_latency_by_action_ms"], row["latency_guard_headroom_by_action_ms"]))
        observed_max_temperature = max(observed_max_temperature, t_start, t_end)
        observed_min_frequency = min(observed_min_frequency, frequency)
        global_max_end_to_end = max(global_max_end_to_end, end_to_end)
    require(close(observed_max_temperature, manifest["maximum_temperature_C"], 0.001), f"manifest maximum temperature mismatch: {run_dir.name}")
    require(close(observed_min_frequency, manifest["minimum_cpu_frequency_mhz"], 0.001), f"manifest minimum frequency mismatch: {run_dir.name}")
    global_max_temperature = max(global_max_temperature, observed_max_temperature)
    global_min_frequency = min(global_min_frequency, observed_min_frequency)
    total_windows += len(rows)
    observed_combinations[(planned["seed"], planned["regime"], planned["policy"])] += 1

    if planned["seed"] not in canonical_risk_by_seed:
        canonical_risk_by_seed[planned["seed"]] = risk_sequence
    else:
        require(risk_sequence == canonical_risk_by_seed[planned["seed"]], f"matched p_risk sequence mismatch: {run_dir.name}")
    if planned["policy"] in {"Safe-Greedy", "FSSQL-R"}:
        shared_guard_rows[(planned["seed"], planned["regime"], planned["policy"])] = guard_sequence

require(observed_combinations == expected_combinations, "8 x 2 x 10 matrix coverage mismatch")
require(total_windows == 96_000, "total window count is not 96,000")
for seed in map(str, config["evaluation_seeds"]):
    for regime in config["regimes"]:
        require(shared_guard_rows[(seed, regime, "Safe-Greedy")] == shared_guard_rows[(seed, regime, "FSSQL-R")], f"shared guard mismatch: seed={seed}, regime={regime}")

aggregate_fields = [
    "policy", "regime", "runs", "windows", "tp", "fp", "tn", "fn", "precision", "recall", "f1",
    "mean_reward", "latency_violations", "thermal_violations", "any_violations", "fallbacks",
    "max_end_to_end_ms", "max_temperature_C", "min_cpu_frequency_mhz", "safe_set_size_counts", "action_counts",
]
aggregate_rows = []
for policy in config["policies"]:
    for regime in config["regimes"]:
        stats = aggregate[(policy, regime)]
        tp, fp, fn = int(stats["tp"]), int(stats["fp"]), int(stats["fn"])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        aggregate_rows.append({
            "policy": policy, "regime": regime, "runs": stats["runs"], "windows": stats["windows"],
            "tp": stats["tp"], "fp": stats["fp"], "tn": stats["tn"], "fn": stats["fn"],
            "precision": f"{precision:.9f}", "recall": f"{recall:.9f}", "f1": f"{f1:.9f}",
            "mean_reward": f"{float(stats['reward_sum']) / int(stats['windows']):.9f}",
            "latency_violations": stats["latency_violations"], "thermal_violations": stats["thermal_violations"],
            "any_violations": stats["any_violations"], "fallbacks": stats["fallbacks"],
            "max_end_to_end_ms": f"{float(stats['max_end_to_end_ms']):.6f}",
            "max_temperature_C": f"{float(stats['max_temperature_C']):.3f}",
            "min_cpu_frequency_mhz": f"{float(stats['min_cpu_frequency_mhz']):.3f}",
            "safe_set_size_counts": json.dumps(dict(sorted(stats["safe_set_sizes"].items())), separators=(",", ":")),
            "action_counts": json.dumps(dict(sorted(stats["actions"].items())), separators=(",", ":")),
        })

total_latency_violations = sum(int(row["latency_violations"]) for row in aggregate_rows)
total_thermal_violations = sum(int(row["thermal_violations"]) for row in aggregate_rows)
total_any_violations = sum(int(row["any_violations"]) for row in aggregate_rows)
total_fallbacks = sum(int(row["fallbacks"]) for row in aggregate_rows)

total_latency_violations = sum(int(stats["latency_violations"]) for stats in aggregate.values())
total_thermal_violations = sum(int(stats["thermal_violations"]) for stats in aggregate.values())
total_any_violations = sum(int(stats["any_violations"]) for stats in aggregate.values())
total_fallbacks = sum(int(stats["fallbacks"]) for stats in aggregate.values())
global_safe_set_sizes = Counter()
for stats in aggregate.values():
    global_safe_set_sizes.update(stats["safe_set_sizes"])

with AGGREGATE_PATH.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=aggregate_fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(aggregate_rows)

validation = {
    "status": "FORMAL_PI3B_POLICY_MATRIX_INDEPENDENT_VALIDATION_PASS",
    "validated_campaign": CAMPAIGN.name,
    "source_commit": FROZEN_COMMIT,
    "campaign_completed_utc": complete["completed_utc"],
    "runs": 160,
    "windows": total_windows,
    "matrix_coverage": "8 policies x 2 regimes x 10 seeds, exactly once",
    "all_run_statuses": "FORMAL_POLICY_RUN_PASS",
    "artifact_hashes_reproduced": 160 * 3,
    "window_schema_fields": len(config["required_window_fields"]),
    "idle_samples": total_idle_samples,
    "fault_monitor_samples": total_fault_samples,
    "current_fault_low_nibble_windows": 0,
    "undervoltage_windows": 0,
    "minimum_cpu_frequency_mhz": global_min_frequency,
    "maximum_temperature_C": global_max_temperature,
    "maximum_fault_monitor_temperature_C": global_fault_max_temperature,
    "maximum_end_to_end_ms": global_max_end_to_end,
    "latency_violations": total_latency_violations,
    "thermal_violations": total_thermal_violations,
    "any_violations": total_any_violations,
    "fallbacks": total_fallbacks,
    "safe_set_size_counts": dict(sorted(global_safe_set_sizes.items())),
    "matched_trace_check": "All 16 policy/regime runs within each seed have identical 600-value p_risk sequences.",
    "shared_guard_check": "Safe-Greedy and FSSQL-R have identical risk, safe-set, latency-prediction, and latency-headroom fields for every matched seed/regime/window.",
    "regime_interpretation": "Natural and near_cap both use the frozen n=1/B0 workload; labels do not establish regime separation.",
    "energy_boundary": "No physical W or J measurement was collected in this campaign.",
    "campaign_progress_sha256": digest(progress_path),
    "campaign_complete_sha256": digest(complete_path),
    "policy_config_sha256": digest(CONFIG_PATH),
    "matrix_sha256": digest(MATRIX_PATH),
    "aggregate_sha256": digest(AGGREGATE_PATH),
    "run_manifest_sha256": run_manifest_hashes,
}
VALIDATION_PATH.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
print(json.dumps({key: validation[key] for key in ("status", "runs", "windows", "minimum_cpu_frequency_mhz", "maximum_temperature_C", "maximum_end_to_end_ms")}, indent=2))
