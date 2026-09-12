#!/usr/bin/env python3
"""Evaluate calibration-only guard feasibility without coverage or policy data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from device_calibration.fit_guard_predictors import thermal_design  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")

    config = json.loads(args.config.read_text())
    if config["status"] != "FROZEN_BEFORE_FEASIBILITY_EVALUATION":
        raise RuntimeError("feasibility configuration is not frozen")
    if config["coverage_access"] != "forbidden" or config["policy_outcomes_access"] != "forbidden":
        raise RuntimeError("forbidden-data rule is not frozen")

    paths: dict[str, Path] = {}
    for key, value in config["inputs"].items():
        if key.endswith("_sha256"):
            continue
        path = ROOT / value
        if digest(path) != config["inputs"][f"{key}_sha256"]:
            raise RuntimeError(f"input hash mismatch: {key}")
        paths[key] = path

    fit_rows = read_csv(paths["fit_raw"])
    latency_rows = read_csv(paths["latency_predictor"])
    validation_rows = read_csv(paths["validation_metrics"])
    margins = json.loads(paths["guard_margins"].read_text())
    thermal = json.loads(paths["thermal_predictor"].read_text())

    f1 = {row["action"]: float(row["f1"]) for row in validation_rows}
    action_names = config["action_name_map"]
    light = config["light_action"]
    light_f1 = f1[action_names[light]]
    if any(f1[action_names[action]] <= light_f1 for action in config["high_utility_actions"]):
        raise RuntimeError("high-utility action rule does not reproduce from validation F1")

    lookup = {
        (row["action"], row["workload_bin"], row["background"]): float(row["tau_hat_ms"])
        for row in latency_rows
    }
    median_latency = {
        action: float(np.median([value for (candidate, _, _), value in lookup.items() if candidate == action]))
        for action in action_names
    }
    if min(median_latency, key=median_latency.get) != light:
        raise RuntimeError("light-action rule does not reproduce from V2 fit predictor")

    mean = np.asarray(thermal["continuous_mean"], dtype=np.float64)
    scale = np.asarray(thermal["continuous_scale"], dtype=np.float64)
    coefficients = np.asarray(thermal["coefficients"], dtype=np.float64)
    thermal_predictions = thermal_design(fit_rows, mean, scale) @ coefficients
    thermal_upper_by_cell: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    fixed_thermal = float(margins["fixed_thermal_buffer_c"])
    epsilon_thermal = float(margins["epsilon_T_C_shared"])
    for row, prediction in zip(fit_rows, thermal_predictions):
        key = row["action"], row["workload_bin"], row["background"]
        thermal_upper_by_cell[key].append(float(prediction) + epsilon_thermal + fixed_thermal)

    low, high = config["selection_criterion"]["high_utility_latency_upper_ms_inclusive"]
    light_limit = float(config["selection_criterion"]["light_action_latency_upper_ms_strictly_below"])
    thermal_limit = float(config["selection_criterion"]["thermal_upper_c_at_or_below"])
    fixed_latency = float(margins["fixed_latency_buffer_ms"])
    evaluated = []
    selected = None
    for workload in config["ordered_candidate_grid"]["workload_records_per_window"]:
        workload_bin = f"n{workload}"
        for background in config["ordered_candidate_grid"]["backgrounds"]:
            action_records = []
            for action in action_names:
                key = action, workload_bin, background
                latency_upper = lookup[key] + float(margins["epsilon_tau_ms_by_action"][action]) + fixed_latency
                thermal_upper = max(thermal_upper_by_cell[key])
                action_records.append({
                    "action": action,
                    "validation_f1": f1[action_names[action]],
                    "latency_upper_ms": latency_upper,
                    "thermal_upper_c_max_fit_state": thermal_upper,
                    "is_light": action == light,
                    "is_high_utility": action in config["high_utility_actions"],
                    "in_near_cap_band": low <= latency_upper <= high,
                    "thermal_feasible": thermal_upper <= thermal_limit,
                })
            light_record = next(record for record in action_records if record["is_light"])
            qualifying = [
                record["action"] for record in action_records
                if record["is_high_utility"] and record["in_near_cap_band"] and record["thermal_feasible"]
            ]
            passed = (
                bool(qualifying)
                and light_record["latency_upper_ms"] < light_limit
                and light_record["thermal_feasible"]
            )
            candidate = {
                "workload_records_per_window": workload,
                "background": background,
                "passed": passed,
                "qualifying_high_utility_actions": qualifying,
                "actions": action_records,
            }
            evaluated.append(candidate)
            if selected is None and passed:
                selected = candidate

    args.output.mkdir(parents=True)
    cells_path = args.output / "feasibility_cells.csv"
    with cells_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([
            "candidate_index", "n_records", "background", "candidate_passed", "action",
            "validation_f1", "is_light", "is_high_utility", "latency_upper_ms",
            "thermal_upper_c_max_fit_state", "in_near_cap_band", "thermal_feasible",
        ])
        for index, candidate in enumerate(evaluated):
            for record in candidate["actions"]:
                writer.writerow([
                    index, candidate["workload_records_per_window"], candidate["background"],
                    int(candidate["passed"]), record["action"], f"{record['validation_f1']:.9f}",
                    int(record["is_light"]), int(record["is_high_utility"]),
                    f"{record['latency_upper_ms']:.9f}",
                    f"{record['thermal_upper_c_max_fit_state']:.9f}",
                    int(record["in_near_cap_band"]), int(record["thermal_feasible"]),
                ])

    summary = {
        "status": "CALIBRATION_ONLY_FEASIBILITY_REPORTED",
        "protocol_version": config["protocol_version"],
        "config_sha256": digest(args.config),
        "input_sha256": {key: digest(path) for key, path in paths.items()},
        "light_action": light,
        "high_utility_actions": config["high_utility_actions"],
        "candidates_evaluated": len(evaluated),
        "selected_candidate": None if selected is None else {
            "workload_records_per_window": selected["workload_records_per_window"],
            "background": selected["background"],
            "qualifying_high_utility_actions": selected["qualifying_high_utility_actions"],
        },
        "coverage_access": "not used",
        "policy_outcomes_access": "not used",
        "artifact_sha256": {cells_path.name: digest(cells_path)},
    }
    (args.output / "feasibility_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
