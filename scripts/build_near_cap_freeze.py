#!/usr/bin/env python3
"""Build the outcome-blind Pi 3B+ near-cap workload freeze evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def thermal_upper(
    thermal: dict[str, object], margins: dict[str, object], action: str,
    temperature: float, n_records: int, background_duty: float, latency_hat: float,
) -> float:
    numeric = np.asarray([
        temperature, math.log2(n_records), background_duty, latency_hat,
    ], dtype=np.float64)
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


def canonical_seed_trace(rows: list[dict[str, object]]) -> bytes:
    lines = ["window_id,scheduler_sequence,source_row_index,feature_group_hex"]
    lines.extend(
        f"{row['window_id']},{row['scheduler_sequence']},{row['source_row_index']},{row['feature_group_hex']}"
        for row in rows
    )
    return ("\n".join(lines) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")

    config = json.loads(args.config.read_text())
    if config["status"] != "FROZEN_BEFORE_FORMAL_POLICY_EVALUATION":
        raise RuntimeError("near-cap workload is not frozen")
    if config["evidence_boundary"]["official_test_access"] != "forbidden":
        raise RuntimeError("official test must remain forbidden")
    if config["evidence_boundary"]["policy_outcomes_access"] != "forbidden":
        raise RuntimeError("policy outcomes must remain forbidden")

    generator = config["generator_script"]
    if sha256(ROOT / generator["path"]) != generator["sha256"]:
        raise RuntimeError("generator script hash mismatch")

    paths: dict[str, Path] = {}
    for name, record in config["inputs"].items():
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"input hash mismatch: {name}")
        paths[name] = path

    feasibility = json.loads(paths["feasibility_summary"].read_text())
    selected = feasibility["selected_candidate"]
    workload = config["workload"]
    if selected["workload_records_per_window"] != workload["n_records"]:
        raise RuntimeError("selected feasibility workload does not match freeze")
    if selected["background"] != workload["background"]["id"]:
        raise RuntimeError("selected feasibility background does not match freeze")

    source_rows = read_csv(paths["source_index"])
    if len(source_rows) != config["generator"]["source_rows"]:
        raise RuntimeError("scheduler-train source row count mismatch")
    fit_rows = [
        row for row in read_csv(paths["fit_raw"])
        if row["workload_bin"] == f"n{workload['n_records']}"
        and row["background"] == workload["background"]["id"]
    ]
    if not fit_rows:
        raise RuntimeError("no calibration fit states for frozen candidate")
    fit_rows.sort(key=lambda row: int(row["obs_id"]))

    latency = {
        (row["action"], row["workload_bin"], row["background"]): float(row["tau_hat_ms"])
        for row in read_csv(paths["latency_predictor"])
    }
    thermal = json.loads(paths["thermal_predictor"].read_text())
    margins = json.loads(paths["guard_margins"].read_text())
    actions = config["actions"]
    latency_cap = float(config["constraints"]["latency_cap_ms"])
    thermal_cap = float(config["constraints"]["thermal_cap_c"])
    background_duty = float(workload["background"]["duty_cycle"])

    args.output.mkdir(parents=True)
    trace_path = args.output / "expected_trace_indices.csv"
    trace_fields = ["seed", "window_id", "scheduler_sequence", "source_row_index", "feature_group_hex"]
    seed_hashes: dict[str, str] = {}
    with trace_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=trace_fields, lineterminator="\n")
        writer.writeheader()
        for seed in config["generator"]["matched_seeds"]:
            order = np.random.default_rng(seed).permutation(len(source_rows))
            rows = []
            for window_id, index in enumerate(order[:config["workload"]["windows_per_run"]]):
                source = source_rows[int(index)]
                row = {
                    "seed": seed,
                    "window_id": window_id,
                    "scheduler_sequence": int(index),
                    "source_row_index": source["source_row_index"],
                    "feature_group_hex": source["feature_group_hex"],
                }
                rows.append(row)
                writer.writerow(row)
            seed_hashes[str(seed)] = hashlib.sha256(canonical_seed_trace(rows)).hexdigest()

    diagnostic_path = args.output / "safe_set_diagnostic.csv"
    diagnostic_fields = [
        "seed", "window_id", "calibration_obs_id", "T_start_C", "safe_set_size",
        "safe_set_mask", *[f"{action}_admissible" for action in actions],
    ]
    size_counts: Counter[int] = Counter()
    action_counts: Counter[str] = Counter()
    diagnostic_rows = 0
    with diagnostic_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=diagnostic_fields, lineterminator="\n")
        writer.writeheader()
        for seed in config["generator"]["matched_seeds"]:
            for window_id in range(config["workload"]["windows_per_run"]):
                state = fit_rows[window_id % len(fit_rows)]
                temperature = float(state["T_start_C"])
                allowed = []
                record: dict[str, object] = {
                    "seed": seed,
                    "window_id": window_id,
                    "calibration_obs_id": state["obs_id"],
                    "T_start_C": f"{temperature:.3f}",
                }
                for action in actions:
                    key = (action, f"n{workload['n_records']}", workload["background"]["id"])
                    latency_hat = latency[key]
                    latency_upper = (
                        latency_hat + float(margins["epsilon_tau_ms_by_action"][action])
                        + float(margins["fixed_latency_buffer_ms"])
                    )
                    temperature_upper = thermal_upper(
                        thermal, margins, action, temperature, int(workload["n_records"]),
                        background_duty, latency_hat,
                    )
                    admissible = latency_upper <= latency_cap and temperature_upper <= thermal_cap
                    record[f"{action}_admissible"] = int(admissible)
                    if admissible:
                        allowed.append(action)
                        action_counts[action] += 1
                record["safe_set_size"] = len(allowed)
                record["safe_set_mask"] = "|".join(allowed)
                writer.writerow(record)
                size_counts[len(allowed)] += 1
                diagnostic_rows += 1

    expected = config.get("expected_workload_generation_sha256", {})
    if expected:
        if expected["expected_trace_indices.csv"] != sha256(trace_path):
            raise RuntimeError("expected trace manifest hash mismatch")
        if expected["per_seed_canonical_trace"] != seed_hashes:
            raise RuntimeError("per-seed expected trace hash mismatch")

    summary = {
        "status": "NEAR_CAP_FREEZE_DIAGNOSTIC_PASS",
        "config_sha256": sha256(args.config),
        "selected_candidate": {"n_records": workload["n_records"], "background": workload["background"]["id"]},
        "candidate_order_was_predefined": True,
        "qualification_criteria_were_predefined": True,
        "official_test_access": "not used",
        "policy_outcomes_access": "not used",
        "diagnostic_state_source": "V2 fit n1/B0 T_start states in obs_id order, cycled outcome-blind",
        "diagnostic_rows": diagnostic_rows,
        "safe_set_size": {
            str(size): {"count": size_counts[size], "fraction": size_counts[size] / diagnostic_rows}
            for size in range(len(actions) + 1)
        },
        "per_action_admissibility": {
            action: {"count": action_counts[action], "rate": action_counts[action] / diagnostic_rows}
            for action in actions
        },
        "per_seed_canonical_trace_sha256": seed_hashes,
        "artifact_sha256": {
            trace_path.name: sha256(trace_path),
            diagnostic_path.name: sha256(diagnostic_path),
        },
    }
    (args.output / "NEAR_CAP_FREEZE_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
