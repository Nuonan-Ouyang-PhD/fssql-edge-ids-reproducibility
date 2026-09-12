#!/usr/bin/env python3
"""Evaluate a held-out coverage split without changing frozen guard artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fit_guard_predictors import cell, read_split, sha256, thermal_design


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-config", type=Path, required=True)
    parser.add_argument("--predictor-config", type=Path, required=True)
    parser.add_argument("--predictors", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rate(rows: list[dict[str, object]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    collection = json.loads(args.collection_config.read_text())
    predictor_config = json.loads(args.predictor_config.read_text())
    collection_hash = sha256(args.collection_config)
    predictor_hash = sha256(args.predictor_config)
    coverage_manifest, coverage_rows = read_split(args.coverage, "coverage", collection_hash)

    fit_summary = json.loads((args.predictors / "FIT_MARGIN_SUMMARY.json").read_text())
    margins = json.loads((args.predictors / "guard_margins.json").read_text())
    thermal = json.loads((args.predictors / "thermal_predictor.json").read_text())
    if fit_summary["status"] != "FIT_AND_MARGIN_FROZEN_BEFORE_COVERAGE":
        raise RuntimeError("fit/margin artifacts are not frozen")
    if margins["status"] != "FROZEN_FROM_MARGIN_SPLIT_BEFORE_COVERAGE":
        raise RuntimeError("guard margins are not frozen")
    if fit_summary["collection_config_sha256"] != collection_hash:
        raise RuntimeError("collection configuration changed after fitting")
    if fit_summary["predictor_config_sha256"] != predictor_hash:
        raise RuntimeError("predictor configuration changed after fitting")
    for name, expected in fit_summary["artifact_sha256"].items():
        if sha256(args.predictors / name) != expected:
            raise RuntimeError(f"frozen predictor artifact changed: {name}")

    latency_lookup = {}
    with (args.predictors / "latency_predictor.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            latency_lookup[(row["action"], row["workload_bin"], row["background"])] = float(row["tau_hat_ms"])
    expected_cells = (
        len(collection["actions"])
        * len(collection["workload_records_per_window"])
        * len(collection["background_conditions"])
    )
    if len(latency_lookup) != expected_cells:
        raise RuntimeError("incomplete frozen latency lookup")

    mean = np.asarray(thermal["continuous_mean"], dtype=np.float64)
    scale = np.asarray(thermal["continuous_scale"], dtype=np.float64)
    coefficients = np.asarray(thermal["coefficients"], dtype=np.float64)
    thermal_predictions = thermal_design(coverage_rows, mean, scale) @ coefficients
    epsilon_tau = margins["epsilon_tau_ms_by_action"]
    epsilon_t = float(margins["epsilon_T_C_shared"])
    fixed_latency_buffer = float(margins["fixed_latency_buffer_ms"])
    fixed_thermal_buffer = float(margins["fixed_thermal_buffer_c"])

    evaluated: list[dict[str, object]] = []
    for row, thermal_prediction in zip(coverage_rows, thermal_predictions):
        latency_prediction = latency_lookup[cell(row)]
        latency = float(row["end_to_end_ms"])
        next_temperature = float(row["T_end_C"])
        latency_upper = (
            latency_prediction
            + float(epsilon_tau[row["action"]])
            + fixed_latency_buffer
        )
        thermal_upper = float(thermal_prediction) + epsilon_t + fixed_thermal_buffer
        evaluated.append({
            "device_id": row["device_id"],
            "split": "coverage",
            "action": row["action"],
            "workload_bin": row["workload_bin"],
            "background": row["background"],
            "obs_id": row["obs_id"],
            "tau_measured_ms": latency,
            "tau_hat_ms": latency_prediction,
            "tau_residual_ms": latency - latency_prediction,
            "tau_upper_ms": latency_upper,
            "latency_covered": latency <= latency_upper,
            "T_next_measured_C": next_temperature,
            "T_next_hat_C": float(thermal_prediction),
            "T_residual_C": next_temperature - float(thermal_prediction),
            "T_upper_C": thermal_upper,
            "thermal_covered": next_temperature <= thermal_upper,
            "joint_covered": latency <= latency_upper and next_temperature <= thermal_upper,
        })

    grouped_action: dict[str, list[dict[str, object]]] = defaultdict(list)
    grouped_cell: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in evaluated:
        grouped_action[str(row["action"])].append(row)
        grouped_cell[(str(row["action"]), str(row["workload_bin"]), str(row["background"]))].append(row)

    def coverage_record(rows: list[dict[str, object]]) -> dict[str, object]:
        return {
            "n": len(rows),
            "latency_coverage": rate(rows, "latency_covered"),
            "thermal_coverage": rate(rows, "thermal_covered"),
            "joint_coverage": rate(rows, "joint_covered"),
        }

    args.output.mkdir(parents=True)
    residual_path = args.output / "guard_residuals_coverage.csv"
    with residual_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evaluated[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(evaluated)

    fit_margin_rows = []
    with (args.predictors / "guard_residuals_fit_margin.csv").open(newline="", encoding="utf-8") as stream:
        fit_margin_rows = list(csv.DictReader(stream))
    ecdf_path = args.output / "guard_residual_ecdf.csv"
    with ecdf_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["split", "action", "metric", "n", "rank", "value", "ecdf"])
        for split in ("fit", "margin", "coverage"):
            source = (
                [row for row in fit_margin_rows if row["split"] == split]
                if split != "coverage"
                else evaluated
            )
            for action in collection["actions"]:
                action_rows = [row for row in source if row["action"] == action]
                for metric in ("tau_residual_ms", "T_residual_C"):
                    values = sorted(float(row[metric]) for row in action_rows)
                    for index, value in enumerate(values, start=1):
                        writer.writerow([split, action, metric, len(values), index, value, index / len(values)])

    summary = {
        "status": "HELD_OUT_COVERAGE_REPORTED_NO_TUNING",
        "device_id": collection["device_id"],
        "protocol_version": collection.get("protocol_version", 1),
        "calibration_round": collection["calibration_round"],
        "coverage_run_id": coverage_manifest["run_id"],
        "coverage_raw_sha256": coverage_manifest["raw_windows_sha256"],
        "fit_run_id": margins["fit_run_id"],
        "margin_run_id": margins["margin_run_id"],
        "collection_config_sha256": collection_hash,
        "predictor_config_sha256": predictor_hash,
        "fit_margin_summary_sha256": sha256(args.predictors / "FIT_MARGIN_SUMMARY.json"),
        "overall": coverage_record(evaluated),
        "by_action": {key: coverage_record(rows) for key, rows in sorted(grouped_action.items())},
        "by_cell": {
            "|".join(key): coverage_record(rows)
            for key, rows in sorted(grouped_cell.items())
        },
        "target_report_not_guarantee": predictor_config["margin"]["quantile"],
        "tuning_performed": False,
        "artifact_sha256": {
            residual_path.name: sha256(residual_path),
            ecdf_path.name: sha256(ecdf_path),
        },
    }
    (args.output / "guard_coverage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
