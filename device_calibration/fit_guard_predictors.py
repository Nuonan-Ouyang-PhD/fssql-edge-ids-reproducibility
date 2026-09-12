#!/usr/bin/env python3
"""Fit frozen guard predictors from fit and margin splits only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-config", type=Path, required=True)
    parser.add_argument("--predictor-config", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--margin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_split(path: Path, expected_split: str, config_hash: str) -> tuple[dict, list[dict[str, str]]]:
    manifest = json.loads((path / "RUN_MANIFEST.json").read_text())
    if manifest["status"] != "FORMAL_CALIBRATION_SPLIT_PASS" or not manifest["formal"]:
        raise RuntimeError(f"not a valid formal split: {path}")
    if manifest["split"] != expected_split:
        raise RuntimeError(f"expected {expected_split}, found {manifest['split']}")
    if manifest["config_sha256"] != config_hash:
        raise RuntimeError(f"collection config mismatch: {path}")
    raw_path = path / "raw_windows.csv"
    if sha256(raw_path) != manifest["raw_windows_sha256"]:
        raise RuntimeError(f"raw hash mismatch: {path}")
    with raw_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != manifest["windows_expected"] or len(rows) != manifest["windows_observed"]:
        raise RuntimeError(f"row count mismatch: {path}")
    if any(row["valid"] != "1" or row["split"] != expected_split for row in rows):
        raise RuntimeError(f"invalid or mixed rows: {path}")
    return manifest, rows


def quantile_higher(values: list[float], q: float) -> float:
    if not values:
        raise RuntimeError("empty quantile input")
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="higher"))


def cell(row: dict[str, str]) -> tuple[str, str, str]:
    return row["action"], row["workload_bin"], row["background"]


def continuous(row: dict[str, str]) -> list[float]:
    return [
        float(row["T_start_C"]),
        math.log2(int(row["n_records"])),
        float(row["background_duty"]),
        float(row["end_to_end_ms"]),
    ]


def thermal_design(
    rows: list[dict[str, str]],
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    numeric = np.asarray([continuous(row) for row in rows], dtype=np.float64)
    standardized = (numeric - mean) / scale
    indicators = np.asarray([
        [
            float(row["action"] == "lucid_revision"),
            float(row["action"] == "tinydl_revision"),
            float(row["action"] == "oi_svdd_as_elm_revision"),
        ]
        for row in rows
    ], dtype=np.float64)
    return np.column_stack((np.ones(len(rows)), standardized, indicators))


def residual_record(
    row: dict[str, str],
    latency_prediction: float,
    thermal_prediction: float,
) -> dict[str, object]:
    latency = float(row["end_to_end_ms"])
    thermal = float(row["T_end_C"])
    return {
        "device_id": row["device_id"],
        "split": row["split"],
        "action": row["action"],
        "workload_bin": row["workload_bin"],
        "background": row["background"],
        "obs_id": row["obs_id"],
        "tau_measured_ms": latency,
        "tau_hat_ms": latency_prediction,
        "tau_residual_ms": latency - latency_prediction,
        "T_next_measured_C": thermal,
        "T_next_hat_C": thermal_prediction,
        "T_residual_C": thermal - thermal_prediction,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    collection = json.loads(args.collection_config.read_text())
    predictor = json.loads(args.predictor_config.read_text())
    if predictor["status"] != "FROZEN_BEFORE_FIT_RESULT_INSPECTION":
        raise RuntimeError("predictor configuration is not frozen")
    if predictor["official_test_access"] != "forbidden":
        raise RuntimeError("unexpected test-access rule")
    fixed_latency_buffer = float(predictor["margin"]["fixed_latency_buffer_ms"])
    fixed_thermal_buffer = float(predictor["margin"]["fixed_thermal_buffer_c"])
    collection_margin = collection["margin_rule"]
    if fixed_latency_buffer != float(collection_margin["fixed_latency_buffer_ms"]):
        raise RuntimeError("fixed latency buffer differs between frozen configurations")
    if fixed_thermal_buffer != float(collection_margin["fixed_thermal_buffer_c"]):
        raise RuntimeError("fixed thermal buffer differs between frozen configurations")
    collection_hash = sha256(args.collection_config)
    predictor_hash = sha256(args.predictor_config)
    fit_manifest, fit_rows = read_split(args.fit, "fit", collection_hash)
    margin_manifest, margin_rows = read_split(args.margin, "margin", collection_hash)
    if fit_manifest["run_id"] == margin_manifest["run_id"]:
        raise RuntimeError("fit and margin run IDs must differ")

    expected_cells = (
        len(collection["actions"])
        * len(collection["workload_records_per_window"])
        * len(collection["background_conditions"])
    )
    observations = int(collection["observations_per_action_workload_background_cell"])
    for split_name, rows in (("fit", fit_rows), ("margin", margin_rows)):
        counts: dict[tuple[str, str, str], int] = defaultdict(int)
        for row in rows:
            counts[cell(row)] += 1
        if len(counts) != expected_cells or set(counts.values()) != {observations}:
            raise RuntimeError(f"incomplete {split_name} cell matrix")

    fit_latency: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in fit_rows:
        fit_latency[cell(row)].append(float(row["end_to_end_ms"]))
    latency_lookup = {
        key: quantile_higher(values, 0.99)
        for key, values in fit_latency.items()
    }

    fit_continuous = np.asarray([continuous(row) for row in fit_rows], dtype=np.float64)
    continuous_mean = fit_continuous.mean(axis=0)
    continuous_scale = fit_continuous.std(axis=0)
    if np.any(continuous_scale == 0):
        raise RuntimeError("zero-variance thermal predictor feature")
    fit_design = thermal_design(fit_rows, continuous_mean, continuous_scale)
    fit_target = np.asarray([float(row["T_end_C"]) for row in fit_rows], dtype=np.float64)
    coefficients, _, rank, singular_values = np.linalg.lstsq(fit_design, fit_target, rcond=None)
    if rank != fit_design.shape[1]:
        raise RuntimeError(f"rank-deficient thermal design: {rank}/{fit_design.shape[1]}")

    fit_thermal = fit_design @ coefficients
    margin_design = thermal_design(margin_rows, continuous_mean, continuous_scale)
    margin_thermal = margin_design @ coefficients
    fit_residuals = [
        residual_record(row, latency_lookup[cell(row)], float(thermal_prediction))
        for row, thermal_prediction in zip(fit_rows, fit_thermal)
    ]
    margin_residuals = [
        residual_record(row, latency_lookup[cell(row)], float(thermal_prediction))
        for row, thermal_prediction in zip(margin_rows, margin_thermal)
    ]

    latency_epsilon = {}
    for action in collection["actions"]:
        values = [
            float(row["tau_residual_ms"])
            for row in margin_residuals
            if row["action"] == action
        ]
        latency_epsilon[action] = max(0.0, quantile_higher(values, 0.99))
    thermal_epsilon = max(
        0.0,
        quantile_higher([float(row["T_residual_C"]) for row in margin_residuals], 0.99),
    )

    args.output.mkdir(parents=True)
    latency_path = args.output / "latency_predictor.csv"
    with latency_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["action", "workload_bin", "background", "fit_n", "tau_hat_ms"])
        for key in sorted(latency_lookup):
            writer.writerow([*key, len(fit_latency[key]), f"{latency_lookup[key]:.9f}"])

    thermal_path = args.output / "thermal_predictor.json"
    thermal_artifact = {
        "feature_order": [
            "intercept", "scaled_T_start_C", "scaled_log2_n_records",
            "scaled_background_duty", "scaled_end_to_end_ms", "is_lucid_revision",
            "is_tinydl_revision", "is_oi_svdd_as_elm_revision",
        ],
        "continuous_feature_order": predictor["thermal_predictor"]["continuous_features"],
        "continuous_mean": continuous_mean.tolist(),
        "continuous_scale": continuous_scale.tolist(),
        "coefficients": coefficients.tolist(),
        "rank": int(rank),
        "singular_values": singular_values.tolist(),
        "fit_n": len(fit_rows),
    }
    thermal_path.write_text(json.dumps(thermal_artifact, indent=2) + "\n")

    margin_path = args.output / "guard_margins.json"
    margins = {
        "status": "FROZEN_FROM_MARGIN_SPLIT_BEFORE_COVERAGE",
        "quantile": 0.99,
        "quantile_method": "higher",
        "fixed_latency_buffer_ms": fixed_latency_buffer,
        "fixed_thermal_buffer_c": fixed_thermal_buffer,
        "epsilon_tau_ms_by_action": latency_epsilon,
        "epsilon_T_C_shared": thermal_epsilon,
        "fit_run_id": fit_manifest["run_id"],
        "margin_run_id": margin_manifest["run_id"],
        "fit_n": len(fit_rows),
        "margin_n": len(margin_rows),
        "coverage_access": "not used",
    }
    margin_path.write_text(json.dumps(margins, indent=2) + "\n")

    residual_path = args.output / "guard_residuals_fit_margin.csv"
    residual_fields = list(fit_residuals[0])
    with residual_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=residual_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(fit_residuals)
        writer.writerows(margin_residuals)

    outputs = [latency_path, thermal_path, margin_path, residual_path]
    summary = {
        "status": "FIT_AND_MARGIN_FROZEN_BEFORE_COVERAGE",
        "device_id": collection["device_id"],
        "protocol_version": collection.get("protocol_version", 1),
        "calibration_round": collection["calibration_round"],
        "fit_run_id": fit_manifest["run_id"],
        "margin_run_id": margin_manifest["run_id"],
        "fit_raw_sha256": fit_manifest["raw_windows_sha256"],
        "margin_raw_sha256": margin_manifest["raw_windows_sha256"],
        "collection_config_sha256": collection_hash,
        "predictor_config_sha256": predictor_hash,
        "fit_n": len(fit_rows),
        "margin_n": len(margin_rows),
        "latency_cells": len(latency_lookup),
        "thermal_design_columns": fit_design.shape[1],
        "coverage_access": "not used",
        "artifact_sha256": {path.name: sha256(path) for path in outputs},
    }
    (args.output / "FIT_MARGIN_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
