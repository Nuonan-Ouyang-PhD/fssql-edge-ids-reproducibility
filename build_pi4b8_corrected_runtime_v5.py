#!/usr/bin/env python3
"""Materialize a corrected Pi4B V5 package from an already frozen manifest.

The builder makes no scientific choices.  Every source path and SHA-256 must
be present in the input manifest.  It never reads the official test and never
falls back to historical V4 assets.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from online.pi4b8.corrected_v5_contract import (
    PACKAGE_MANIFEST_NAME,
    PACKAGE_SCHEMA,
    PACKAGE_STATUS,
    SOURCE_MANIFEST_NAME,
    ContractError,
    load_json,
    sha256,
    validate_materialized_package,
    validate_source_manifest,
)
from runtime.preprocessor_numpy import PortablePreprocessor
from runtime.action_pool_numpy import NumpyActionModel
from runtime.risk_proxy_numpy import RiskProxy


ROOT = Path(__file__).resolve().parent


def _array(path: Path, role: str) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ContractError(f"cannot load {role}: {path}: {error}") from error
    if np.iscomplexobj(value) or not np.issubdtype(value.dtype, np.number):
        raise ContractError(f"{role} must be a real numeric array")
    if not np.isfinite(value).all():
        raise ContractError(f"{role} contains non-finite values")
    return value


def _raw_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        lowered = [field.strip().lower() for field in fieldnames]
        if len(lowered) != len(set(lowered)):
            raise ContractError(f"duplicate raw-input columns: {path}")
        forbidden = {
            "id",
            "label",
            "attack_cat",
            "source_id",
            "source_row_index",
            "runtime_sequence",
            "source_feature_hash",
            "feature_group_hex",
            "group_id",
            "duplicate_group",
            "fold",
            "split",
            "row_id",
            "original_row_index",
        }
        leaked = [field for field in lowered if field in forbidden or "label" in field]
        if leaked:
            raise ContractError(f"raw runtime input contains scoring/audit columns: {leaked}")
        rows = list(reader)
    if len(rows) != 6000:
        raise ContractError(f"raw base input must contain 6000 rows: {path}")
    return rows


def validate_cell_payloads(package_root: Path, manifest: dict[str, Any]) -> None:
    """Validate 6000-row bases plus an explicit 600 x 16 batching schedule."""
    by_role = {entry["role"]: entry for entry in manifest["artifacts"]}
    preprocessor = PortablePreprocessor(
        package_root / by_role["preprocessor"]["package_path"]
    )
    checked_bases: set[tuple[str, str, str]] = set()
    cells = list(manifest["cells"])
    commissioning = manifest.get("commissioning_cell")
    if isinstance(commissioning, dict):
        cells.append(commissioning)
    for cell in cells:
        runtime = cell["runtime"]
        scoring = cell["scoring"]
        raw_path = package_root / by_role[runtime["raw_features"]]["package_path"]
        feature_path = package_root / by_role[runtime["features"]]["package_path"]
        risk_path = package_root / by_role[runtime["risk_features"]]["package_path"]
        backlog_path = package_root / by_role[runtime["backlog"]]["package_path"]
        indices_path = package_root / by_role[runtime["batch_indices"]]["package_path"]
        labels_path = package_root / by_role[scoring["labels"]]["package_path"]

        base_key = (str(raw_path), str(feature_path), str(risk_path))
        if base_key not in checked_bases:
            rows = _raw_rows(raw_path)
            features = _array(feature_path, "base features")
            risk_features = _array(risk_path, "base risk features")
            if features.ndim != 2 or features.shape[0] != 6000:
                raise ContractError(f"base features must have shape (6000, d): {features.shape}")
            if risk_features.ndim != 2 or risk_features.shape[0] != 6000:
                raise ContractError(
                    f"base risk features must have shape (6000, d): {risk_features.shape}"
                )
            transformed = preprocessor.transform(rows)
            if not np.array_equal(transformed, features):
                raise ContractError(
                    f"portable-preprocessor roundtrip does not reproduce {feature_path}"
                )
            checked_bases.add(base_key)

        backlog = _array(backlog_path, "backlog")
        if backlog.shape != (600,) or np.any(backlog != 0):
            raise ContractError(f"backlog must contain 600 frozen zeros: {backlog.shape}")

        indices = _array(indices_path, "batch indices")
        if indices.shape != (600, 16) or not np.issubdtype(indices.dtype, np.integer):
            raise ContractError(f"batch indices must be an integer (600, 16) array: {indices.shape}")
        if int(indices.min()) < 0 or int(indices.max()) >= 6000:
            raise ContractError("batch indices reference rows outside the 6000-row base")
        if any(len(set(row.tolist())) != 16 for row in indices):
            raise ContractError("a true-n16 package window repeats a base record")
        seed = int(cell["seed"])
        order = np.random.default_rng(seed).permutation(6000)
        expected = order[np.arange(600 * 16, dtype=np.int64) % 6000].reshape(600, 16)
        if not np.array_equal(indices, expected):
            raise ContractError(
                f"batch indices do not reproduce the frozen schedule for seed {seed}"
            )

        labels = _array(labels_path, "scoring labels")
        if labels.shape != (6000,) or not np.isin(labels, [0, 1]).all():
            raise ContractError("scoring sidecar must contain 6000 binary base labels")


def validate_shared_payloads(package_root: Path, manifest: dict[str, Any]) -> None:
    by_role = {entry["role"]: entry for entry in manifest["artifacts"]}

    def path(role: str) -> Path:
        return package_root / by_role[role]["package_path"]

    if manifest.get("runtime_dependencies_required") is True:
        dependencies = load_json(path("runtime_dependencies"))
        if (
            dependencies.get("status")
            != "FROZEN_BEFORE_CORRECTED_PI4B8_COMMISSIONING"
            or dependencies.get("official_test_access") != "forbidden"
            or dependencies.get("pi_worker", {}).get("cpu_governor_required")
            != "performance"
            or dependencies.get("powerz_device", {}).get("serial") != "075356"
            or dependencies.get("device_identity", {}).get("hostname") != "pi4b8g"
            or dependencies.get("device_identity", {}).get("model")
            != "Raspberry Pi 4 Model B Rev 1.5"
            or dependencies.get("device_identity", {}).get("cpu_serial")
            != "100000005368e39d"
            or dependencies.get("device_identity", {}).get("mem_total_kib")
            != 8007464
            or dependencies.get("cooldown")
            != {
                "idle_samples": 30,
                "idle_interval_s": 1.0,
                "maximum_start_above_idle_median_c": 2.0,
            }
        ):
            raise ContractError("corrected runtime dependency manifest is incomplete")

    # Loadability only; thresholds remain bound to model_hyperparameters and
    # are deliberately not supplied or selected by the package builder.
    for action in (
        "fisvdd_revision",
        "lucid_revision",
        "tinydl_revision",
        "oi_svdd_as_elm_revision",
    ):
        NumpyActionModel(path(f"model:{action}"))
    RiskProxy(path("risk_proxy"))

    q_manifest = load_json(path("q_manifest"))
    for seed in (5101, 5102, 5103, 5104, 5105):
        table_path = path(f"q_table:{seed}")
        matching = [
            record
            for record in q_manifest.get("runs", [])
            if record.get("policy") == "FSSQL-R" and int(record.get("seed", -1)) == seed
        ]
        if len(matching) != 1 or matching[0].get("table_sha256") != sha256(table_path):
            raise ContractError(f"Q manifest has no unique hash binding for seed {seed}")
        with np.load(table_path, allow_pickle=False) as table:
            required = {"q_values", "actions", "training_seed", "shielded"}
            if not required.issubset(table.files):
                raise ContractError(f"Q table {seed} is missing embedded lineage")
            if table["q_values"].shape != (1280, 4) or not np.isfinite(
                table["q_values"]
            ).all():
                raise ContractError(f"Q table {seed} has invalid values/shape")
            if tuple(table["actions"].tolist()) != (
                "fisvdd_revision",
                "lucid_revision",
                "tinydl_revision",
                "oi_svdd_as_elm_revision",
            ):
                raise ContractError(f"Q table {seed} action order mismatch")
            if int(table["training_seed"]) != seed or not bool(table["shielded"]):
                raise ContractError(f"Q table {seed} seed/shield lineage mismatch")


def build(source_manifest_path: Path, output: Path, repository_root: Path = ROOT) -> Path:
    source_manifest_path = source_manifest_path.resolve()
    output = output.resolve()
    repository_root = repository_root.resolve()
    if output.exists():
        raise ContractError(f"refusing to overwrite corrected package: {output}")
    source = load_json(source_manifest_path)
    artifacts = validate_source_manifest(source)

    missing: list[dict[str, Any]] = []
    for role, record in artifacts.items():
        path = repository_root / record["source_path"]
        if not path.is_file():
            missing.append(
                {
                    "role": role,
                    "source_path": record["source_path"],
                    "expected_sha256": record["sha256"],
                    "reason": "missing",
                }
            )
            continue
        observed = sha256(path)
        if observed != record["sha256"]:
            missing.append(
                {
                    "role": role,
                    "source_path": record["source_path"],
                    "expected_sha256": record["sha256"],
                    "observed_sha256": observed,
                    "reason": "hash_mismatch",
                }
            )
    if missing:
        raise ContractError("source artifact validation failed:\n" + json.dumps(missing, indent=2))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pi4b8-corrected-v5-", dir=output.parent) as tmp:
        staging = Path(tmp) / output.name
        staging.mkdir()
        packaged_artifacts = []
        for role in sorted(artifacts):
            record = artifacts[role]
            source_path = repository_root / record["source_path"]
            destination = staging / record["package_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination)
            observed = sha256(destination)
            if observed != record["sha256"]:
                raise ContractError(f"post-copy hash mismatch for {role}")
            packaged_artifacts.append(
                {
                    "role": role,
                    "package_path": record["package_path"],
                    "bytes": destination.stat().st_size,
                    "sha256": observed,
                }
            )

        frozen_source_copy = staging / SOURCE_MANIFEST_NAME
        shutil.copyfile(source_manifest_path, frozen_source_copy)
        package_manifest = {
            "schema_version": PACKAGE_SCHEMA,
            "status": PACKAGE_STATUS,
            "official_test_access": "forbidden",
            "source_manifest_sha256": sha256(frozen_source_copy),
            "source_commit": source["source_commit"],
            "runtime_dependencies_required": source.get(
                "runtime_dependencies_required", False
            ),
            "commissioning_cell_required": source.get(
                "commissioning_cell_required", False
            ),
            "commissioning_preregistration_sha256": source.get(
                "commissioning_preregistration_sha256"
            ),
            "base_records": source["base_records"],
            "windows_per_run": source["windows_per_run"],
            "records_per_window": source["records_per_window"],
            "conditions": source["conditions"],
            "seeds": source["seeds"],
            "artifacts": packaged_artifacts,
            "cells": source["cells"],
            "commissioning_cell": source.get("commissioning_cell"),
            "execution_authorized": False,
        }
        package_manifest_path = staging / PACKAGE_MANIFEST_NAME
        package_manifest_path.write_text(
            json.dumps(package_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_shared_payloads(staging, package_manifest)
        validate_cell_payloads(staging, package_manifest)
        validate_materialized_package(staging, package_manifest_path)
        staging.rename(output)
    return output / PACKAGE_MANIFEST_NAME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = build(args.source_manifest, args.output)
    except ContractError as error:
        raise SystemExit(f"PI4B8_CORRECTED_RUNTIME_V5_BUILD_BLOCKED: {error}") from error
    print("PI4B8_CORRECTED_RUNTIME_V5_LOCAL_GATE_PASS")
    print(f"manifest={manifest}")
    print(f"manifest_sha256={sha256(manifest)}")


if __name__ == "__main__":
    main()
