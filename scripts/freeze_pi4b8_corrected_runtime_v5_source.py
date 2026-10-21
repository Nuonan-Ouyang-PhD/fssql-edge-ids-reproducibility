#!/usr/bin/env python3
"""Freeze the authoritative source inventory for the corrected Pi4B V5 runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from online.pi4b8.corrected_v5_contract import (
    COMMISSIONING_CONDITION,
    COMMISSIONING_POLICY,
    COMMISSIONING_SEED,
    CONDITIONS,
    SEEDS,
    SOURCE_SCHEMA,
    SOURCE_STATUS,
    ContractError,
    validate_source_manifest,
)


INPUT_FREEZE = ROOT / "protocols/PI4B8_TRUE_N16_INPUT_FREEZE_V1.json"
Q_MANIFEST = ROOT / "q_training/pi4b8_true_n16_v2/Q_TABLE_MANIFEST.json"
COMMISSIONING_PREREG = (
    ROOT / "protocols/PI4B8_CORRECTED_V5_COMMISSIONING_PREREGISTRATION.json"
)


SHARED_SPECS = (
    ("worker", "online/pi4b8/pi4b8_online_worker.py", "pi4b8_online_worker.py"),
    (
        "runtime_contract",
        "online/pi4b8/corrected_v5_contract.py",
        "runtime/corrected_v5_contract.py",
    ),
    (
        "runtime_helper:action_pool_numpy",
        "runtime/action_pool_numpy.py",
        "runtime/action_pool_numpy.py",
    ),
    (
        "runtime_helper:preprocessor_numpy",
        "runtime/preprocessor_numpy.py",
        "runtime/preprocessor_numpy.py",
    ),
    (
        "runtime_helper:risk_proxy_numpy",
        "runtime/risk_proxy_numpy.py",
        "runtime/risk_proxy_numpy.py",
    ),
    (
        "runtime_config",
        "configs/pi4b8_corrected_runtime_v5.json",
        "config/pi4b8_corrected_runtime_v5.json",
    ),
    (
        "runtime_dependencies",
        "configs/PI4B8_RUNTIME_DEPENDENCIES_V1.json",
        "config/PI4B8_RUNTIME_DEPENDENCIES_V1.json",
    ),
    (
        "model_hyperparameters",
        "models/MODEL_HYPERPARAMETERS.json",
        "config/MODEL_HYPERPARAMETERS.json",
    ),
    (
        "validation_metrics",
        "models/VALIDATION_METRICS.csv",
        "config/VALIDATION_METRICS.csv",
    ),
    (
        "q_config",
        "configs/pi4b8_true_n16_q_v2.json",
        "config/pi4b8_true_n16_q_v2.json",
    ),
    (
        "q_manifest",
        "q_training/pi4b8_true_n16_v2/Q_TABLE_MANIFEST.json",
        "q_tables/Q_TABLE_MANIFEST.json",
    ),
    (
        "preprocessor",
        "preprocessing/preprocessor_portable.npz",
        "preprocessing/preprocessor_portable.npz",
    ),
    ("risk_proxy", "risk_proxy/risk_proxy.npz", "risk_proxy/risk_proxy.npz"),
    (
        "guard:latency_predictor",
        "calibration/pi4b8/frozen_guard_fit_margin_v1_run2/latency_predictor.csv",
        "guard/latency_predictor.csv",
    ),
    (
        "guard:thermal_predictor",
        "calibration/pi4b8/frozen_guard_fit_margin_v1_run2/thermal_predictor.json",
        "guard/thermal_predictor.json",
    ),
    (
        "guard:margins",
        "calibration/pi4b8/frozen_guard_fit_margin_v1_run2/guard_margins.json",
        "guard/guard_margins.json",
    ),
    (
        "powerz:capture",
        "online/pi4b8/powerz_hid_capture.py",
        "powerz/powerz_hid_capture.py",
    ),
    (
        "powerz:specification",
        "protocols/POWER_Z_HID_ACQUISITION_V1.md",
        "powerz/POWER_Z_HID_ACQUISITION_V1.md",
    ),
)

MODEL_SPECS = tuple(
    (
        f"model:{name}",
        f"models/artifacts/{name}.npz",
        f"models/{name}.npz",
    )
    for name in (
        "fisvdd_revision",
        "lucid_revision",
        "tinydl_revision",
        "oi_svdd_as_elm_revision",
    )
)

Q_SPECS = tuple(
    (
        f"q_table:{seed}",
        f"q_training/pi4b8_true_n16_v2/tables/fssql_r_seed_{seed}.npz",
        f"q_tables/fssql_r_seed_{seed}.npz",
    )
    for seed in SEEDS
)

INPUT_SPECS = (
    (
        "base:raw_features",
        "device_calibration/inputs/scheduler_train_6000_raw_features.csv",
        "inputs/base/raw_features.csv",
    ),
    (
        "base:features",
        "device_calibration/inputs/scheduler_train_6000_features.npy",
        "inputs/base/features.npy",
    ),
    (
        "base:risk_features",
        "device_calibration/inputs/scheduler_train_6000_risk_features.npy",
        "inputs/base/risk_features.npy",
    ),
    (
        "base:backlog",
        "device_calibration/inputs/pi4b8_true_n16_v1/evaluation_backlog_zero.npy",
        "inputs/base/backlog.npy",
    ),
    (
        "base:labels",
        "device_calibration/inputs/scheduler_train_6000_labels.npy",
        "scoring/base/labels.npy",
    ),
) + tuple(
    (
        f"batch_indices:{seed}",
        f"device_calibration/inputs/pi4b8_true_n16_v1/evaluation_batch_indices_seed_{seed}.npy",
        f"inputs/batches/seed_{seed}.npy",
    )
    for seed in SEEDS
)

COMMISSIONING_INPUT_SPEC = (
    f"commissioning_batch_indices:{COMMISSIONING_SEED}",
    (
        "device_calibration/inputs/pi4b8_true_n16_v1/"
        f"commissioning_batch_indices_seed_{COMMISSIONING_SEED}.npy"
    ),
    f"inputs/commissioning/seed_{COMMISSIONING_SEED}.npy",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def git(*arguments: str) -> bytes:
    return subprocess.check_output(["git", *arguments], cwd=ROOT)


def expected_input_hashes(document: dict[str, Any]) -> dict[str, str]:
    expected = {
        "base:raw_features": document["base_artifacts"]["raw_features"]["sha256"],
        "base:features": document["base_artifacts"]["features"]["sha256"],
        "base:risk_features": document["base_artifacts"]["risk_features"]["sha256"],
        "base:labels": document["base_artifacts"]["scoring_labels"]["sha256"],
        "base:backlog": document["evaluation_backlog"]["sha256"],
    }
    expected.update(
        {
            f"batch_indices:{seed}": document["evaluation_batch_indices"][str(seed)]
            for seed in SEEDS
        }
    )
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "protocols/PI4B8_CORRECTED_RUNTIME_SOURCE_MANIFEST_V5.json",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite source manifest: {output}")
    if git("status", "--porcelain", "--untracked-files=all").strip():
        raise SystemExit("source-manifest freeze requires a clean Git worktree")

    commit = git("rev-parse", "HEAD").decode().strip()
    input_freeze = load_json(INPUT_FREEZE)
    q_manifest = load_json(Q_MANIFEST)
    commissioning_prereg = load_json(COMMISSIONING_PREREG)
    if input_freeze.get("status") != "FROZEN_BEFORE_PI4B8_TRUE_N16_Q_TRAINING":
        raise SystemExit("true-n16 input manifest is not frozen")
    if q_manifest.get("status") != "PI4B8_TRUE_N16_Q_TRAINING_COMPLETE":
        raise SystemExit("corrected Q manifest is not complete")
    if (
        commissioning_prereg.get("status")
        != "FROZEN_BEFORE_CORRECTED_PI4B8_COMMISSIONING"
        or commissioning_prereg.get("official_test_access") != "forbidden"
        or commissioning_prereg.get("policy") != COMMISSIONING_POLICY
        or commissioning_prereg.get("condition") != COMMISSIONING_CONDITION
        or int(commissioning_prereg.get("seed", -1)) != COMMISSIONING_SEED
        or commissioning_prereg.get("windows") != 600
        or commissioning_prereg.get("records_per_window") != 16
        or commissioning_prereg.get("formal_seed") is not False
        or commissioning_prereg.get("batch_rule")
        != "order=default_rng(seed).permutation(6000); index[w,j]=order[(16*w+j)%6000]"
        or commissioning_prereg.get("batch_indices", {}).get("path")
        != COMMISSIONING_INPUT_SPEC[1]
        or commissioning_prereg.get("scientific_use")
        != "COMMISSIONING_ONLY_EXCLUDED_FROM_FORMAL_STATISTICS"
    ):
        raise SystemExit("corrected commissioning preregistration is inconsistent")

    expected_inputs = expected_input_hashes(input_freeze)
    expected_inputs[COMMISSIONING_INPUT_SPEC[0]] = commissioning_prereg[
        "batch_indices"
    ]["sha256"]
    expected_q = {
        f"q_table:{row['seed']}": row["table_sha256"]
        for row in q_manifest.get("runs", [])
        if row.get("policy") == "FSSQL-R"
    }
    if set(expected_q) != {f"q_table:{seed}" for seed in SEEDS}:
        raise SystemExit("corrected Q manifest does not bind five unique seeds")

    artifacts = []
    all_input_specs = INPUT_SPECS + (COMMISSIONING_INPUT_SPEC,)
    input_roles = {role for role, _, _ in all_input_specs}
    for role, source_path, package_path in (
        SHARED_SPECS + MODEL_SPECS + Q_SPECS + all_input_specs
    ):
        path = ROOT / source_path
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"missing or symlinked source artifact: {role}: {path}")
        digest = sha256_bytes(path.read_bytes())
        if role in expected_inputs and digest != expected_inputs[role]:
            raise SystemExit(f"frozen input hash mismatch: {role}")
        if role in expected_q and digest != expected_q[role]:
            raise SystemExit(f"corrected Q-table hash mismatch: {role}")
        if role not in input_roles:
            committed = git("show", f"{commit}:{source_path}")
            if sha256_bytes(committed) != digest:
                raise SystemExit(f"source_commit does not contain packaged bytes: {role}")
        artifacts.append(
            {
                "role": role,
                "source_path": source_path,
                "package_path": package_path,
                "sha256": digest,
                "authority": (
                    "protocols/PI4B8_CORRECTED_V5_COMMISSIONING_PREREGISTRATION.json"
                    if role == COMMISSIONING_INPUT_SPEC[0]
                    else "protocols/PI4B8_TRUE_N16_INPUT_FREEZE_V1.json"
                    if role in input_roles
                    else "source_commit"
                ),
            }
        )

    cells = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            cells.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "runtime": {
                        "raw_features": "base:raw_features",
                        "features": "base:features",
                        "risk_features": "base:risk_features",
                        "backlog": "base:backlog",
                        "batch_indices": f"batch_indices:{seed}",
                    },
                    "scoring": {"labels": "base:labels"},
                }
            )

    document = {
        "schema_version": SOURCE_SCHEMA,
        "status": SOURCE_STATUS,
        "official_test_access": "forbidden",
        "execution_authorized": False,
        "runtime_dependencies_required": True,
        "commissioning_cell_required": True,
        "source_commit": commit,
        "base_records": 6000,
        "windows_per_run": 600,
        "records_per_window": 16,
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "input_freeze_manifest_sha256": sha256_bytes(INPUT_FREEZE.read_bytes()),
        "q_manifest_sha256": sha256_bytes(Q_MANIFEST.read_bytes()),
        "commissioning_preregistration_sha256": sha256_bytes(
            COMMISSIONING_PREREG.read_bytes()
        ),
        "artifacts": artifacts,
        "cells": cells,
        "commissioning_cell": {
            "scientific_use": "COMMISSIONING_ONLY_EXCLUDED_FROM_FORMAL_STATISTICS",
            "policy": COMMISSIONING_POLICY,
            "condition": COMMISSIONING_CONDITION,
            "seed": COMMISSIONING_SEED,
            "runtime": {
                "raw_features": "base:raw_features",
                "features": "base:features",
                "risk_features": "base:risk_features",
                "backlog": "base:backlog",
                "batch_indices": COMMISSIONING_INPUT_SPEC[0],
            },
            "scoring": {"labels": "base:labels"},
        },
    }
    validate_source_manifest(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PI4B8_CORRECTED_RUNTIME_V5_SOURCE_FREEZE_PASS")
    print(f"source_commit={commit}")
    print(f"artifacts={len(artifacts)} cells={len(cells)}")
    print(f"manifest_sha256={sha256_bytes(output.read_bytes())}")


if __name__ == "__main__":
    main()
