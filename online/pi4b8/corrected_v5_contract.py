#!/usr/bin/env python3
"""Hash-bound software contract for a future corrected Pi4B runtime package.

This module does not authorize execution.  It validates a materialized package
whose scientific inputs have already been frozen in a separate source
manifest.  Historical V4 artifacts are deliberately outside this contract.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np


SOURCE_SCHEMA = "PI4B8_CORRECTED_RUNTIME_V5_SOURCE_MANIFEST_1"
PACKAGE_SCHEMA = "PI4B8_CORRECTED_RUNTIME_V5_PACKAGE_MANIFEST_1"
SOURCE_STATUS = "FROZEN_BEFORE_CORRECTED_PI4B8_V5_MATERIALIZATION"
PACKAGE_STATUS = "PI4B8_CORRECTED_RUNTIME_V5_MATERIALIZED"
PACKAGE_MANIFEST_NAME = "PI4B8_CORRECTED_RUNTIME_PACKAGE_MANIFEST_V5.json"
SOURCE_MANIFEST_NAME = "PI4B8_CORRECTED_RUNTIME_SOURCE_MANIFEST_V5.json"

ACTIONS = (
    "fisvdd_revision",
    "lucid_revision",
    "tinydl_revision",
    "oi_svdd_as_elm_revision",
)
POLICIES = ("FSSQL-R", "Safe-Greedy", "Threshold", "Static-Light")
CONDITIONS = ("n16/B0", "n16/B1", "n16/B2")
SEEDS = (5101, 5102, 5103, 5104, 5105)
COMMISSIONING_POLICY = "Static-Light"
COMMISSIONING_CONDITION = "n16/B0"
COMMISSIONING_SEED = 5099
BASE_RECORDS = 6000
RUNTIME_INPUT_KEYS = (
    "raw_features",
    "features",
    "risk_features",
    "backlog",
    "batch_indices",
)
SCORING_INPUT_KEYS = ("labels",)

REQUIRED_SHARED_ROLES = {
    "worker",
    "runtime_contract",
    "runtime_helper:action_pool_numpy",
    "runtime_helper:preprocessor_numpy",
    "runtime_helper:risk_proxy_numpy",
    "runtime_config",
    "model_hyperparameters",
    "validation_metrics",
    "q_config",
    "q_manifest",
    "preprocessor",
    "risk_proxy",
    "guard:latency_predictor",
    "guard:thermal_predictor",
    "guard:margins",
    "powerz:capture",
    "powerz:specification",
    *{f"model:{action}" for action in ACTIONS},
    *{f"q_table:{seed}" for seed in SEEDS},
}

FIXED_PACKAGE_PATHS = {
    "worker": "pi4b8_online_worker.py",
    "runtime_contract": "runtime/corrected_v5_contract.py",
    "runtime_helper:action_pool_numpy": "runtime/action_pool_numpy.py",
    "runtime_helper:preprocessor_numpy": "runtime/preprocessor_numpy.py",
    "runtime_helper:risk_proxy_numpy": "runtime/risk_proxy_numpy.py",
    "runtime_dependencies": "config/PI4B8_RUNTIME_DEPENDENCIES_V1.json",
    f"commissioning_batch_indices:{COMMISSIONING_SEED}": (
        f"inputs/commissioning/seed_{COMMISSIONING_SEED}.npy"
    ),
    "validation_metrics": "config/VALIDATION_METRICS.csv",
    "preprocessor": "preprocessing/preprocessor_portable.npz",
    "risk_proxy": "risk_proxy/risk_proxy.npz",
    "guard:latency_predictor": "guard/latency_predictor.csv",
    "guard:thermal_predictor": "guard/thermal_predictor.json",
    "guard:margins": "guard/guard_margins.json",
    **{f"model:{action}": f"models/{action}.npz" for action in ACTIONS},
    **{f"q_table:{seed}": f"q_tables/fssql_r_seed_{seed}.npz" for seed in SEEDS},
}

FORBIDDEN_PATH_MARKERS = (
    "official_test",
    "official-test",
    "pi3b",
    "near_cap",
    "near-cap",
    "formal_runtime_v4",
)

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(RuntimeError):
    """The corrected package contract is absent, ambiguous, or inconsistent."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"missing JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid JSON artifact: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"JSON artifact must contain an object: {path}")
    return value


def _relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{field} must be a nonempty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ContractError(f"{field} escapes the package/source root: {value!r}")
    return path.as_posix()


def _hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ContractError(f"{field} is not a lowercase SHA-256")
    return value


def _artifact_index(document: Mapping[str, Any], *, source: bool) -> dict[str, dict[str, Any]]:
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ContractError("manifest artifacts must be a nonempty list")
    by_role: dict[str, dict[str, Any]] = {}
    package_paths: set[str] = set()
    for index, raw in enumerate(artifacts):
        if not isinstance(raw, dict):
            raise ContractError(f"artifact {index} is not an object")
        role = raw.get("role")
        if not isinstance(role, str) or not role:
            raise ContractError(f"artifact {index} has no role")
        if role in by_role:
            raise ContractError(f"duplicate artifact role: {role}")
        package_path = _relative_path(raw.get("package_path"), f"artifact {role} package_path")
        frozen_path = FIXED_PACKAGE_PATHS.get(role)
        if frozen_path is not None and package_path != frozen_path:
            raise ContractError(
                f"artifact {role} must use package path {frozen_path}, got {package_path}"
            )
        if package_path in package_paths:
            raise ContractError(f"duplicate package path: {package_path}")
        package_paths.add(package_path)
        _hash(raw.get("sha256"), f"artifact {role} sha256")
        if source:
            _relative_path(raw.get("source_path"), f"artifact {role} source_path")
        lowered = f"{role}/{package_path}/{raw.get('source_path', '')}".lower()
        if any(marker in lowered for marker in FORBIDDEN_PATH_MARKERS):
            raise ContractError(f"forbidden historical/test artifact in corrected package: {role}")
        by_role[role] = raw
    missing = sorted(REQUIRED_SHARED_ROLES - set(by_role))
    if missing:
        raise ContractError(f"missing required artifact roles: {missing}")
    return by_role


def _cell_key(cell: Mapping[str, Any]) -> tuple[str, int]:
    condition = cell.get("condition")
    try:
        seed = int(cell.get("seed"))
    except (TypeError, ValueError) as error:
        raise ContractError(f"invalid cell seed: {cell.get('seed')!r}") from error
    if condition not in CONDITIONS or seed not in SEEDS:
        raise ContractError(f"unexpected corrected cell: {condition}/{seed}")
    return str(condition), seed


def _validate_cells(document: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]]) -> None:
    cells = document.get("cells")
    if not isinstance(cells, list):
        raise ContractError("manifest cells must be a list")
    expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    observed: set[tuple[str, int]] = set()
    shared_roles: dict[str, str] = {}
    batch_role_by_seed: dict[int, str] = {}
    for raw in cells:
        if not isinstance(raw, dict):
            raise ContractError("cell binding is not an object")
        key = _cell_key(raw)
        if key in observed:
            raise ContractError(f"duplicate cell binding: {key[0]}/{key[1]}")
        observed.add(key)
        runtime = raw.get("runtime")
        scoring = raw.get("scoring")
        if not isinstance(runtime, dict) or set(runtime) != set(RUNTIME_INPUT_KEYS):
            raise ContractError(f"cell {key} runtime roles are incomplete")
        if not isinstance(scoring, dict) or set(scoring) != set(SCORING_INPUT_KEYS):
            raise ContractError(f"cell {key} scoring roles are incomplete")
        runtime_roles = []
        for name in RUNTIME_INPUT_KEYS:
            role = runtime[name]
            if not isinstance(role, str) or role not in artifacts:
                raise ContractError(f"cell {key} has unknown runtime role {role!r}")
            runtime_roles.append(role)
            path = str(artifacts[role]["package_path"])
            if path.startswith("scoring/") or "label" in role.lower() or "label" in path.lower():
                raise ContractError(f"cell {key} leaks scoring data into runtime role {role}")
        label_role = scoring["labels"]
        if not isinstance(label_role, str) or label_role not in artifacts:
            raise ContractError(f"cell {key} has unknown scoring role {label_role!r}")
        label_path = str(artifacts[label_role]["package_path"])
        if not label_path.startswith("scoring/") or label_role in runtime_roles:
            raise ContractError(f"cell {key} labels are not physically isolated")
        for name in ("raw_features", "features", "risk_features", "backlog"):
            previous = shared_roles.setdefault(name, runtime[name])
            if runtime[name] != previous:
                raise ContractError(f"{name} must use one shared frozen base across all cells")
        previous_batch = batch_role_by_seed.setdefault(key[1], runtime["batch_indices"])
        if runtime["batch_indices"] != previous_batch:
            raise ContractError(
                f"seed {key[1]} must reuse identical batch indices across B0/B1/B2"
            )
        previous_labels = shared_roles.setdefault("labels", label_role)
        if label_role != previous_labels:
            raise ContractError("all cells must use one shared scoring-label base")
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ContractError(f"cell matrix mismatch: missing={missing}, extra={extra}")


def _validate_commissioning_cell(
    document: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]]
) -> None:
    required = document.get("commissioning_cell_required") is True
    cell = document.get("commissioning_cell")
    if not required:
        if cell is not None:
            raise ContractError("commissioning cell is present but not required")
        return
    if not isinstance(cell, dict):
        raise ContractError("required commissioning cell is missing")
    try:
        seed = int(cell.get("seed", -1))
    except (TypeError, ValueError) as error:
        raise ContractError("commissioning seed is invalid") from error
    if (
        cell.get("policy") != COMMISSIONING_POLICY
        or cell.get("condition") != COMMISSIONING_CONDITION
        or seed != COMMISSIONING_SEED
        or cell.get("scientific_use")
        != "COMMISSIONING_ONLY_EXCLUDED_FROM_FORMAL_STATISTICS"
    ):
        raise ContractError("commissioning identity differs from the frozen protocol")
    _hash(
        document.get("commissioning_preregistration_sha256"),
        "commissioning_preregistration_sha256",
    )
    if COMMISSIONING_SEED in SEEDS:
        raise ContractError("commissioning must not consume a formal matched seed")
    runtime = cell.get("runtime")
    scoring = cell.get("scoring")
    if not isinstance(runtime, dict) or set(runtime) != set(RUNTIME_INPUT_KEYS):
        raise ContractError("commissioning runtime roles are incomplete")
    if not isinstance(scoring, dict) or set(scoring) != set(SCORING_INPUT_KEYS):
        raise ContractError("commissioning scoring roles are incomplete")
    for name in RUNTIME_INPUT_KEYS:
        role = runtime[name]
        if not isinstance(role, str) or role not in artifacts:
            raise ContractError(f"commissioning has unknown runtime role {role!r}")
        path = str(artifacts[role]["package_path"])
        if path.startswith("scoring/") or "label" in role.lower() or "label" in path.lower():
            raise ContractError(f"commissioning leaks scoring data into runtime role {role}")
    label_role = scoring["labels"]
    if not isinstance(label_role, str) or label_role not in artifacts:
        raise ContractError(f"commissioning has unknown scoring role {label_role!r}")
    label_path = str(artifacts[label_role]["package_path"])
    if not label_path.startswith("scoring/") or label_role in runtime.values():
        raise ContractError("commissioning labels are not physically isolated")
    formal_reference = document["cells"][0]
    for name in ("raw_features", "features", "risk_features", "backlog"):
        if runtime[name] != formal_reference["runtime"][name]:
            raise ContractError(f"commissioning {name} must use the formal frozen base")
    if label_role != formal_reference["scoring"]["labels"]:
        raise ContractError("commissioning must use the frozen scoring-label base")
    formal_batches = {cell["runtime"]["batch_indices"] for cell in document["cells"]}
    if runtime["batch_indices"] in formal_batches:
        raise ContractError("commissioning must use a distinct non-formal batch schedule")


def validate_source_manifest(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if document.get("schema_version") != SOURCE_SCHEMA:
        raise ContractError("corrected source-manifest schema mismatch")
    if document.get("status") != SOURCE_STATUS:
        raise ContractError("corrected source manifest is not frozen")
    if document.get("official_test_access") != "forbidden":
        raise ContractError("corrected package must forbid official-test access")
    if (
        document.get("base_records") != BASE_RECORDS
        or document.get("windows_per_run") != 600
        or document.get("records_per_window") != 16
    ):
        raise ContractError("corrected package must freeze 6000 base rows and 600 x 16 windows")
    if document.get("conditions") != list(CONDITIONS) or document.get("seeds") != list(SEEDS):
        raise ContractError("condition/seed order differs from the corrected contract")
    commit = document.get("source_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ContractError("source_commit must be a full 40-character Git commit")
    artifacts = _artifact_index(document, source=True)
    if document.get("runtime_dependencies_required") is True and "runtime_dependencies" not in artifacts:
        raise ContractError("corrected source manifest requires runtime dependencies")
    _validate_cells(document, artifacts)
    _validate_commissioning_cell(document, artifacts)
    return artifacts


def validate_package_manifest(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if document.get("schema_version") != PACKAGE_SCHEMA:
        raise ContractError("corrected package-manifest schema mismatch")
    if document.get("status") != PACKAGE_STATUS:
        raise ContractError("corrected package is not materialized")
    if document.get("execution_authorized") is not False:
        raise ContractError("package materialization must not authorize execution")
    if document.get("official_test_access") != "forbidden":
        raise ContractError("corrected package does not forbid official-test access")
    _hash(document.get("source_manifest_sha256"), "source_manifest_sha256")
    if (
        document.get("base_records") != BASE_RECORDS
        or document.get("windows_per_run") != 600
        or document.get("records_per_window") != 16
    ):
        raise ContractError("materialized package is not 6000 base rows and 600 x 16 windows")
    if document.get("conditions") != list(CONDITIONS) or document.get("seeds") != list(SEEDS):
        raise ContractError("materialized package cell order mismatch")
    artifacts = _artifact_index(document, source=False)
    if document.get("runtime_dependencies_required") is True and "runtime_dependencies" not in artifacts:
        raise ContractError("corrected package requires runtime dependencies")
    _validate_cells(document, artifacts)
    _validate_commissioning_cell(document, artifacts)
    return artifacts


def validate_materialized_package(package_root: Path, manifest_path: Path) -> dict[str, Any]:
    package_root = package_root.resolve()
    manifest_path = manifest_path.resolve()
    if manifest_path != package_root / PACKAGE_MANIFEST_NAME:
        raise ContractError("package manifest must use the corrected V5 canonical path")
    document = load_json(manifest_path)
    artifacts = validate_package_manifest(document)
    source_manifest_path = package_root / SOURCE_MANIFEST_NAME
    if not source_manifest_path.is_file() or source_manifest_path.is_symlink():
        raise ContractError("materialized package is missing its immutable source manifest")
    if sha256(source_manifest_path) != document["source_manifest_sha256"]:
        raise ContractError("materialized source-manifest hash mismatch")
    source_document = load_json(source_manifest_path)
    source_artifacts = validate_source_manifest(source_document)
    shared_fields = (
        "source_commit",
        "runtime_dependencies_required",
        "commissioning_cell_required",
        "commissioning_preregistration_sha256",
        "base_records",
        "windows_per_run",
        "records_per_window",
        "conditions",
        "seeds",
        "cells",
        "commissioning_cell",
    )
    drifted = [
        field
        for field in shared_fields
        if document.get(field) != source_document.get(field)
    ]
    if drifted:
        raise ContractError(f"package/source manifest field drift: {drifted}")
    if set(artifacts) != set(source_artifacts):
        raise ContractError("package/source manifest artifact-role drift")
    for role, record in artifacts.items():
        source_record = source_artifacts[role]
        if (
            record["package_path"] != source_record["package_path"]
            or record["sha256"] != source_record["sha256"]
        ):
            raise ContractError(f"package/source manifest artifact drift: {role}")
    expected_files = {PACKAGE_MANIFEST_NAME, SOURCE_MANIFEST_NAME}
    for role, record in artifacts.items():
        relative = str(record["package_path"])
        expected_files.add(relative)
        path = package_root / relative
        if not path.is_file() or path.is_symlink():
            raise ContractError(f"missing or symlinked packaged artifact {role}: {path}")
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise ContractError(f"packaged byte-size mismatch for {role}: {path}")
        observed = sha256(path)
        if observed != record["sha256"]:
            raise ContractError(
                f"packaged hash mismatch for {role}: expected {record['sha256']}, "
                f"observed {observed}"
            )
    actual_files = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ContractError(
            f"package inventory mismatch: missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )
    return document


def find_cell(
    document: Mapping[str, Any], condition: str, seed: int, policy: str | None = None
) -> dict[str, Any]:
    candidates = list(document.get("cells", []))
    commissioning = document.get("commissioning_cell")
    if isinstance(commissioning, dict):
        candidates.append(commissioning)
    matching = [
        cell
        for cell in candidates
        if cell.get("condition") == condition and int(cell.get("seed", -1)) == seed
        and ("policy" not in cell or policy == cell.get("policy"))
    ]
    if len(matching) != 1:
        raise ContractError(f"no unique corrected cell binding for {condition}/{seed}")
    return matching[0]


def validate_worker_cli_binding(
    package_root: Path,
    manifest_path: Path,
    *,
    policy: str,
    condition: str,
    seed: int,
    supplied: Mapping[str, Path | None],
) -> dict[str, Any]:
    """Require every worker CLI path to equal its hash-validated manifest role."""
    document = validate_materialized_package(package_root, manifest_path)
    artifacts = _artifact_index(document, source=False)
    cell = find_cell(document, condition, seed, policy)
    root = package_root.resolve()

    def expected(role: str) -> Path:
        return (root / str(artifacts[role]["package_path"])).resolve()

    expected_paths = {
        "runtime_config": expected("runtime_config"),
        "model_hyperparameters": expected("model_hyperparameters"),
        "raw_features": expected(cell["runtime"]["raw_features"]),
        "features": expected(cell["runtime"]["features"]),
        "risk_features": expected(cell["runtime"]["risk_features"]),
        "backlog": expected(cell["runtime"]["backlog"]),
        "batch_indices": expected(cell["runtime"]["batch_indices"]),
        "labels": expected(cell["scoring"]["labels"]),
    }
    if policy == "FSSQL-R":
        expected_paths.update(
            {
                "q_config": expected("q_config"),
                "q_manifest": expected("q_manifest"),
                "q_table": expected(f"q_table:{seed}"),
            }
        )
    elif any(supplied.get(name) is not None for name in ("q_config", "q_manifest", "q_table")):
        raise ContractError("non-FSSQL policy must not receive Q artifacts")

    for name, expected_path in expected_paths.items():
        value = supplied.get(name)
        if value is None or Path(value).resolve() != expected_path:
            raise ContractError(
                f"worker {name} path is not manifest-bound: expected {expected_path}, got {value}"
            )
    if Path(supplied.get("labels") or "").resolve() in {
        expected_paths[name]
        for name in ("raw_features", "features", "risk_features", "backlog", "batch_indices")
    }:
        raise ContractError("scoring labels alias a runtime decision input")
    return document


def build_worker_argv(
    package_root: Path,
    manifest_path: Path,
    *,
    run_id: str,
    policy: str,
    condition: str,
    seed: int,
    output: Path,
    execution_root: Path | None = None,
) -> list[str]:
    """Validate the full package, then build the exact corrected worker CLI."""
    if not RUN_ID_RE.fullmatch(run_id):
        raise ContractError(f"invalid run_id: {run_id!r}")
    if policy not in POLICIES:
        raise ContractError(f"unsupported corrected policy: {policy}")
    document = validate_materialized_package(package_root, manifest_path)
    artifacts = _artifact_index(document, source=False)
    cell = find_cell(document, condition, seed, policy)

    root = package_root.resolve()
    execution_root = root if execution_root is None else Path(execution_root)
    output = Path(output)
    if output == execution_root or execution_root in output.parents:
        raise ContractError("run output must be outside the immutable package directory")

    def role_path(role: str) -> str:
        return str(execution_root / str(artifacts[role]["package_path"]))

    runtime = cell["runtime"]
    scoring = cell["scoring"]
    command = [
        "python3",
        "-B",
        role_path("worker"),
        "--asset-root",
        str(execution_root),
        "--package-manifest",
        str(execution_root / PACKAGE_MANIFEST_NAME),
        "--runtime-config",
        role_path("runtime_config"),
        "--model-hyperparameters",
        role_path("model_hyperparameters"),
        "--raw-features",
        role_path(runtime["raw_features"]),
        "--features",
        role_path(runtime["features"]),
        "--risk-features",
        role_path(runtime["risk_features"]),
        "--backlog",
        role_path(runtime["backlog"]),
        "--batch-indices",
        role_path(runtime["batch_indices"]),
        "--labels",
        role_path(scoring["labels"]),
        "--policy",
        policy,
        "--condition",
        condition,
        "--seed",
        str(seed),
        "--run-id",
        run_id,
        "--output",
        str(output),
        "--windows",
        "600",
    ]
    if policy == "FSSQL-R":
        command.extend(
            [
                "--q-config",
                role_path("q_config"),
                "--q-manifest",
                role_path("q_manifest"),
                "--q-table",
                role_path(f"q_table:{seed}"),
            ]
        )
    return command


CORRECTED_TIMING_FIELDS = (
    "preprocessing_ms",
    "risk_proxy_ms",
    "telemetry_ms",
    "state_construction_ms",
    "latency_guard_ms",
    "thermal_guard_ms",
    "admissible_set_ms",
    "guard_total_ms",
    "q_lookup_ms",
    "policy_selection_ms",
    "switching_overhead_ms",
    "detector_inference_ms",
    "post_action_telemetry_ms",
    "controller_ms",
    "e2e_ms",
    "scoring_ms",
    "q_update_ms",
    "replay_update_ms",
)


def _json_vector(row: Mapping[str, str], field: str, length: int) -> np.ndarray:
    try:
        value = np.asarray(json.loads(row[field]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid per-record field {field}") from error
    if value.shape != (length,):
        raise ContractError(f"{field} must contain exactly {length} records")
    return value


def validate_worker_output(
    path: Path,
    *,
    run_id: str,
    policy: str,
    condition: str,
    seed: int,
    start_ack: Mapping[str, Any],
    end_ack: Mapping[str, Any],
    expected_package_manifest_sha256: str,
    expected_q_table_sha256: str,
    minimum_frequency_mhz: float = 1350.0,
) -> dict[str, Any]:
    """Validate the corrected 600 x 16 output schema and independent scoring."""
    if not path.is_file():
        raise ContractError(f"missing corrected worker output: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 600:
        raise ContractError(f"expected 600 corrected windows, found {len(rows)}")
    if [int(row.get("window_id", -1)) for row in rows] != list(range(600)):
        raise ContractError("corrected window IDs are incomplete or out of order")
    for row in rows:
        if (
            row.get("run_id") != run_id
            or row.get("policy") != policy
            or row.get("condition") != condition
            or int(row.get("seed", -1)) != seed
        ):
            raise ContractError("corrected output identity mismatch")
        if int(row.get("records_processed", 0)) != 16:
            raise ContractError("corrected output did not process 16 records per window")
        for field in CORRECTED_TIMING_FIELDS:
            try:
                value = float(row[field])
            except (KeyError, ValueError) as error:
                raise ContractError(f"invalid corrected timing field: {field}") from error
            if not math.isfinite(value) or value < 0:
                raise ContractError(f"invalid corrected timing value: {field}={value}")
        scores = _json_vector(row, "prediction_scores", 16).astype(np.float64)
        predictions = _json_vector(row, "prediction_bits", 16).astype(np.float64)
        labels = _json_vector(row, "labels_scoring_only", 16).astype(np.float64)
        if (
            not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))
            or not np.isin(predictions, [0, 1]).all()
            or not np.isin(labels, [0, 1]).all()
        ):
            raise ContractError("invalid corrected prediction/label values")
        pred = predictions.astype(bool)
        truth = labels.astype(np.int8)
        expected = (
            int(np.sum(pred & (truth == 1))),
            int(np.sum(pred & (truth == 0))),
            int(np.sum(~pred & (truth == 1))),
            int(np.sum(~pred & (truth == 0))),
        )
        observed = tuple(int(row[name]) for name in ("tp", "fp", "fn", "tn"))
        if observed != expected or sum(observed) != 16:
            raise ContractError("corrected confusion counts are not independently reproducible")
        for field in ("throttled_start", "throttled_end"):
            try:
                value = int(row[field].split("=", 1)[1], 16)
            except (KeyError, IndexError, ValueError) as error:
                raise ContractError(f"malformed corrected throttle field: {field}") from error
            if value & 0xF:
                raise ContractError(f"current throttle/undervoltage event in {field}")
        frequency = min(
            float(row["cpu_frequency_start_mhz_min"]),
            float(row["cpu_frequency_end_mhz_min"]),
        )
        if not math.isfinite(frequency) or frequency < minimum_frequency_mhz:
            raise ContractError("corrected frequency invariant failed")
    observed_hash = sha256(path)
    if (
        start_ack.get("event") != "RUN_START_ACK"
        or start_ack.get("run_id") != run_id
        or int(start_ack.get("records_per_window", -1)) != 16
        or start_ack.get("package_manifest_sha256")
        != expected_package_manifest_sha256
        or start_ack.get("q_table_sha256", "") != expected_q_table_sha256
        or end_ack.get("package_manifest_sha256")
        != expected_package_manifest_sha256
        or not math.isfinite(float(start_ack.get("pi_monotonic", math.nan)))
        or not math.isfinite(float(end_ack.get("pi_monotonic", math.nan)))
        or float(start_ack["pi_monotonic"]) >= float(end_ack["pi_monotonic"])
        or
        end_ack.get("event") != "RUN_END_ACK"
        or end_ack.get("run_id") != run_id
        or int(end_ack.get("windows_completed", -1)) != 600
        or int(end_ack.get("records_completed", -1)) != 9600
        or end_ack.get("raw_artifact_sha256") != observed_hash
    ):
        raise ContractError("corrected RUN_END_ACK/output binding mismatch")
    return {"windows": 600, "records": 9600, "sha256": observed_hash}


def validate_power_boundary(
    meter_path: Path,
    handshake: Mapping[str, Any],
    *,
    run_id: str,
    minimum_roll_s: float = 5.0,
) -> dict[str, Any]:
    """Integrate only between Mac ACK receipts, without endpoint extension."""
    if handshake.get("run_id") != run_id:
        raise ContractError("POWER-Z handshake run_id mismatch")
    start_ack = handshake.get("pi_start_ack")
    end_ack = handshake.get("pi_end_ack")
    if not isinstance(start_ack, dict) or not isinstance(end_ack, dict):
        raise ContractError("POWER-Z handshake is missing Pi ACK payloads")
    if (
        start_ack.get("event") != "RUN_START_ACK"
        or end_ack.get("event") != "RUN_END_ACK"
        or start_ack.get("run_id") != run_id
        or end_ack.get("run_id") != run_id
    ):
        raise ContractError("POWER-Z ACK identity mismatch")
    order = np.asarray(
        [
            handshake.get("meter_capture_start_mac_monotonic"),
            handshake.get("start_request_mac_monotonic"),
            handshake.get("start_ack_receive_mac_monotonic"),
            handshake.get("end_ack_receive_mac_monotonic"),
            handshake.get("meter_capture_end_mac_monotonic"),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(order).all() or not (order[0] < order[1] <= order[2] < order[3] < order[4]):
        raise ContractError("invalid Mac monotonic ACK ordering")
    if float(start_ack.get("pi_monotonic", math.nan)) >= float(
        end_ack.get("pi_monotonic", math.nan)
    ):
        raise ContractError("invalid Pi monotonic ACK ordering")
    with meter_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ContractError("insufficient POWER-Z samples")
    try:
        times = np.asarray([float(row["sample_monotonic"]) for row in rows])
        power = np.asarray([float(row["power_W"]) for row in rows])
    except (KeyError, ValueError) as error:
        raise ContractError("malformed POWER-Z fields") from error
    if not np.isfinite(times).all() or not np.isfinite(power).all() or np.any(power < 0):
        raise ContractError("non-finite/negative POWER-Z sample")
    if not np.all(np.diff(times) > 0):
        raise ContractError("POWER-Z timestamps are not strictly monotonic")
    start, end = float(order[2]), float(order[3])
    if not times[0] <= start < end <= times[-1]:
        raise ContractError("Mac ACK boundaries fall outside POWER-Z samples")
    pre_roll, post_roll = start - times[0], times[-1] - end
    if pre_roll < minimum_roll_s or post_roll < minimum_roll_s:
        raise ContractError(
            f"insufficient POWER-Z roll: pre={pre_roll:.6f}, post={post_roll:.6f}"
        )
    interior = (times > start) & (times < end)
    integration_times = np.r_[start, times[interior], end]
    integration_power = np.interp(integration_times, times, power)
    energy_j = float(np.trapezoid(integration_power, integration_times))
    if not math.isfinite(energy_j) or energy_j < 0:
        raise ContractError("invalid POWER-Z energy integral")
    return {
        "samples": len(rows),
        "sha256": sha256(meter_path),
        "boundary": "start_ack_receive_mac_monotonic_to_end_ack_receive_mac_monotonic",
        "boundary_includes_post_decision_scoring_and_evidence_write": True,
        "start_request_to_ack_ms": (order[2] - order[1]) * 1000.0,
        "pre_roll_s": pre_roll,
        "post_roll_s": post_roll,
        "duration_s": end - start,
        "energy_J": energy_j,
        "mean_power_W": energy_j / (end - start),
        "peak_observed_with_interpolated_boundaries_W": float(integration_power.max()),
    }
