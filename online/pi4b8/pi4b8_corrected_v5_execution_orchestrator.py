#!/usr/bin/env python3
"""Mac-side execution boundary for the corrected Pi4B V5 package.

Commissioning may execute one explicitly selected cell.  Formal execution is
fail-closed behind a hash-bound corrected commissioning, frozen 60-cell plan,
and final execution manifest.  This file does not create that final
authorization manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TextIO


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from online.pi4b8.corrected_v5_contract import (  # noqa: E402
    COMMISSIONING_CONDITION,
    COMMISSIONING_POLICY,
    COMMISSIONING_SEED,
    PACKAGE_MANIFEST_NAME,
    ContractError,
    build_worker_argv,
    sha256,
    validate_materialized_package,
    validate_power_boundary,
    validate_worker_output,
)
from online.pi4b8.pi4b8_corrected_v5_orchestrator import (  # noqa: E402
    build_plan,
)


COMMISSIONING_VALIDATION = "PI4B8_CORRECTED_V5_COMMISSIONING_VALIDATION.json"
COMMISSIONING_DECISION = "PI4B8_CORRECTED_V5_COMMISSIONING_DECISION.md"
FORMAL_PLAN = "PI4B8_CORRECTED_V5_FORMAL_PLAN.json"
FINAL_MANIFEST = "PI4B8_CORRECTED_V5_FINAL_EXECUTION_MANIFEST.json"
RUNTIME_PREFLIGHT = "PI4B8_CORRECTED_V5_RUNTIME_PREFLIGHT.json"
COMMISSIONING_MARKER = "COMMISSIONING_ONLY_EXCLUDED_FROM_FORMAL_STATISTICS"
RUNTIME_DEPENDENCY_ROLE = "runtime_dependencies"
RUNTIME_DEPENDENCY_PATH = "config/PI4B8_RUNTIME_DEPENDENCIES_V1.json"
PRE_POST_ROLL_SECONDS = 5.0
METER_CAPTURE_SECONDS = 90.0
class ExecutionError(RuntimeError):
    """A corrected execution dependency or technical run condition failed."""


def validate_commissioning_identity(policy: str, condition: str, seed: int) -> None:
    """Keep commissioning outside the five formal matched seeds."""
    expected = (
        COMMISSIONING_POLICY,
        COMMISSIONING_CONDITION,
        COMMISSIONING_SEED,
    )
    if (policy, condition, seed) != expected:
        raise ExecutionError(
            "commissioning must use the frozen non-formal cell "
            f"{COMMISSIONING_POLICY}/{COMMISSIONING_CONDITION}/"
            f"seed-{COMMISSIONING_SEED}"
        )


def validate_commissioning_package_binding(
    package: dict[str, Any],
) -> dict[str, Any]:
    """Require the non-formal batch schedule to be an explicit package role."""
    cell = package.get("commissioning_cell")
    if not isinstance(cell, dict):
        raise ExecutionError("package lacks the frozen non-formal commissioning_cell")
    try:
        seed = int(cell.get("seed", -1))
    except (TypeError, ValueError) as error:
        raise ExecutionError("commissioning_cell has an invalid seed") from error
    validate_commissioning_identity(
        str(cell.get("policy", "")), str(cell.get("condition", "")), seed
    )
    runtime = cell.get("runtime")
    if not isinstance(runtime, dict):
        raise ExecutionError("commissioning_cell lacks runtime role bindings")
    batch_role = runtime.get("batch_indices")
    artifacts = _artifact_index(package)
    if not isinstance(batch_role, str) or batch_role not in artifacts:
        raise ExecutionError(
            "commissioning_cell batch_indices is not a hash-bound package role"
        )
    record = artifacts[batch_role]
    return {
        "batch_indices_role": batch_role,
        "batch_indices_sha256": str(record["sha256"]),
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExecutionError(f"missing required JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExecutionError(f"JSON artifact must contain an object: {path}")
    return value


def parse_ack(line: str, event: str, run_id: str) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise ExecutionError(f"malformed {event}: {line!r}") from error
    if value.get("event") != event or value.get("run_id") != run_id:
        raise ExecutionError(f"{event} run binding mismatch: {value}")
    for field in ("pi_utc", "pi_monotonic"):
        try:
            number = float(value[field])
        except (KeyError, TypeError, ValueError) as error:
            raise ExecutionError(f"invalid {event} {field}") from error
        if not math.isfinite(number):
            raise ExecutionError(f"non-finite {event} {field}")
    return value


def read_ack(
    stream: TextIO, event: str, run_id: str
) -> tuple[dict[str, Any], float, list[str]]:
    """Read through non-ACK diagnostics and timestamp the matching ACK line."""
    lines: list[str] = []
    while True:
        line = stream.readline()
        received = time.monotonic()
        if not line:
            raise ExecutionError(f"missing {event}")
        lines.append(line)
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if candidate.get("event") == event:
            return parse_ack(line, event, run_id), received, lines


def _artifact_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(record["role"]): record for record in manifest["artifacts"]}


def runtime_dependency_contract(
    package_root: Path, package: dict[str, Any]
) -> tuple[Path, dict[str, Any], str]:
    artifacts = _artifact_index(package)
    record = artifacts.get(RUNTIME_DEPENDENCY_ROLE)
    if not isinstance(record, dict):
        raise ExecutionError(
            "corrected package has no hash-bound runtime dependency manifest"
        )
    if record.get("package_path") != RUNTIME_DEPENDENCY_PATH:
        raise ExecutionError("runtime dependency manifest package path is not frozen")
    path = package_root / RUNTIME_DEPENDENCY_PATH
    dependency = load_json(path)
    if (
        dependency.get("status") != "FROZEN_BEFORE_CORRECTED_PI4B8_COMMISSIONING"
        or dependency.get("official_test_access") != "forbidden"
    ):
        raise ExecutionError("runtime dependency manifest is not frozen")
    pi_worker = dependency.get("pi_worker")
    mac_runtime = dependency.get("mac_orchestrator_and_powerz")
    if not isinstance(pi_worker, dict) or not isinstance(mac_runtime, dict):
        raise ExecutionError("runtime dependency manifest lacks Pi/Mac environments")
    required_fields = {
        "pi_worker": (
            "python_implementation",
            "python_version",
            "numpy_version",
            "system",
            "machine",
        ),
        "mac_runtime": (
            "python_implementation",
            "python_version",
            "numpy_version",
            "hidapi_version",
            "system",
            "machine",
            "python_executable",
        ),
    }
    for scope, value in (("pi_worker", pi_worker), ("mac_runtime", mac_runtime)):
        for field in required_fields[scope]:
            if not isinstance(value.get(field), str) or not value[field]:
                raise ExecutionError(f"runtime dependency manifest lacks {scope}.{field}")
    if pi_worker.get("cpu_governor_required") != "performance":
        raise ExecutionError("runtime dependency manifest does not freeze performance governor")
    try:
        minimum_frequency_mhz = float(pi_worker["minimum_frequency_mhz"])
    except (KeyError, TypeError, ValueError) as error:
        raise ExecutionError(
            "runtime dependency manifest lacks Pi minimum_frequency_mhz"
        ) from error
    if not math.isfinite(minimum_frequency_mhz) or minimum_frequency_mhz <= 0:
        raise ExecutionError("invalid Pi minimum_frequency_mhz")
    return path, dependency, str(record["sha256"])


def validate_remote_package_preflight(
    package_root: Path,
    *,
    remote_package_root: Path,
    ssh_target: str,
) -> dict[str, Any]:
    """Recompute the immutable remote package inventory before execution."""
    remote_package_root = Path(remote_package_root)
    if not remote_package_root.is_absolute():
        raise ExecutionError("remote package root must be absolute")
    script = r'''
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
manifest_path=root/"PI4B8_CORRECTED_RUNTIME_PACKAGE_MANIFEST_V5.json"
if not manifest_path.is_file():
    raise SystemExit("missing remote package manifest")
document=json.loads(manifest_path.read_text(encoding="utf-8"))
expected={
    "PI4B8_CORRECTED_RUNTIME_PACKAGE_MANIFEST_V5.json",
    "PI4B8_CORRECTED_RUNTIME_SOURCE_MANIFEST_V5.json",
}
for record in document.get("artifacts", []):
    relative=record["package_path"]
    expected.add(relative)
    path=root/relative
    if not path.is_file() or path.is_symlink():
        raise SystemExit("missing or symlinked remote artifact: "+relative)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["sha256"] or path.stat().st_size != int(record["bytes"]):
        raise SystemExit("remote artifact mismatch: "+relative)
actual={path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
if actual != expected:
    raise SystemExit("remote package inventory mismatch")
print(json.dumps({
    "status":"PASS",
    "remote_package_root":str(root),
    "package_manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "artifacts":len(document["artifacts"]),
    "files":len(actual),
},sort_keys=True))
'''
    result = subprocess.run(
        [
            "ssh",
            ssh_target,
            shlex.join(
                [
                    "env",
                    "PYTHONDONTWRITEBYTECODE=1",
                    "python3",
                    "-c",
                    script,
                    str(remote_package_root),
                ]
            ),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ExecutionError(
            f"remote package validation failed: {result.stdout}{result.stderr}".strip()
        )
    try:
        observed = json.loads(result.stdout.strip())
    except json.JSONDecodeError as error:
        raise ExecutionError("remote package validation returned malformed JSON") from error
    local_manifest = package_root / PACKAGE_MANIFEST_NAME
    if (
        observed.get("status") != "PASS"
        or observed.get("remote_package_root") != str(remote_package_root)
        or observed.get("package_manifest_sha256") != sha256(local_manifest)
    ):
        raise ExecutionError("remote package differs from the frozen local package")
    return observed


def validate_remote_runtime_preflight(
    package_root: Path,
    package: dict[str, Any],
    *,
    ssh_target: str,
) -> dict[str, Any]:
    """Check exact Python/NumPy and all four CPU governors without changing them."""
    dependency_path, dependency, dependency_sha256 = runtime_dependency_contract(
        package_root, package
    )
    script = (
        "import glob,json,pathlib,platform,numpy;"
        "paths=sorted(glob.glob('/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor'));"
        "print(json.dumps({'python_implementation':platform.python_implementation().lower(),"
        "'python_version':platform.python_version(),'numpy_version':numpy.__version__,"
        "'system':platform.system(),'machine':platform.machine(),'governor_paths':paths,"
        "'governors':[pathlib.Path(p).read_text().strip() for p in paths]}))"
    )
    result = subprocess.run(
        ["ssh", ssh_target, shlex.join(["python3", "-c", script])],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ExecutionError(
            f"remote runtime preflight command failed: {result.stdout}{result.stderr}".strip()
        )
    observed = None
    for line in reversed(result.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            observed = candidate
            break
    if observed is None:
        raise ExecutionError("remote runtime preflight returned no machine-readable result")
    expected = dependency["pi_worker"]
    paths = observed.get("governor_paths")
    governors = observed.get("governors")
    expected_paths = [
        f"/sys/devices/system/cpu/cpu{index}/cpufreq/scaling_governor"
        for index in range(4)
    ]
    if (
        paths != expected_paths
        or not isinstance(governors, list)
        or len(governors) != 4
        or any(value != "performance" for value in governors)
    ):
        raise ExecutionError(
            f"remote four-core performance-governor invariant failed: {governors}"
        )
    mismatched = [
        field
        for field in (
            "python_implementation",
            "python_version",
            "numpy_version",
            "system",
            "machine",
        )
        if observed.get(field) != expected[field]
    ]
    if mismatched:
        raise ExecutionError(
            "remote runtime dependency version mismatch: "
            + ", ".join(
                f"{field} expected={expected[field]} observed={observed.get(field)}"
                for field in mismatched
            )
        )
    return {
        "status": "PASS",
        "ssh_target": ssh_target,
        "dependency_manifest": RUNTIME_DEPENDENCY_PATH,
        "dependency_manifest_sha256": dependency_sha256,
        "dependency_manifest_local_sha256": sha256(dependency_path),
        "python_implementation": observed["python_implementation"],
        "python_version": observed["python_version"],
        "numpy_version": observed["numpy_version"],
        "system": observed["system"],
        "machine": observed["machine"],
        "governor_paths": paths,
        "governors": governors,
    }


def validate_mac_runtime_preflight(
    dependency: dict[str, Any], *, mac_python: str
) -> dict[str, Any]:
    expected = dependency["mac_orchestrator_and_powerz"]
    expected_executable = (ROOT / expected["python_executable"]).resolve()
    if (
        Path(mac_python).resolve() != expected_executable
        or Path(sys.executable).resolve() != expected_executable
    ):
        raise ExecutionError(
            f"Mac orchestrator must run under frozen interpreter {expected_executable}"
        )
    script = (
        "import importlib.metadata,json,platform,numpy;"
        "print(json.dumps({'python_implementation':platform.python_implementation().lower(),"
        "'python_version':platform.python_version(),'numpy_version':numpy.__version__,"
        "'hidapi_version':importlib.metadata.version('hidapi'),"
        "'system':platform.system(),'machine':platform.machine()}))"
    )
    result = subprocess.run(
        [mac_python, "-c", script], capture_output=True, text=True
    )
    if result.returncode:
        raise ExecutionError(
            f"Mac runtime preflight command failed: {result.stdout}{result.stderr}".strip()
        )
    try:
        observed = json.loads(result.stdout.strip())
    except json.JSONDecodeError as error:
        raise ExecutionError("Mac runtime preflight returned malformed JSON") from error
    fields = (
        "python_implementation",
        "python_version",
        "numpy_version",
        "hidapi_version",
        "system",
        "machine",
    )
    mismatched = [field for field in fields if observed.get(field) != expected[field]]
    if mismatched:
        raise ExecutionError(
            "Mac runtime dependency version mismatch: "
            + ", ".join(
                f"{field} expected={expected[field]} observed={observed.get(field)}"
                for field in mismatched
            )
        )
    return observed


def validate_runtime_preflight(
    package_root: Path,
    package: dict[str, Any],
    *,
    remote_package_root: Path,
    ssh_target: str,
    mac_python: str,
) -> dict[str, Any]:
    _, dependency, dependency_sha256 = runtime_dependency_contract(package_root, package)
    validate_commissioning_package_binding(package)
    remote_package = validate_remote_package_preflight(
        package_root,
        remote_package_root=remote_package_root,
        ssh_target=ssh_target,
    )
    mac = validate_mac_runtime_preflight(dependency, mac_python=mac_python)
    pi = validate_remote_runtime_preflight(
        package_root, package, ssh_target=ssh_target
    )
    return {
        "status": "PASS",
        "dependency_manifest": RUNTIME_DEPENDENCY_PATH,
        "dependency_manifest_sha256": dependency_sha256,
        "remote_package": remote_package,
        "mac_orchestrator_and_powerz": mac,
        "pi_worker": {
            "python_implementation": pi["python_implementation"],
            "python_version": pi["python_version"],
            "numpy_version": pi["numpy_version"],
            "system": pi["system"],
            "machine": pi["machine"],
            "governor_paths": pi["governor_paths"],
            "governors": pi["governors"],
        },
    }


def validate_frozen_plan(
    package_root: Path, manifest_path: Path, plan_path: Path
) -> dict[str, Any]:
    plan = load_json(plan_path)
    try:
        expected = build_plan(
            package_root,
            manifest_path,
            remote_package_root=Path(plan["remote_package_root"]),
            remote_output_root=Path(plan["remote_output_root"]),
        )
    except (KeyError, ContractError) as error:
        raise ExecutionError(f"invalid corrected formal plan: {error}") from error
    if plan != expected:
        raise ExecutionError("frozen corrected formal plan differs from the package contract")
    if len(plan.get("runs", [])) != 60:
        raise ExecutionError("corrected formal plan does not contain exactly 60 cells")
    return plan


def validate_formal_authorization(
    package_root: Path,
    manifest_path: Path,
    authorization_dir: Path,
    *,
    orchestrator_path: Path | None = None,
) -> dict[str, Any]:
    """Require a successful corrected commissioning and exact final hashes."""
    orchestrator_path = Path(__file__) if orchestrator_path is None else orchestrator_path
    try:
        package = validate_materialized_package(package_root, manifest_path)
    except ContractError as error:
        raise ExecutionError(f"corrected package gate failed: {error}") from error
    artifacts = _artifact_index(package)
    commissioning_binding = validate_commissioning_package_binding(package)

    validation_path = authorization_dir / COMMISSIONING_VALIDATION
    decision_path = authorization_dir / COMMISSIONING_DECISION
    preflight_path = authorization_dir / RUNTIME_PREFLIGHT
    plan_path = authorization_dir / FORMAL_PLAN
    final_path = authorization_dir / FINAL_MANIFEST
    validation = load_json(validation_path)
    preflight = load_json(preflight_path)
    final = load_json(final_path)
    plan = validate_frozen_plan(package_root, manifest_path, plan_path)

    if (
        validation.get("status") != "PASS"
        or validation.get("marker") != COMMISSIONING_MARKER
    ):
        raise ExecutionError("corrected commissioning validation is not PASS")
    try:
        validation_seed = int(validation.get("seed", -1))
    except (TypeError, ValueError) as error:
        raise ExecutionError("corrected commissioning validation seed is invalid") from error
    validate_commissioning_identity(
        str(validation.get("policy", "")),
        str(validation.get("condition", "")),
        validation_seed,
    )
    required_checks = {
        "windows_600",
        "records_9600",
        "worker_output_valid",
        "scoring_reproducible",
        "label_isolation",
        "powerz_valid",
        "meter_monotonic",
        "energy_reproducible",
        "run_id_binding",
        "hash_binding",
        "throttle_zero",
        "frequency_invariant",
        "governor_performance",
        "runtime_versions_exact",
    }
    checks = validation.get("checks", {})
    failed = sorted(name for name in required_checks if checks.get(name) is not True)
    if failed:
        raise ExecutionError(f"corrected commissioning checks failed: {failed}")
    if float(checks.get("pre_roll_s", 0.0)) < PRE_POST_ROLL_SECONDS or float(
        checks.get("post_roll_s", 0.0)
    ) < PRE_POST_ROLL_SECONDS:
        raise ExecutionError("corrected commissioning pre/post roll is below 5 seconds")

    if not decision_path.is_file():
        raise ExecutionError(f"missing corrected commissioning decision: {decision_path}")
    decision = decision_path.read_text(encoding="utf-8")
    run_id = str(validation.get("run_id", ""))
    if (
        "CORRECTED V5 COMMISSIONING PASS" not in decision
        or run_id not in decision
        or COMMISSIONING_MARKER not in decision
    ):
        raise ExecutionError("corrected commissioning decision does not match validation")

    if (
        final.get("status") != "PI4B8_CORRECTED_V5_FINAL_EXECUTION_MANIFEST"
        or final.get("execution_authorized") is not True
        or final.get("official_test_access") != "forbidden"
        or final.get("commissioning_run_id") != run_id
    ):
        raise ExecutionError("corrected final execution manifest is not authorized")

    _, dependency, dependency_sha256 = runtime_dependency_contract(package_root, package)
    expected_validation_lineage = {
        "package_manifest_sha256": sha256(manifest_path),
        "source_manifest_sha256": package["source_manifest_sha256"],
        "worker_sha256": artifacts["worker"]["sha256"],
        "orchestrator_sha256": sha256(orchestrator_path),
        "runtime_preflight_sha256": sha256(preflight_path),
        "runtime_dependencies_sha256": dependency_sha256,
        "commissioning_batch_indices_sha256": commissioning_binding[
            "batch_indices_sha256"
        ],
    }
    stale_validation = sorted(
        field
        for field, expected in expected_validation_lineage.items()
        if validation.get(field) != expected
    )
    if stale_validation:
        raise ExecutionError(
            f"corrected commissioning lineage mismatch: {stale_validation}"
        )

    expected_hashes = {
        "commissioning_validation_sha256": sha256(validation_path),
        "commissioning_decision_sha256": sha256(decision_path),
        "formal_plan_sha256": sha256(plan_path),
        "orchestrator_sha256": sha256(orchestrator_path),
        "runtime_package_manifest_sha256": sha256(manifest_path),
        "source_manifest_sha256": package["source_manifest_sha256"],
        "worker_sha256": artifacts["worker"]["sha256"],
        "powerz_capture_sha256": artifacts["powerz:capture"]["sha256"],
        "powerz_specification_sha256": artifacts["powerz:specification"]["sha256"],
        "runtime_preflight_sha256": sha256(preflight_path),
        "runtime_dependencies_sha256": dependency_sha256,
        "commissioning_batch_indices_sha256": commissioning_binding[
            "batch_indices_sha256"
        ],
    }
    expected_pi = dependency["pi_worker"]
    expected_mac = dependency["mac_orchestrator_and_powerz"]
    if (
        preflight.get("status") != "PASS"
        or preflight.get("dependency_manifest_sha256") != dependency_sha256
    ):
        raise ExecutionError("corrected commissioning runtime preflight is not valid")
    remote_package_preflight = preflight.get("remote_package", {})
    if (
        remote_package_preflight.get("status") != "PASS"
        or remote_package_preflight.get("remote_package_root")
        != plan["remote_package_root"]
        or remote_package_preflight.get("package_manifest_sha256")
        != sha256(manifest_path)
    ):
        raise ExecutionError("corrected commissioning remote package preflight is stale")
    pi_preflight = preflight.get("pi_worker", {})
    mac_preflight = preflight.get("mac_orchestrator_and_powerz", {})
    for field in (
        "python_implementation",
        "python_version",
        "numpy_version",
        "system",
        "machine",
    ):
        if pi_preflight.get(field) != expected_pi[field]:
            raise ExecutionError("corrected commissioning Pi runtime preflight is stale")
    for field in (
        "python_implementation",
        "python_version",
        "numpy_version",
        "hidapi_version",
        "system",
        "machine",
    ):
        if mac_preflight.get(field) != expected_mac[field]:
            raise ExecutionError("corrected commissioning Mac runtime preflight is stale")
    if pi_preflight.get("governor_paths") != [
        f"/sys/devices/system/cpu/cpu{index}/cpufreq/scaling_governor"
        for index in range(4)
    ] or pi_preflight.get("governors") != ["performance"] * 4:
        raise ExecutionError("corrected commissioning governor preflight is stale")
    mismatched = sorted(
        field for field, expected in expected_hashes.items() if final.get(field) != expected
    )
    if mismatched:
        raise ExecutionError(f"corrected final execution hash mismatch: {mismatched}")
    return {
        "status": "PI4B8_CORRECTED_V5_FINAL_FORMAL_GATE_PASS",
        "final_manifest_sha256": sha256(final_path),
        "formal_plan_sha256": sha256(plan_path),
        "runs": len(plan["runs"]),
        "plan": plan,
    }


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def execute_cell(
    package_root: Path,
    manifest_path: Path,
    run: dict[str, Any],
    *,
    evidence_root: Path,
    ssh_target: str,
    mac_python: str,
) -> dict[str, Any]:
    """Execute one contemporaneous Pi/KM003C cell and validate its evidence."""
    try:
        package = validate_materialized_package(package_root, manifest_path)
    except ContractError as error:
        raise ExecutionError(f"corrected package gate failed: {error}") from error
    artifacts = _artifact_index(package)
    _, dependency, _ = runtime_dependency_contract(package_root, package)
    run_id = str(run["run_id"])
    local_run = evidence_root / run_id
    local_run.mkdir(parents=True, exist_ok=False)
    meter_path = local_run / "POWERZ_raw.csv"
    pi_path = local_run / "pi_windows.csv"
    handshake_path = local_run / "HANDSHAKE.json"
    meter_stdout_path = local_run / "meter.stdout"
    meter_stderr_path = local_run / "meter.stderr"
    worker_stdout_path = local_run / "worker.stdout"
    worker_stderr_path = local_run / "worker.stderr"

    capture = package_root / artifacts["powerz:capture"]["package_path"]
    meter_start = time.monotonic()
    meter = subprocess.Popen(
        [
            mac_python,
            str(capture),
            "--output",
            str(meter_path),
            "--seconds",
            str(METER_CAPTURE_SECONDS),
            "--sps",
            "10",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    worker: subprocess.Popen[str] | None = None
    handshake: dict[str, Any] = {"run_id": run_id, "status": "INVALID_PENDING"}
    try:
        time.sleep(PRE_POST_ROLL_SECONDS)
        request_time = time.monotonic()
        worker = subprocess.Popen(
            ["ssh", ssh_target, shlex.join(run["worker_argv"])],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert worker.stdout is not None and worker.stderr is not None
        start_ack, start_ack_time, start_lines = read_ack(
            worker.stdout, "RUN_START_ACK", run_id
        )
        end_ack, end_ack_time, end_lines = read_ack(worker.stdout, "RUN_END_ACK", run_id)
        remaining_stdout, worker_stderr = worker.communicate()
        worker_stdout_path.write_text(
            "".join(start_lines + end_lines) + remaining_stdout, encoding="utf-8"
        )
        worker_stderr_path.write_text(worker_stderr, encoding="utf-8")
        if worker.returncode:
            raise ExecutionError(f"{run_id}: worker exited {worker.returncode}")

        transfer = subprocess.run(
            [
                "scp",
                f"{ssh_target}:{run['remote_output']}/windows.csv",
                str(pi_path),
            ],
            capture_output=True,
            text=True,
        )
        (local_run / "scp.stdout").write_text(transfer.stdout, encoding="utf-8")
        (local_run / "scp.stderr").write_text(transfer.stderr, encoding="utf-8")
        if transfer.returncode:
            raise ExecutionError(f"{run_id}: Pi evidence retrieval failed")

        time.sleep(PRE_POST_ROLL_SECONDS)
        try:
            meter_stdout, meter_stderr = meter.communicate(timeout=METER_CAPTURE_SECONDS + 15)
        except subprocess.TimeoutExpired as error:
            raise ExecutionError(f"{run_id}: POWER-Z capture did not terminate") from error
        meter_end = time.monotonic()
        meter_stdout_path.write_text(meter_stdout, encoding="utf-8")
        meter_stderr_path.write_text(meter_stderr, encoding="utf-8")
        if meter.returncode:
            raise ExecutionError(f"{run_id}: POWER-Z capture exited {meter.returncode}")

        handshake = {
            "run_id": run_id,
            "policy": run["policy"],
            "condition": run["condition"],
            "seed": run["seed"],
            "meter_capture_start_mac_monotonic": meter_start,
            "start_request_mac_monotonic": request_time,
            "start_ack_receive_mac_monotonic": start_ack_time,
            "end_ack_receive_mac_monotonic": end_ack_time,
            "meter_capture_end_mac_monotonic": meter_end,
            "pi_start_ack": start_ack,
            "pi_end_ack": end_ack,
            "status": "PENDING_VALIDATION",
        }
        handshake_path.write_text(
            json.dumps(handshake, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        pi_result = validate_worker_output(
            pi_path,
            run_id=run_id,
            policy=str(run["policy"]),
            condition=str(run["condition"]),
            seed=int(run["seed"]),
            start_ack=start_ack,
            end_ack=end_ack,
            expected_package_manifest_sha256=sha256(manifest_path),
            expected_q_table_sha256=str(run.get("q_table_sha256", "")),
            minimum_frequency_mhz=float(
                dependency["pi_worker"]["minimum_frequency_mhz"]
            ),
        )
        power_result = validate_power_boundary(meter_path, handshake, run_id=run_id)
        reproduced = validate_power_boundary(meter_path, handshake, run_id=run_id)
        if power_result["energy_J"] != reproduced["energy_J"]:
            raise ExecutionError(f"{run_id}: POWER-Z integration is not reproducible")

        handshake["status"] = "VALID_COMPLETE"
        handshake_path.write_text(
            json.dumps(handshake, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validation = {
            "status": "VALID_COMPLETE",
            "run_id": run_id,
            "pi": pi_result,
            "powerz": power_result,
            "handshake_sha256": sha256(handshake_path),
            "package_manifest_sha256": sha256(manifest_path),
        }
        _write_json_new(local_run / "RUN_VALIDATION.json", validation)
        run_manifest = {
            "run_id": run_id,
            "policy": run["policy"],
            "condition": run["condition"],
            "seed": run["seed"],
            "pi_raw": pi_path.name,
            "pi_raw_sha256": pi_result["sha256"],
            "powerz_raw": meter_path.name,
            "powerz_raw_sha256": power_result["sha256"],
            "handshake": handshake_path.name,
            "handshake_sha256": sha256(handshake_path),
            "validity_status": "VALID_COMPLETE",
            "package_manifest_sha256": sha256(manifest_path),
        }
        _write_json_new(local_run / "RUN_MANIFEST.json", run_manifest)
        return {
            "run_id": run_id,
            "policy": run["policy"],
            "condition": run["condition"],
            "seed": run["seed"],
            "pi": pi_result,
            "powerz": power_result,
            "run_manifest_sha256": sha256(local_run / "RUN_MANIFEST.json"),
        }
    except Exception as error:
        handshake["status"] = "INVALID_TECHNICAL"
        handshake["reason"] = str(error)
        handshake_path.write_text(
            json.dumps(handshake, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        invalid = {
            "status": "INVALID_TECHNICAL",
            "run_id": run_id,
            "reason": str(error),
        }
        validation_path = local_run / "RUN_VALIDATION.json"
        if not validation_path.exists():
            _write_json_new(validation_path, invalid)
        raise
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            worker.wait()
        if meter.poll() is None:
            meter.terminate()
            meter.wait()


def write_commissioning_artifacts(
    authorization_dir: Path,
    result: dict[str, Any],
    *,
    package_manifest_sha256: str,
    source_manifest_sha256: str,
    worker_sha256: str,
    orchestrator_sha256: str,
    runtime_preflight: dict[str, Any],
    commissioning_batch_indices_sha256: str,
) -> tuple[Path, Path]:
    validate_commissioning_identity(
        str(result.get("policy", "")),
        str(result.get("condition", "")),
        int(result.get("seed", -1)),
    )
    power = result["powerz"]
    pi = result["pi"]
    checks = {
        "windows_600": pi["windows"] == 600,
        "records_9600": pi["records"] == 9600,
        "worker_output_valid": True,
        "scoring_reproducible": True,
        "label_isolation": True,
        "powerz_valid": True,
        "meter_monotonic": True,
        "energy_reproducible": True,
        "run_id_binding": True,
        "hash_binding": True,
        "throttle_zero": True,
        "frequency_invariant": True,
        "governor_performance": runtime_preflight.get("pi_worker", {}).get(
            "governors"
        )
        == ["performance"] * 4,
        "runtime_versions_exact": runtime_preflight.get("status") == "PASS",
        "pre_roll_s": power["pre_roll_s"],
        "post_roll_s": power["post_roll_s"],
    }
    if not all(value is True for value in checks.values() if isinstance(value, bool)):
        raise ExecutionError("cannot freeze failed corrected commissioning")
    validation = {
        "status": "PASS",
        "marker": COMMISSIONING_MARKER,
        "run_id": result["run_id"],
        "policy": result["policy"],
        "condition": result["condition"],
        "seed": result["seed"],
        "checks": checks,
        "pi_raw_sha256": pi["sha256"],
        "powerz_raw_sha256": power["sha256"],
        "run_manifest_sha256": result["run_manifest_sha256"],
        "package_manifest_sha256": package_manifest_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "worker_sha256": worker_sha256,
        "orchestrator_sha256": orchestrator_sha256,
        "runtime_preflight_sha256": sha256(authorization_dir / RUNTIME_PREFLIGHT),
        "runtime_dependencies_sha256": runtime_preflight[
            "dependency_manifest_sha256"
        ],
        "commissioning_batch_indices_sha256": (
            commissioning_batch_indices_sha256
        ),
    }
    validation_path = authorization_dir / COMMISSIONING_VALIDATION
    decision_path = authorization_dir / COMMISSIONING_DECISION
    _write_json_new(validation_path, validation)
    authorization_dir.mkdir(parents=True, exist_ok=True)
    with decision_path.open("x", encoding="utf-8") as stream:
        stream.write("# Corrected Pi4B V5 commissioning\n\n")
        stream.write(f"CORRECTED V5 COMMISSIONING PASS: `{result['run_id']}`\n\n")
        stream.write(f"`{COMMISSIONING_MARKER}`\n")
    return validation_path, decision_path


def commissioning_run(args: argparse.Namespace) -> None:
    validate_commissioning_identity(args.policy, args.condition, args.seed)
    package = validate_materialized_package(args.package_root, args.package_manifest)
    artifacts = _artifact_index(package)
    commissioning_binding = validate_commissioning_package_binding(package)
    runtime_preflight = validate_runtime_preflight(
        args.package_root,
        package,
        remote_package_root=args.remote_package_root,
        ssh_target=args.ssh_target,
        mac_python=args.mac_python,
    )
    _write_json_new(args.authorization_dir / RUNTIME_PREFLIGHT, runtime_preflight)
    remote_output = args.remote_output_root / args.run_id
    argv = build_worker_argv(
        args.package_root,
        args.package_manifest,
        run_id=args.run_id,
        policy=args.policy,
        condition=args.condition,
        seed=args.seed,
        output=remote_output,
        execution_root=args.remote_package_root,
    )
    q_role = f"q_table:{args.seed}" if args.policy == "FSSQL-R" else None
    run = {
        "run_id": args.run_id,
        "policy": args.policy,
        "condition": args.condition,
        "seed": args.seed,
        "remote_output": str(remote_output),
        "worker_argv": argv,
        "q_table_sha256": "" if q_role is None else artifacts[q_role]["sha256"],
    }
    result = execute_cell(
        args.package_root,
        args.package_manifest,
        run,
        evidence_root=args.evidence_root,
        ssh_target=args.ssh_target,
        mac_python=args.mac_python,
    )
    validation_path, decision_path = write_commissioning_artifacts(
        args.authorization_dir,
        result,
        package_manifest_sha256=sha256(args.package_manifest),
        source_manifest_sha256=package["source_manifest_sha256"],
        worker_sha256=artifacts["worker"]["sha256"],
        orchestrator_sha256=sha256(Path(__file__)),
        runtime_preflight=runtime_preflight,
        commissioning_batch_indices_sha256=commissioning_binding[
            "batch_indices_sha256"
        ],
    )
    print("PI4B8_CORRECTED_V5_COMMISSIONING_PASS")
    print(f"validation={validation_path} decision={decision_path}")


def formal_run(args: argparse.Namespace) -> None:
    receipt = validate_formal_authorization(
        args.package_root, args.package_manifest, args.authorization_dir
    )
    if (
        str(args.remote_package_root) != receipt["plan"]["remote_package_root"]
        or str(args.remote_output_root) != receipt["plan"]["remote_output_root"]
    ):
        raise ExecutionError("formal CLI remote roots differ from the frozen plan")
    receipt_path = args.evidence_root / "FORMAL_GATE_RECEIPT.json"
    package = validate_materialized_package(args.package_root, args.package_manifest)
    live_preflight = validate_runtime_preflight(
        args.package_root,
        package,
        remote_package_root=args.remote_package_root,
        ssh_target=args.ssh_target,
        mac_python=args.mac_python,
    )
    live_preflight_path = args.evidence_root / "FORMAL_RUNTIME_PREFLIGHT.json"
    _write_json_new(live_preflight_path, live_preflight)
    _write_json_new(
        receipt_path,
        {
            **{key: value for key, value in receipt.items() if key != "plan"},
            "formal_runtime_preflight_sha256": sha256(live_preflight_path),
        },
    )
    for index, run in enumerate(receipt["plan"]["runs"], 1):
        execute_cell(
            args.package_root,
            args.package_manifest,
            run,
            evidence_root=args.evidence_root,
            ssh_target=args.ssh_target,
            mac_python=args.mac_python,
        )
        print(f"{index}/60 VALID_COMPLETE", flush=True)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--remote-package-root", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--authorization-dir", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--ssh-target", default="pi@pi4b8g.local")
    parser.add_argument("--mac-python", default=sys.executable)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    plan_parser = subparsers.add_parser("plan")
    add_common(plan_parser)

    commissioning_parser = subparsers.add_parser("commissioning")
    add_common(commissioning_parser)
    commissioning_parser.add_argument("--run-id", required=True)
    commissioning_parser.add_argument("--policy", required=True)
    commissioning_parser.add_argument("--condition", required=True)
    commissioning_parser.add_argument("--seed", type=int, required=True)

    formal_parser = subparsers.add_parser("formal")
    add_common(formal_parser)

    args = parser.parse_args()
    try:
        if args.mode == "plan":
            plan = build_plan(
                args.package_root,
                args.package_manifest,
                remote_package_root=args.remote_package_root,
                remote_output_root=args.remote_output_root,
            )
            plan_path = args.authorization_dir / FORMAL_PLAN
            _write_json_new(plan_path, plan)
            print("PI4B8_CORRECTED_V5_FORMAL_PLAN_FROZEN_NOT_AUTHORIZED")
            print(f"runs={len(plan['runs'])} sha256={sha256(plan_path)}")
        elif args.mode == "commissioning":
            commissioning_run(args)
        else:
            formal_run(args)
    except (ContractError, ExecutionError, OSError, ValueError) as error:
        raise SystemExit(f"PI4B8_CORRECTED_V5_EXECUTION_BLOCKED: {error}") from error


if __name__ == "__main__":
    main()
