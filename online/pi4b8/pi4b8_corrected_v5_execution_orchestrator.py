#!/usr/bin/env python3
"""Mac-side execution boundary for the corrected Pi4B V5 package.

Commissioning may execute one explicitly selected cell.  Formal execution is
fail-closed behind a hash-bound corrected commissioning, frozen 60-cell plan,
and final execution manifest.  A successful commissioning freezes that final
manifest and immediately reopens every authorization dependency for validation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import selectors
import shlex
import statistics
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
from online.pi4b8.powerz_evidence_validator import (  # noqa: E402
    validate_powerz_evidence,
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
METER_READY_TIMEOUT_SECONDS = 10.0
START_ACK_TIMEOUT_SECONDS = 30.0
END_ACK_TIMEOUT_SECONDS = 180.0
FORMAL_REGISTRY = "PI4B8_CORRECTED_V5_FORMAL_RUN_REGISTRY.jsonl"
FORMAL_COMPLETENESS = "PI4B8_CORRECTED_V5_FORMAL_EVIDENCE_COMPLETENESS.json"
COOLDOWN_EVIDENCE = "COOLDOWN_GATE.json"


class ExecutionError(RuntimeError):
    """A corrected execution dependency or technical run condition failed."""


def mac_monotonic() -> float:
    """Return the system-wide monotonic clock shared by Mac processes."""
    return time.clock_gettime(time.CLOCK_MONOTONIC)


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
    stream: TextIO,
    event: str,
    run_id: str,
    *,
    timeout_s: float,
    transcript: list[str] | None = None,
) -> tuple[dict[str, Any], float, list[str]]:
    """Read through diagnostics and timestamp a matching ACK before a deadline."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ExecutionError(f"invalid {event} timeout")
    lines: list[str] = []
    deadline = mac_monotonic() + timeout_s
    selector = selectors.DefaultSelector()
    try:
        selector.register(stream, selectors.EVENT_READ)
        while True:
            remaining = deadline - mac_monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise ExecutionError(f"timeout waiting for {event}")
            line = stream.readline()
            received = mac_monotonic()
            if not line:
                raise ExecutionError(f"missing {event}")
            lines.append(line)
            if transcript is not None:
                transcript.append(line)
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if candidate.get("event") == event:
                return parse_ack(line, event, run_id), received, lines
    finally:
        selector.close()


def read_powerz_ready(
    stream: TextIO,
    *,
    meter_path: Path,
    serial: str,
    timeout_s: float,
    transcript: list[str] | None = None,
) -> tuple[dict[str, Any], float, list[str]]:
    """Wait for the capture process's explicit post-HID-open readiness event."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ExecutionError("invalid POWERZ_READY timeout")
    lines: list[str] = []
    deadline = mac_monotonic() + timeout_s
    selector = selectors.DefaultSelector()
    try:
        selector.register(stream, selectors.EVENT_READ)
        while True:
            remaining = deadline - mac_monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise ExecutionError("timeout waiting for POWERZ_READY")
            line = stream.readline()
            received = mac_monotonic()
            if not line:
                raise ExecutionError("missing POWERZ_READY")
            lines.append(line)
            if transcript is not None:
                transcript.append(line)
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "POWERZ_READY":
                continue
            if event.get("serial") != serial:
                raise ExecutionError("POWERZ_READY meter serial mismatch")
            if Path(str(event.get("output", ""))).resolve() != meter_path.resolve():
                raise ExecutionError("POWERZ_READY output path mismatch")
            return event, received, lines
    finally:
        selector.close()


def validate_powerz_ready_binding(
    handshake: dict[str, Any], *, run_id: str, serial: str
) -> None:
    if handshake.get("run_id") != run_id:
        raise ExecutionError("POWERZ_READY handshake run_id mismatch")
    ready = handshake.get("powerz_ready")
    if not isinstance(ready, dict) or ready.get("event") != "POWERZ_READY":
        raise ExecutionError("POWERZ_READY event is missing")
    if ready.get("serial") != serial:
        raise ExecutionError("POWERZ_READY serial mismatch")
    try:
        meter_start = float(handshake["meter_capture_start_mac_monotonic"])
        ready_time = float(handshake["powerz_ready_receive_mac_monotonic"])
        request_time = float(handshake["start_request_mac_monotonic"])
    except (KeyError, TypeError, ValueError) as error:
        raise ExecutionError("POWERZ_READY timing is malformed") from error
    if not all(math.isfinite(value) for value in (meter_start, ready_time, request_time)):
        raise ExecutionError("POWERZ_READY timing is non-finite")
    if not meter_start < ready_time <= request_time:
        raise ExecutionError("POWERZ_READY timing order is invalid")
    if request_time - ready_time < PRE_POST_ROLL_SECONDS:
        raise ExecutionError("POWERZ_READY pre-roll is below 5 seconds")


def _artifact_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(record["role"]): record for record in manifest["artifacts"]}


def validate_local_validation_code_binding(package: dict[str, Any]) -> dict[str, str]:
    """Bind the Mac-imported validators to the exact packaged source bytes."""
    artifacts = _artifact_index(package)
    local_paths = {
        "runtime_contract": ROOT / "online/pi4b8/corrected_v5_contract.py",
        "powerz:validator": ROOT / "online/pi4b8/powerz_evidence_validator.py",
    }
    result: dict[str, str] = {}
    for role, path in local_paths.items():
        record = artifacts.get(role)
        if not isinstance(record, dict) or not path.is_file():
            raise ExecutionError(f"missing local validation-code binding: {role}")
        observed = sha256(path)
        if observed != record.get("sha256"):
            raise ExecutionError(f"local validation code differs from package: {role}")
        result[role] = observed
    return result


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
    identity = dependency.get("device_identity")
    if not isinstance(identity, dict):
        raise ExecutionError("runtime dependency manifest lacks device_identity")
    for field in ("hostname", "model", "cpu_serial"):
        if not isinstance(identity.get(field), str) or not identity[field]:
            raise ExecutionError(
                f"runtime dependency manifest lacks device_identity.{field}"
            )
    try:
        mem_total_kib = int(identity["mem_total_kib"])
    except (KeyError, TypeError, ValueError) as error:
        raise ExecutionError(
            "runtime dependency manifest lacks device_identity.mem_total_kib"
        ) from error
    if mem_total_kib <= 0:
        raise ExecutionError("invalid device_identity.mem_total_kib")
    cooldown = dependency.get("cooldown")
    if not isinstance(cooldown, dict):
        raise ExecutionError("runtime dependency manifest lacks cooldown")
    try:
        idle_samples = int(cooldown["idle_samples"])
        interval_s = float(cooldown["idle_interval_s"])
        max_above_median = float(
            cooldown["maximum_start_above_idle_median_c"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ExecutionError("runtime dependency manifest has invalid cooldown") from error
    if (
        idle_samples != 30
        or interval_s != 1.0
        or max_above_median != 2.0
    ):
        raise ExecutionError(
            "runtime dependency manifest must freeze cooldown at 30x1 s and +2 C"
        )
    powerz_device = dependency.get("powerz_device")
    if not isinstance(powerz_device, dict):
        raise ExecutionError("runtime dependency manifest lacks powerz_device")
    if powerz_device.get("serial") != "075356":
        raise ExecutionError("runtime dependency manifest has unexpected POWER-Z serial")
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
        "import glob,json,pathlib,platform,numpy,socket;"
        "paths=sorted(glob.glob('/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor'));"
        "cpuinfo=pathlib.Path('/proc/cpuinfo').read_text();"
        "meminfo=pathlib.Path('/proc/meminfo').read_text();"
        "serial=next(line.split(':',1)[1].strip() for line in cpuinfo.splitlines() if line.startswith('Serial'));"
        "mem_kib=int(next(line.split()[1] for line in meminfo.splitlines() if line.startswith('MemTotal:')));"
        "print(json.dumps({'python_implementation':platform.python_implementation().lower(),"
        "'python_version':platform.python_version(),'numpy_version':numpy.__version__,"
        "'system':platform.system(),'machine':platform.machine(),'governor_paths':paths,"
        "'governors':[pathlib.Path(p).read_text().strip() for p in paths],"
        "'hostname':socket.gethostname(),"
        "'model':pathlib.Path('/proc/device-tree/model').read_text().rstrip(chr(0)),"
        "'cpu_serial':serial,'mem_total_kib':mem_kib,"
        "'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))"
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
    identity = dependency["device_identity"]
    identity_mismatches = [
        field
        for field in ("hostname", "model", "cpu_serial", "mem_total_kib")
        if observed.get(field) != identity[field]
    ]
    if identity_mismatches:
        raise ExecutionError(
            "remote device identity mismatch: "
            + ", ".join(
                f"{field} expected={identity[field]} observed={observed.get(field)}"
                for field in identity_mismatches
            )
        )
    if not isinstance(observed.get("boot_id"), str) or not observed["boot_id"]:
        raise ExecutionError("remote runtime preflight lacks boot_id")
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
        "device_identity": {
            field: observed[field]
            for field in ("hostname", "model", "cpu_serial", "mem_total_kib")
        },
        "boot_id": observed["boot_id"],
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
    local_validation_code = validate_local_validation_code_binding(package)
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
        "local_validation_code": local_validation_code,
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
            "device_identity": pi["device_identity"],
            "boot_id": pi["boot_id"],
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
        "cooldown_gate",
        "device_identity_bound",
        "actual_start_temperature_gate",
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
        "powerz_validator_sha256": artifacts["powerz:validator"]["sha256"],
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
        "powerz_validator_sha256": artifacts["powerz:validator"]["sha256"],
        "powerz_specification_sha256": artifacts["powerz:specification"]["sha256"],
        "runtime_preflight_sha256": sha256(preflight_path),
        "runtime_dependencies_sha256": dependency_sha256,
        "commissioning_batch_indices_sha256": commissioning_binding[
            "batch_indices_sha256"
        ],
        "commissioning_pi_raw_sha256": validation.get("pi_raw_sha256"),
        "commissioning_powerz_raw_sha256": validation.get("powerz_raw_sha256"),
        "commissioning_handshake_sha256": validation.get("handshake_sha256"),
        "commissioning_cooldown_gate_sha256": validation.get(
            "cooldown_gate_sha256"
        ),
        "commissioning_run_manifest_sha256": validation.get(
            "run_manifest_sha256"
        ),
    }
    expected_pi = dependency["pi_worker"]
    expected_mac = dependency["mac_orchestrator_and_powerz"]
    expected_local_validation_code = {
        role: artifacts[role]["sha256"]
        for role in ("runtime_contract", "powerz:validator")
    }
    if (
        preflight.get("status") != "PASS"
        or preflight.get("dependency_manifest_sha256") != dependency_sha256
        or preflight.get("local_validation_code")
        != expected_local_validation_code
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
    expected_static_identity = {
        field: dependency["device_identity"][field]
        for field in ("hostname", "model", "cpu_serial", "mem_total_kib")
    }
    if pi_preflight.get("device_identity") != expected_static_identity:
        raise ExecutionError("corrected commissioning device identity is stale")
    if not isinstance(pi_preflight.get("boot_id"), str) or not pi_preflight[
        "boot_id"
    ]:
        raise ExecutionError("corrected commissioning boot_id is missing")
    try:
        evidence_directory = Path(validation["evidence_directory"]).resolve()
    except (KeyError, TypeError) as error:
        raise ExecutionError(
            "corrected commissioning evidence directory is missing"
        ) from error
    expected_paths = {
        "pi_raw_path": evidence_directory / "pi_windows.csv",
        "powerz_raw_path": evidence_directory / "POWERZ_raw.csv",
        "handshake_path": evidence_directory / "HANDSHAKE.json",
        "cooldown_gate_path": evidence_directory / COOLDOWN_EVIDENCE,
        "run_manifest_path": evidence_directory / "RUN_MANIFEST.json",
    }
    for field, path in expected_paths.items():
        if validation.get(field) != str(path):
            raise ExecutionError(
                f"corrected commissioning evidence path mismatch: {field}"
            )
    reopened = revalidate_complete_cell(
        evidence_directory,
        {
            "run_id": run_id,
            "policy": validation["policy"],
            "condition": validation["condition"],
            "seed": validation_seed,
            "q_table_sha256": "",
        },
        package_root=package_root,
        manifest_path=manifest_path,
        dependency=dependency,
    )
    reopened_hashes = {
        "pi_raw_sha256": reopened["pi_raw_sha256"],
        "powerz_raw_sha256": reopened["powerz_raw_sha256"],
        "handshake_sha256": reopened["handshake_sha256"],
        "cooldown_gate_sha256": reopened["cooldown_gate_sha256"],
        "run_manifest_sha256": reopened["run_manifest_sha256"],
    }
    stale_raw = [
        field
        for field, digest in reopened_hashes.items()
        if validation.get(field) != digest
    ]
    if stale_raw:
        raise ExecutionError(
            f"corrected commissioning raw evidence mismatch: {stale_raw}"
        )
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


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def validate_cooldown_payload(
    identity: Any, samples: Any, dependency: dict[str, Any]
) -> dict[str, Any]:
    cooldown = dependency["cooldown"]
    count = int(cooldown["idle_samples"])
    interval_s = float(cooldown["idle_interval_s"])
    maximum_above_median_c = float(
        cooldown["maximum_start_above_idle_median_c"]
    )
    expected_identity = dependency["device_identity"]
    if not isinstance(identity, dict):
        raise ExecutionError("idle cooldown telemetry lacks device identity")
    for field in ("hostname", "model", "cpu_serial", "mem_total_kib"):
        if identity.get(field) != expected_identity[field]:
            raise ExecutionError(f"idle cooldown device identity mismatch: {field}")
    if not isinstance(identity.get("boot_id"), str) or not identity["boot_id"]:
        raise ExecutionError("idle cooldown telemetry lacks boot_id")
    if not isinstance(samples, list) or len(samples) != count:
        raise ExecutionError(f"idle cooldown requires exactly {count} samples")
    minimum_frequency = float(dependency["pi_worker"]["minimum_frequency_mhz"])
    temperatures: list[float] = []
    monotonic_values: list[float] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or sample.get("sample_index") != index:
            raise ExecutionError("idle cooldown sample order mismatch")
        try:
            temperature = float(sample["temperature_c"])
            monotonic = float(sample["pi_monotonic"])
            frequencies = [float(value) for value in sample["frequency_mhz"]]
        except (KeyError, TypeError, ValueError) as error:
            raise ExecutionError("idle cooldown sample is malformed") from error
        if (
            not math.isfinite(temperature)
            or not math.isfinite(monotonic)
            or len(frequencies) != 4
            or not all(math.isfinite(value) for value in frequencies)
        ):
            raise ExecutionError("idle cooldown sample contains non-finite values")
        if sample.get("governors") != ["performance"] * 4:
            raise ExecutionError("idle cooldown governor invariant failed")
        if sample.get("throttled") != "throttled=0x0" or sample.get(
            "throttle_returncode"
        ) != 0:
            raise ExecutionError("idle cooldown throttle invariant failed")
        if min(frequencies) < minimum_frequency:
            raise ExecutionError("idle cooldown frequency invariant failed")
        temperatures.append(temperature)
        monotonic_values.append(monotonic)
    if any(
        later <= earlier
        for earlier, later in zip(monotonic_values, monotonic_values[1:])
    ):
        raise ExecutionError("idle cooldown timestamps are not strictly monotonic")
    observed_duration_s = monotonic_values[-1] - monotonic_values[0]
    if observed_duration_s < (count - 1) * interval_s - 0.1:
        raise ExecutionError("idle cooldown did not span the frozen 30x1 s interval")
    idle_median_c = float(statistics.median(temperatures))
    start_temperature_c = temperatures[-1]
    if start_temperature_c > idle_median_c + maximum_above_median_c:
        raise ExecutionError(
            "idle cooldown start temperature exceeds frozen median allowance"
        )
    return {
        "status": "PASS",
        "configuration": {
            "idle_samples": count,
            "idle_interval_s": interval_s,
            "maximum_start_above_idle_median_c": maximum_above_median_c,
        },
        "device_identity": identity,
        "idle_median_temperature_c": idle_median_c,
        "start_temperature_c": start_temperature_c,
        "observed_idle_duration_s": observed_duration_s,
        "samples": samples,
    }


def validate_actual_run_start_temperature(
    pi_result: dict[str, Any],
    cooldown: dict[str, Any],
    dependency: dict[str, Any],
) -> float:
    try:
        actual = float(pi_result["first_window_temperature_start_C"])
        median = float(cooldown["idle_median_temperature_c"])
        allowance = float(
            dependency["cooldown"]["maximum_start_above_idle_median_c"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ExecutionError("actual first-window start temperature is missing") from error
    if not all(math.isfinite(value) for value in (actual, median, allowance)):
        raise ExecutionError("actual first-window start temperature is non-finite")
    if actual > median + allowance:
        raise ExecutionError(
            "actual first-window temperature exceeds frozen idle-median allowance"
        )
    return actual


def collect_idle_cooldown_gate(
    dependency: dict[str, Any],
    *,
    ssh_target: str,
    evidence_dir: Path,
) -> dict[str, Any]:
    """Collect and validate the frozen 30x1 s pre-cell idle evidence."""
    cooldown = dependency["cooldown"]
    count = int(cooldown["idle_samples"])
    interval_s = float(cooldown["idle_interval_s"])
    script = r'''
import datetime,json,pathlib,socket,subprocess,sys,time
count=int(sys.argv[1]); interval=float(sys.argv[2])
cpuinfo=pathlib.Path('/proc/cpuinfo').read_text()
meminfo=pathlib.Path('/proc/meminfo').read_text()
identity={
 "hostname":socket.gethostname(),
 "model":pathlib.Path('/proc/device-tree/model').read_text().rstrip(chr(0)),
 "cpu_serial":next(line.split(':',1)[1].strip() for line in cpuinfo.splitlines() if line.startswith('Serial')),
 "mem_total_kib":int(next(line.split()[1] for line in meminfo.splitlines() if line.startswith('MemTotal:'))),
 "boot_id":pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
}
governor_paths=[pathlib.Path(f'/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor') for i in range(4)]
frequency_paths=[pathlib.Path(f'/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq') for i in range(4)]
samples=[]; base=time.monotonic()
for index in range(count):
 if index:
  time.sleep(max(0.0,base+index*interval-time.monotonic()))
 throttle=subprocess.run(['vcgencmd','get_throttled'],capture_output=True,text=True,check=False)
 samples.append({
  "sample_index":index,
  "pi_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
  "pi_monotonic":time.monotonic(),
  "temperature_c":float(pathlib.Path('/sys/class/thermal/thermal_zone0/temp').read_text().strip())/1000.0,
  "frequency_mhz":[float(path.read_text().strip())/1000.0 for path in frequency_paths],
  "governors":[path.read_text().strip() for path in governor_paths],
  "throttled":throttle.stdout.strip(),
  "throttle_returncode":throttle.returncode,
 })
print(json.dumps({"identity":identity,"samples":samples},sort_keys=True))
'''
    completed = subprocess.run(
        [
            "ssh",
            ssh_target,
            shlex.join(["python3", "-c", script, str(count), str(interval_s)]),
        ],
        capture_output=True,
        text=True,
    )
    (evidence_dir / "cooldown.stdout").write_text(
        completed.stdout, encoding="utf-8"
    )
    (evidence_dir / "cooldown.stderr").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode:
        raise ExecutionError("idle cooldown telemetry command failed")
    observed = None
    for line in reversed(completed.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            observed = candidate
            break
    if observed is None:
        raise ExecutionError("idle cooldown telemetry returned malformed JSON")
    receipt = validate_cooldown_payload(
        observed.get("identity"), observed.get("samples"), dependency
    )
    _write_json_new(evidence_dir / COOLDOWN_EVIDENCE, receipt)
    return receipt


def _artifact_inventory(root: Path, *, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = set() if exclude is None else exclude
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def _stop_and_collect(process: subprocess.Popen[str] | None) -> tuple[str, str]:
    if process is None:
        return "", ""
    if process.poll() is None:
        process.terminate()
    try:
        return process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.communicate()


def recover_remote_partial(
    *,
    ssh_target: str,
    remote_output: str,
    local_run: Path,
) -> dict[str, Any]:
    """Inventory and copy existing run files without changing the remote attempt."""
    listing_stdout = local_run / "remote_partial_listing.stdout"
    listing_stderr = local_run / "remote_partial_listing.stderr"
    script = r'''
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
records=[]
if root.is_dir():
 for path in sorted(root.rglob('*')):
  if path.is_file() and not path.is_symlink():
   relative=path.relative_to(root).as_posix()
   records.append({"path":relative,"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
print(json.dumps({"root":str(root),"files":records},sort_keys=True))
'''
    completed = subprocess.run(
        [
            "ssh",
            ssh_target,
            shlex.join(["python3", "-c", script, remote_output]),
        ],
        capture_output=True,
        text=True,
    )
    listing_stdout.write_text(completed.stdout, encoding="utf-8")
    listing_stderr.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        return {"status": "LISTING_FAILED", "files": []}
    listing = None
    for line in reversed(completed.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            listing = candidate
            break
    if listing is None:
        return {"status": "LISTING_MALFORMED", "files": []}
    records = listing.get("files")
    if not isinstance(records, list):
        return {"status": "LISTING_MALFORMED", "files": []}
    recovered: list[dict[str, Any]] = []
    transfers: list[dict[str, Any]] = []
    partial_root = local_run / "remote_partial"
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            continue
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            continue
        target = partial_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        transfer = subprocess.run(
            ["scp", f"{ssh_target}:{remote_output}/{relative.as_posix()}", str(target)],
            capture_output=True,
            text=True,
        )
        transfers.append(
            {
                "path": relative.as_posix(),
                "returncode": transfer.returncode,
                "stdout": transfer.stdout,
                "stderr": transfer.stderr,
            }
        )
        if (
            transfer.returncode == 0
            and target.is_file()
            and sha256(target) == record.get("sha256")
            and target.stat().st_size == int(record.get("bytes", -1))
        ):
            recovered.append(record)
    receipt = {
        "status": "PASS" if len(recovered) == len(records) else "PARTIAL",
        "remote_output": remote_output,
        "remote_files": records,
        "recovered_files": recovered,
        "transfers": transfers,
    }
    _write_json_new(local_run / "REMOTE_PARTIAL_RECOVERY.json", receipt)
    return receipt


def _append_registry(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"utc": _utc_now(), **record}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def revalidate_complete_cell(
    local_run: Path,
    run: dict[str, Any],
    *,
    package_root: Path,
    manifest_path: Path,
    dependency: dict[str, Any],
) -> dict[str, Any]:
    """Reopen raw evidence and independently revalidate a completed cell."""
    run_id = str(run["run_id"])
    manifest_file = local_run / "RUN_MANIFEST.json"
    validation_file = local_run / "RUN_VALIDATION.json"
    handshake_file = local_run / "HANDSHAKE.json"
    cooldown_file = local_run / COOLDOWN_EVIDENCE
    pi_file = local_run / "pi_windows.csv"
    meter_file = local_run / "POWERZ_raw.csv"
    manifest = load_json(manifest_file)
    validation = load_json(validation_file)
    handshake = load_json(handshake_file)
    cooldown = load_json(cooldown_file)
    expected = {
        "run_id": run_id,
        "policy": run["policy"],
        "condition": run["condition"],
        "seed": run["seed"],
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ExecutionError(f"{run_id}: run manifest {field} mismatch")
    if (
        manifest.get("validity_status") != "VALID_COMPLETE"
        or validation.get("status") != "VALID_COMPLETE"
        or validation.get("run_id") != run_id
    ):
        raise ExecutionError(f"{run_id}: accepted evidence is not VALID_COMPLETE")
    if (
        manifest.get("pi_raw") != pi_file.name
        or manifest.get("powerz_raw") != meter_file.name
        or manifest.get("handshake") != handshake_file.name
        or manifest.get("cooldown_gate") != cooldown_file.name
    ):
        raise ExecutionError(f"{run_id}: run manifest artifact path mismatch")
    expected_hashes = {
        "pi_raw_sha256": sha256(pi_file),
        "powerz_raw_sha256": sha256(meter_file),
        "handshake_sha256": sha256(handshake_file),
        "cooldown_gate_sha256": sha256(cooldown_file),
        "package_manifest_sha256": sha256(manifest_path),
    }
    stale = [
        field for field, digest in expected_hashes.items() if manifest.get(field) != digest
    ]
    if stale:
        raise ExecutionError(f"{run_id}: run manifest hash mismatch: {stale}")
    if manifest.get("artifacts") != _artifact_inventory(
        local_run, exclude={"RUN_MANIFEST.json"}
    ):
        raise ExecutionError(f"{run_id}: run manifest inventory mismatch")
    reproduced_cooldown = validate_cooldown_payload(
        cooldown.get("device_identity"), cooldown.get("samples"), dependency
    )
    if cooldown != reproduced_cooldown:
        raise ExecutionError(f"{run_id}: cooldown evidence is not reproducible")
    identity = reproduced_cooldown["device_identity"]
    if handshake.get("device_identity") != identity:
        raise ExecutionError(f"{run_id}: handshake device identity mismatch")
    validate_powerz_ready_binding(
        handshake,
        run_id=run_id,
        serial=str(dependency["powerz_device"]["serial"]),
    )
    pi_result = validate_worker_output(
        pi_file,
        package_root=package_root,
        package_manifest_path=manifest_path,
        run_id=run_id,
        policy=str(run["policy"]),
        condition=str(run["condition"]),
        seed=int(run["seed"]),
        start_ack=handshake.get("pi_start_ack", {}),
        end_ack=handshake.get("pi_end_ack", {}),
        expected_package_manifest_sha256=sha256(manifest_path),
        expected_q_table_sha256=str(run.get("q_table_sha256", "")),
        minimum_frequency_mhz=float(
            dependency["pi_worker"]["minimum_frequency_mhz"]
        ),
    )
    validate_actual_run_start_temperature(pi_result, cooldown, dependency)
    contract_power = validate_power_boundary(meter_file, handshake, run_id=run_id)
    power_result = validate_powerz_evidence(meter_file, handshake, run_id=run_id)
    if not math.isclose(
        power_result["energy_J"],
        contract_power["energy_J"],
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise ExecutionError(f"{run_id}: energy is not reproducible")
    if (
        validation.get("pi", {}).get("sha256") != pi_result["sha256"]
        or validation.get("powerz", {}).get("sha256") != power_result["sha256"]
        or validation.get("handshake_sha256") != sha256(handshake_file)
        or validation.get("cooldown_gate_sha256") != sha256(cooldown_file)
    ):
        raise ExecutionError(f"{run_id}: validation no longer matches raw evidence")
    return {
        "status": "VALID_COMPLETE",
        **expected,
        "evidence_directory": str(local_run.resolve()),
        "run_manifest_sha256": sha256(manifest_file),
        "pi_raw_sha256": pi_result["sha256"],
        "powerz_raw_sha256": power_result["sha256"],
        "handshake_sha256": sha256(handshake_file),
        "cooldown_gate_sha256": sha256(cooldown_file),
        "boot_id": identity["boot_id"],
        "energy_J": power_result["energy_J"],
    }


def validate_formal_registry(
    registry_path: Path,
    runs: list[dict[str, Any]],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    if not registry_path.is_file():
        raise ExecutionError("formal append-only registry is missing")
    planned = {str(run["run_id"]): run for run in runs}
    if len(planned) != len(runs):
        raise ExecutionError("formal plan contains duplicate run IDs")
    started = {run_id: 0 for run_id in planned}
    accepted = {run_id: 0 for run_id in planned}
    records = 0
    for line_number, line in enumerate(
        registry_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            raise ExecutionError(f"formal registry line {line_number} is blank")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ExecutionError(
                f"formal registry line {line_number} is malformed"
            ) from error
        if not isinstance(record, dict) or not isinstance(record.get("event"), str):
            raise ExecutionError(f"formal registry line {line_number} is invalid")
        records += 1
        run_id = record.get("run_id")
        if run_id is None:
            if record["event"] != "FORMAL_RUNTIME_PREFLIGHT_PASS":
                raise ExecutionError(
                    f"formal registry line {line_number} lacks run_id"
                )
            continue
        if run_id not in planned:
            raise ExecutionError(f"formal registry contains unplanned run_id {run_id}")
        expected_index = int(planned[run_id]["execution_index"])
        if int(record.get("execution_index", -1)) != expected_index:
            raise ExecutionError(
                f"formal registry execution index mismatch for {run_id}"
            )
        event = record["event"]
        if event == "EXECUTION_STARTED":
            started[run_id] += 1
            if started[run_id] > 1:
                raise ExecutionError(f"formal run started more than once: {run_id}")
            for field in ("policy", "condition", "seed"):
                if record.get(field) != planned[run_id][field]:
                    raise ExecutionError(
                        f"formal registry {field} mismatch for {run_id}"
                    )
        elif event in {"VALID_COMPLETE", "SKIPPED_REVALIDATED_VALID_COMPLETE"}:
            accepted[run_id] += 1
        elif event in {
            "EXECUTION_INVALID_TECHNICAL",
            "BLOCKED_EXISTING_ATTEMPT_NOT_VALID_COMPLETE",
        }:
            pass
        else:
            raise ExecutionError(f"formal registry has unknown event: {event}")
    missing_start = sorted(run_id for run_id, count in started.items() if count != 1)
    missing_acceptance = sorted(
        run_id for run_id, count in accepted.items() if count < 1
    )
    if require_complete and missing_start:
        raise ExecutionError(f"formal registry start count invalid: {missing_start}")
    if require_complete and missing_acceptance:
        raise ExecutionError(
            f"formal registry lacks complete-cell evidence: {missing_acceptance}"
        )
    return {
        "status": "PASS",
        "records": records,
        "planned_run_ids": len(planned),
        "execution_started_once": not missing_start,
        "no_unplanned_run_ids": True,
        "started_run_ids": sorted(
            run_id for run_id, count in started.items() if count == 1
        ),
    }


def build_formal_completeness(
    evidence_root: Path,
    plan: dict[str, Any],
    *,
    package_root: Path,
    manifest_path: Path,
    dependency: dict[str, Any],
    registry_path: Path,
) -> dict[str, Any]:
    runs = plan.get("runs", [])
    if len(runs) != 60:
        raise ExecutionError("formal completeness requires exactly 60 planned cells")
    registry_validation = validate_formal_registry(registry_path, runs)
    cells = []
    for run in runs:
        result = revalidate_complete_cell(
            evidence_root / str(run["run_id"]),
            run,
            package_root=package_root,
            manifest_path=manifest_path,
            dependency=dependency,
        )
        cells.append(
            {
                "execution_index": run["execution_index"],
                "run_id": result["run_id"],
                "policy": result["policy"],
                "condition": result["condition"],
                "seed": result["seed"],
                "status": result["status"],
                "run_manifest_sha256": result["run_manifest_sha256"],
                "pi_raw_sha256": result["pi_raw_sha256"],
                "powerz_raw_sha256": result["powerz_raw_sha256"],
                "handshake_sha256": result["handshake_sha256"],
                "cooldown_gate_sha256": result["cooldown_gate_sha256"],
                "boot_id": result["boot_id"],
            }
        )
    return {
        "status": "PI4B8_CORRECTED_V5_FORMAL_EVIDENCE_COMPLETE_PASS",
        "expected_cells": 60,
        "valid_complete_cells": 60,
        "accepted_windows": 36000,
        "package_manifest_sha256": sha256(manifest_path),
        "formal_gate_receipt_sha256": sha256(
            evidence_root / "FORMAL_GATE_RECEIPT.json"
        )
        if (evidence_root / "FORMAL_GATE_RECEIPT.json").is_file()
        else None,
        "append_only_registry": registry_path.name,
        "append_only_registry_sha256": sha256(registry_path),
        "append_only_registry_validation": registry_validation,
        "cells": cells,
    }


def validate_or_write_formal_completeness(
    evidence_root: Path,
    plan: dict[str, Any],
    *,
    package_root: Path,
    manifest_path: Path,
    dependency: dict[str, Any],
    registry_path: Path,
) -> Path:
    document = build_formal_completeness(
        evidence_root,
        plan,
        package_root=package_root,
        manifest_path=manifest_path,
        dependency=dependency,
        registry_path=registry_path,
    )
    path = evidence_root / FORMAL_COMPLETENESS
    if path.exists():
        if load_json(path) != document:
            raise ExecutionError("existing formal completeness artifact is stale")
    else:
        _write_json_new(path, document)
    return path


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
    meter: subprocess.Popen[str] | None = None
    worker: subprocess.Popen[str] | None = None
    meter_stdout_lines: list[str] = []
    worker_stdout_lines: list[str] = []
    meter_stderr = ""
    worker_stderr = ""
    handshake: dict[str, Any] = {"run_id": run_id, "status": "INVALID_PENDING"}
    try:
        cooldown = collect_idle_cooldown_gate(
            dependency, ssh_target=ssh_target, evidence_dir=local_run
        )
        meter_start = mac_monotonic()
        handshake["meter_capture_start_mac_monotonic"] = meter_start
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
            bufsize=1,
        )
        assert meter.stdout is not None and meter.stderr is not None
        meter_ready, meter_ready_time, _ = read_powerz_ready(
            meter.stdout,
            meter_path=meter_path,
            serial=str(dependency["powerz_device"]["serial"]),
            timeout_s=METER_READY_TIMEOUT_SECONDS,
            transcript=meter_stdout_lines,
        )
        handshake["powerz_ready"] = meter_ready
        handshake["powerz_ready_receive_mac_monotonic"] = meter_ready_time
        time.sleep(
            max(
                0.0,
                meter_ready_time + PRE_POST_ROLL_SECONDS - mac_monotonic(),
            )
        )
        request_time = mac_monotonic()
        handshake["start_request_mac_monotonic"] = request_time
        worker = subprocess.Popen(
            ["ssh", ssh_target, shlex.join(run["worker_argv"])],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert worker.stdout is not None and worker.stderr is not None
        start_ack, start_ack_time, _ = read_ack(
            worker.stdout,
            "RUN_START_ACK",
            run_id,
            timeout_s=START_ACK_TIMEOUT_SECONDS,
            transcript=worker_stdout_lines,
        )
        handshake["start_ack_receive_mac_monotonic"] = start_ack_time
        handshake["pi_start_ack"] = start_ack
        end_ack, end_ack_time, _ = read_ack(
            worker.stdout,
            "RUN_END_ACK",
            run_id,
            timeout_s=END_ACK_TIMEOUT_SECONDS,
            transcript=worker_stdout_lines,
        )
        handshake["end_ack_receive_mac_monotonic"] = end_ack_time
        handshake["pi_end_ack"] = end_ack
        remaining_stdout, worker_stderr = worker.communicate()
        worker_stdout_lines.append(remaining_stdout)
        worker_stdout_path.write_text(
            "".join(worker_stdout_lines), encoding="utf-8"
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

        time.sleep(
            max(0.0, end_ack_time + PRE_POST_ROLL_SECONDS - mac_monotonic())
        )
        try:
            meter_stdout, meter_stderr = meter.communicate(
                timeout=METER_CAPTURE_SECONDS + 15
            )
        except subprocess.TimeoutExpired as error:
            raise ExecutionError(f"{run_id}: POWER-Z capture did not terminate") from error
        meter_stdout_lines.append(meter_stdout)
        meter_end = mac_monotonic()
        meter_stdout_path.write_text("".join(meter_stdout_lines), encoding="utf-8")
        meter_stderr_path.write_text(meter_stderr, encoding="utf-8")
        if meter.returncode:
            raise ExecutionError(f"{run_id}: POWER-Z capture exited {meter.returncode}")
        if request_time - meter_ready_time < PRE_POST_ROLL_SECONDS:
            raise ExecutionError(f"{run_id}: POWER-Z READY pre-roll is below 5 seconds")
        if meter_end - end_ack_time < PRE_POST_ROLL_SECONDS:
            raise ExecutionError(f"{run_id}: POWER-Z post-roll is below 5 seconds")

        handshake = {
            "run_id": run_id,
            "policy": run["policy"],
            "condition": run["condition"],
            "seed": run["seed"],
            "meter_capture_start_mac_monotonic": meter_start,
            "powerz_ready_receive_mac_monotonic": meter_ready_time,
            "powerz_ready": meter_ready,
            "start_request_mac_monotonic": request_time,
            "start_ack_receive_mac_monotonic": start_ack_time,
            "end_ack_receive_mac_monotonic": end_ack_time,
            "meter_capture_end_mac_monotonic": meter_end,
            "pi_start_ack": start_ack,
            "pi_end_ack": end_ack,
            "cooldown_gate": COOLDOWN_EVIDENCE,
            "cooldown_gate_sha256": sha256(local_run / COOLDOWN_EVIDENCE),
            "device_identity": cooldown["device_identity"],
            "status": "PENDING_VALIDATION",
        }
        handshake_path.write_text(
            json.dumps(handshake, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        pi_result = validate_worker_output(
            pi_path,
            package_root=package_root,
            package_manifest_path=manifest_path,
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
        validate_actual_run_start_temperature(pi_result, cooldown, dependency)
        validate_powerz_ready_binding(
            handshake,
            run_id=run_id,
            serial=str(dependency["powerz_device"]["serial"]),
        )
        contract_power = validate_power_boundary(meter_path, handshake, run_id=run_id)
        power_result = validate_powerz_evidence(meter_path, handshake, run_id=run_id)
        if not math.isclose(
            power_result["energy_J"],
            contract_power["energy_J"],
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
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
            "cooldown_gate_sha256": sha256(local_run / COOLDOWN_EVIDENCE),
            "device_identity": cooldown["device_identity"],
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
            "cooldown_gate": COOLDOWN_EVIDENCE,
            "cooldown_gate_sha256": sha256(local_run / COOLDOWN_EVIDENCE),
            "device_identity": cooldown["device_identity"],
            "validity_status": "VALID_COMPLETE",
            "package_manifest_sha256": sha256(manifest_path),
            "artifacts": _artifact_inventory(
                local_run, exclude={"RUN_MANIFEST.json"}
            ),
        }
        _write_json_new(local_run / "RUN_MANIFEST.json", run_manifest)
        return {
            "run_id": run_id,
            "policy": run["policy"],
            "condition": run["condition"],
            "seed": run["seed"],
            "pi": pi_result,
            "powerz": power_result,
            "evidence_directory": str(local_run.resolve()),
            "run_manifest_sha256": sha256(local_run / "RUN_MANIFEST.json"),
        }
    except Exception as error:
        remaining_worker_stdout, remaining_worker_stderr = _stop_and_collect(worker)
        if remaining_worker_stdout:
            worker_stdout_lines.append(remaining_worker_stdout)
        if remaining_worker_stderr:
            worker_stderr += remaining_worker_stderr
        worker_stdout_path.write_text("".join(worker_stdout_lines), encoding="utf-8")
        worker_stderr_path.write_text(worker_stderr, encoding="utf-8")
        remaining_meter_stdout, remaining_meter_stderr = _stop_and_collect(meter)
        if remaining_meter_stdout:
            meter_stdout_lines.append(remaining_meter_stdout)
        if remaining_meter_stderr:
            meter_stderr += remaining_meter_stderr
        meter_stdout_path.write_text("".join(meter_stdout_lines), encoding="utf-8")
        meter_stderr_path.write_text(meter_stderr, encoding="utf-8")
        handshake.setdefault("meter_capture_end_mac_monotonic", mac_monotonic())
        handshake["status"] = "INVALID_TECHNICAL"
        handshake["reason"] = str(error)
        handshake_path.write_text(
            json.dumps(handshake, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            remote_recovery = recover_remote_partial(
                ssh_target=ssh_target,
                remote_output=str(run["remote_output"]),
                local_run=local_run,
            )
        except Exception as recovery_error:  # preserve the original failure
            remote_recovery = {
                "status": "RECOVERY_FAILED",
                "reason": str(recovery_error),
            }
            _write_json_new(
                local_run / "REMOTE_PARTIAL_RECOVERY.json", remote_recovery
            )
        invalid = {
            "status": "INVALID_TECHNICAL",
            "run_id": run_id,
            "reason": str(error),
            "remote_partial_recovery": remote_recovery,
        }
        validation_path = local_run / "RUN_VALIDATION.json"
        if not validation_path.exists():
            _write_json_new(validation_path, invalid)
        run_manifest_path = local_run / "RUN_MANIFEST.json"
        if not run_manifest_path.exists():
            invalid_manifest = {
                "run_id": run_id,
                "policy": run["policy"],
                "condition": run["condition"],
                "seed": run["seed"],
                "validity_status": "INVALID_TECHNICAL",
                "reason": str(error),
                "package_manifest_sha256": sha256(manifest_path),
                "artifacts": _artifact_inventory(
                    local_run, exclude={"RUN_MANIFEST.json"}
                ),
            }
            _write_json_new(run_manifest_path, invalid_manifest)
        raise
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            worker.wait()
        if meter is not None and meter.poll() is None:
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
    powerz_validator_sha256: str,
) -> tuple[Path, Path]:
    validate_commissioning_identity(
        str(result.get("policy", "")),
        str(result.get("condition", "")),
        int(result.get("seed", -1)),
    )
    power = result["powerz"]
    pi = result["pi"]
    evidence_directory = Path(result["evidence_directory"]).resolve()
    cooldown_path = evidence_directory / COOLDOWN_EVIDENCE
    cooldown = load_json(cooldown_path)
    preflight_pi = runtime_preflight.get("pi_worker", {})
    expected_identity = dict(preflight_pi.get("device_identity", {}))
    expected_identity["boot_id"] = preflight_pi.get("boot_id")
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
        "cooldown_gate": cooldown.get("status") == "PASS",
        "device_identity_bound": cooldown.get("device_identity")
        == expected_identity,
        "actual_start_temperature_gate": pi[
            "first_window_temperature_start_C"
        ]
        <= cooldown["idle_median_temperature_c"]
        + cooldown["configuration"]["maximum_start_above_idle_median_c"],
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
        "evidence_directory": str(evidence_directory),
        "pi_raw_path": str(evidence_directory / "pi_windows.csv"),
        "powerz_raw_path": str(evidence_directory / "POWERZ_raw.csv"),
        "handshake_path": str(evidence_directory / "HANDSHAKE.json"),
        "cooldown_gate_path": str(cooldown_path),
        "run_manifest_path": str(evidence_directory / "RUN_MANIFEST.json"),
        "handshake_sha256": sha256(evidence_directory / "HANDSHAKE.json"),
        "cooldown_gate_sha256": sha256(cooldown_path),
        "package_manifest_sha256": package_manifest_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "worker_sha256": worker_sha256,
        "orchestrator_sha256": orchestrator_sha256,
        "runtime_preflight_sha256": sha256(authorization_dir / RUNTIME_PREFLIGHT),
        "runtime_dependencies_sha256": runtime_preflight[
            "dependency_manifest_sha256"
        ],
        "powerz_validator_sha256": powerz_validator_sha256,
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


def write_final_execution_manifest(
    package_root: Path,
    package_manifest: Path,
    authorization_dir: Path,
    package: dict[str, Any],
) -> Path:
    """Freeze the deterministic final authorization inputs after commissioning."""
    validation_path = authorization_dir / COMMISSIONING_VALIDATION
    decision_path = authorization_dir / COMMISSIONING_DECISION
    preflight_path = authorization_dir / RUNTIME_PREFLIGHT
    plan_path = authorization_dir / FORMAL_PLAN
    validation = load_json(validation_path)
    validate_frozen_plan(package_root, package_manifest, plan_path)
    if validation.get("status") != "PASS":
        raise ExecutionError("cannot freeze final manifest from failed commissioning")
    artifacts = _artifact_index(package)
    _, _, dependency_sha256 = runtime_dependency_contract(package_root, package)
    commissioning_binding = validate_commissioning_package_binding(package)
    final = {
        "status": "PI4B8_CORRECTED_V5_FINAL_EXECUTION_MANIFEST",
        "execution_authorized": True,
        "official_test_access": "forbidden",
        "commissioning_run_id": validation["run_id"],
        "commissioning_validation_sha256": sha256(validation_path),
        "commissioning_decision_sha256": sha256(decision_path),
        "formal_plan_sha256": sha256(plan_path),
        "orchestrator_sha256": sha256(Path(__file__)),
        "runtime_package_manifest_sha256": sha256(package_manifest),
        "source_manifest_sha256": package["source_manifest_sha256"],
        "worker_sha256": artifacts["worker"]["sha256"],
        "powerz_capture_sha256": artifacts["powerz:capture"]["sha256"],
        "powerz_validator_sha256": artifacts["powerz:validator"]["sha256"],
        "powerz_specification_sha256": artifacts["powerz:specification"]["sha256"],
        "runtime_preflight_sha256": sha256(preflight_path),
        "runtime_dependencies_sha256": dependency_sha256,
        "commissioning_batch_indices_sha256": commissioning_binding[
            "batch_indices_sha256"
        ],
        "commissioning_pi_raw_sha256": validation["pi_raw_sha256"],
        "commissioning_powerz_raw_sha256": validation["powerz_raw_sha256"],
        "commissioning_handshake_sha256": validation["handshake_sha256"],
        "commissioning_cooldown_gate_sha256": validation[
            "cooldown_gate_sha256"
        ],
        "commissioning_run_manifest_sha256": validation["run_manifest_sha256"],
    }
    path = authorization_dir / FINAL_MANIFEST
    _write_json_new(path, final)
    return path


def commissioning_run(args: argparse.Namespace) -> None:
    validate_commissioning_identity(args.policy, args.condition, args.seed)
    package = validate_materialized_package(args.package_root, args.package_manifest)
    validate_frozen_plan(
        args.package_root,
        args.package_manifest,
        args.authorization_dir / FORMAL_PLAN,
    )
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
        powerz_validator_sha256=artifacts["powerz:validator"]["sha256"],
    )
    final_path = write_final_execution_manifest(
        args.package_root,
        args.package_manifest,
        args.authorization_dir,
        package,
    )
    gate = validate_formal_authorization(
        args.package_root,
        args.package_manifest,
        args.authorization_dir,
    )
    print("PI4B8_CORRECTED_V5_COMMISSIONING_PASS")
    print(f"validation={validation_path} decision={decision_path}")
    print(gate["status"])
    print(f"final_manifest={final_path} sha256={sha256(final_path)}")


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
    _, dependency, _ = runtime_dependency_contract(args.package_root, package)
    registry_path = args.evidence_root / FORMAL_REGISTRY
    completeness_path = args.evidence_root / FORMAL_COMPLETENESS
    if completeness_path.is_file():
        validate_or_write_formal_completeness(
            args.evidence_root,
            receipt["plan"],
            package_root=args.package_root,
            manifest_path=args.package_manifest,
            dependency=dependency,
            registry_path=registry_path,
        )
        print("PI4B8_CORRECTED_V5_FORMAL_EVIDENCE_COMPLETE_PASS")
        return
    live_preflight = validate_runtime_preflight(
        args.package_root,
        package,
        remote_package_root=args.remote_package_root,
        ssh_target=args.ssh_target,
        mac_python=args.mac_python,
    )
    live_preflight_path = args.evidence_root / "FORMAL_RUNTIME_PREFLIGHT.json"
    if live_preflight_path.exists():
        index = 1
        while True:
            candidate = args.evidence_root / (
                f"FORMAL_RUNTIME_PREFLIGHT_RESUME_{index:04d}.json"
            )
            if not candidate.exists():
                live_preflight_path = candidate
                break
            index += 1
    _write_json_new(live_preflight_path, live_preflight)
    if not receipt_path.exists():
        _write_json_new(
            receipt_path,
            {
                **{key: value for key, value in receipt.items() if key != "plan"},
                "formal_runtime_preflight": live_preflight_path.name,
                "formal_runtime_preflight_sha256": sha256(live_preflight_path),
            },
        )
    else:
        frozen_receipt = load_json(receipt_path)
        for field in (
            "status",
            "final_manifest_sha256",
            "formal_plan_sha256",
            "runs",
        ):
            if frozen_receipt.get(field) != receipt.get(field):
                raise ExecutionError(f"formal gate receipt mismatch: {field}")
    _append_registry(
        registry_path,
        {
            "event": "FORMAL_RUNTIME_PREFLIGHT_PASS",
            "preflight": live_preflight_path.name,
            "preflight_sha256": sha256(live_preflight_path),
            "boot_id": live_preflight["pi_worker"]["boot_id"],
        },
    )
    registry_state = validate_formal_registry(
        registry_path, receipt["plan"]["runs"], require_complete=False
    )
    started_run_ids = set(registry_state["started_run_ids"])
    for index, run in enumerate(receipt["plan"]["runs"], 1):
        local_run = args.evidence_root / str(run["run_id"])
        if local_run.exists():
            if run["run_id"] not in started_run_ids:
                _append_registry(
                    registry_path,
                    {
                        "event": "BLOCKED_EXISTING_ATTEMPT_NOT_VALID_COMPLETE",
                        "execution_index": index,
                        "run_id": run["run_id"],
                        "reason": "evidence directory has no registered execution start",
                    },
                )
                raise ExecutionError(
                    f"{run['run_id']}: evidence directory has no registered execution start"
                )
            try:
                revalidated = revalidate_complete_cell(
                    local_run,
                    run,
                    package_root=args.package_root,
                    manifest_path=args.package_manifest,
                    dependency=dependency,
                )
            except Exception as error:
                _append_registry(
                    registry_path,
                    {
                        "event": "BLOCKED_EXISTING_ATTEMPT_NOT_VALID_COMPLETE",
                        "execution_index": index,
                        "run_id": run["run_id"],
                        "reason": str(error),
                    },
                )
                raise ExecutionError(
                    f"{run['run_id']}: existing attempt is not a revalidated "
                    "VALID_COMPLETE cell; replacement authorization is required"
                ) from error
            _append_registry(
                registry_path,
                {
                    "event": "SKIPPED_REVALIDATED_VALID_COMPLETE",
                    "execution_index": index,
                    "run_id": run["run_id"],
                    "run_manifest_sha256": revalidated["run_manifest_sha256"],
                },
            )
            print(f"{index}/60 REVALIDATED_VALID_COMPLETE", flush=True)
            continue
        if run["run_id"] in started_run_ids:
            _append_registry(
                registry_path,
                {
                    "event": "BLOCKED_EXISTING_ATTEMPT_NOT_VALID_COMPLETE",
                    "execution_index": index,
                    "run_id": run["run_id"],
                    "reason": "registered attempt evidence directory is missing",
                },
            )
            raise ExecutionError(
                f"{run['run_id']}: registered attempt evidence directory is missing"
            )
        _append_registry(
            registry_path,
            {
                "event": "EXECUTION_STARTED",
                "execution_index": index,
                "run_id": run["run_id"],
                "policy": run["policy"],
                "condition": run["condition"],
                "seed": run["seed"],
            },
        )
        started_run_ids.add(run["run_id"])
        try:
            execute_cell(
                args.package_root,
                args.package_manifest,
                run,
                evidence_root=args.evidence_root,
                ssh_target=args.ssh_target,
                mac_python=args.mac_python,
            )
            revalidated = revalidate_complete_cell(
                local_run,
                run,
                package_root=args.package_root,
                manifest_path=args.package_manifest,
                dependency=dependency,
            )
        except Exception as error:
            _append_registry(
                registry_path,
                {
                    "event": "EXECUTION_INVALID_TECHNICAL",
                    "execution_index": index,
                    "run_id": run["run_id"],
                    "reason": str(error),
                },
            )
            raise
        _append_registry(
            registry_path,
            {
                "event": "VALID_COMPLETE",
                "execution_index": index,
                "run_id": run["run_id"],
                "run_manifest_sha256": revalidated["run_manifest_sha256"],
            },
        )
        print(f"{index}/60 VALID_COMPLETE", flush=True)
    final_path = validate_or_write_formal_completeness(
        args.evidence_root,
        receipt["plan"],
        package_root=args.package_root,
        manifest_path=args.package_manifest,
        dependency=dependency,
        registry_path=registry_path,
    )
    print("PI4B8_CORRECTED_V5_FORMAL_EVIDENCE_COMPLETE_PASS")
    print(f"validation={final_path} sha256={sha256(final_path)}")


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
