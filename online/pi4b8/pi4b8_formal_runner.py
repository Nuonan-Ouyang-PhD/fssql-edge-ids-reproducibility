#!/usr/bin/env python3
"""Fail-closed Pi4B8 lineage and formal-authorization gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


class GateError(RuntimeError):
    """A frozen formal dependency is absent, inconsistent, or unverified."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise GateError(f"missing required artifact: {path}")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"invalid JSON artifact: {path}: {exc}") from exc


def require_hash(path: Path, expected: str, role: str) -> None:
    if not path.is_file():
        raise GateError(f"missing {role}: {path}")
    observed = sha256(path)
    if observed != expected:
        raise GateError(
            f"{role} hash mismatch: expected {expected}, observed {observed}: {path}"
        )


def validate_background_load_freeze(config: dict) -> None:
    """Reject labelled B1/B2 conditions without a frozen executable load spec."""
    requested = {
        item["background"]
        for item in config.get("workload", {}).get("background_robustness", [])
        if item.get("background") in {"B1", "B2"}
    }
    if not requested:
        return

    raise GateError(
        "B1/B2 formal execution blocked: the frozen Pi4B8 config labels "
        f"{','.join(sorted(requested))} but does not freeze an executable load "
        "command, CPU pin, duty/period, verification command, and cleanup command"
    )


def validate_lineage(root: Path, config_path: Path) -> dict:
    config = load_json(config_path)
    if config.get("status") != "FROZEN_BEFORE_PI4B8_ONLINE_COMPARISON":
        raise GateError("Pi4B8 experiment config is not frozen")
    if config.get("official_test") != "sealed":
        raise GateError("historical config does not record sealed-at-execution test state")

    guard = config.get("guard", {})
    if (
        guard.get("latency_cap_ms") != 40.0
        or guard.get("thermal_cap_C") != 82.0
        or guard.get("decision_window_ms") != 100.0
    ):
        raise GateError("guard mismatch")
    for key in ("latency_predictor", "thermal_predictor", "margins"):
        expected = guard.get("hashes", {}).get(f"{key}_sha256")
        if not expected:
            raise GateError(f"missing frozen guard hash: {key}")
        require_hash(root / guard[key], expected, f"guard {key}")

    package = root / ".formal_inputs/pi4b8_scheduler_train_v1"
    manifest = load_json(package / "PACKAGE_MANIFEST.json")
    if manifest.get("official_test") != "sealed_not_read":
        raise GateError("historical input package is not recorded as development-only data")
    if manifest.get("windows_per_run") != 600:
        raise GateError("formal input package must contain 600 windows per run")
    for name, record in manifest.get("files", {}).items():
        require_hash(package / name, record["sha256"], f"formal input {name}")

    q_manifest_path = root / "q_training/pi4b8_q_v1/Q_TABLE_MANIFEST.json"
    q_manifest = load_json(q_manifest_path)
    if q_manifest.get("seeds") != [5101, 5102, 5103, 5104, 5105]:
        raise GateError("Q seed mapping mismatch")
    for name, expected in q_manifest.get("artifact_sha256", {}).items():
        require_hash(q_manifest_path.parent / name, expected, f"Q table {name}")

    for path in package.glob("*_features.npy"):
        values = np.load(path, allow_pickle=False)
        if values.shape[0] != 600 or values.shape[1] not in (191, 8):
            raise GateError(f"bad runtime input shape: {path.name}: {values.shape}")
    for path in package.glob("*_raw_features.csv"):
        header = path.read_text().splitlines()[0].split(",")
        forbidden = {"label", "attack_cat", "source_id", "source_row_index"}
        if forbidden.intersection(header):
            raise GateError(f"label-bearing runtime input: {path.name}")

    power_protocol = root / "protocols/POWER_Z_HID_ACQUISITION_V1.md"
    if not power_protocol.is_file():
        raise GateError(f"missing POWER-Z protocol: {power_protocol}")
    match = re.search(r"SHA-256:\s*`([0-9a-f]{64})`", power_protocol.read_text())
    if not match:
        raise GateError("POWER-Z protocol does not contain a capture-script SHA-256")
    require_hash(
        root / "online/pi4b8/powerz_hid_capture.py",
        match.group(1),
        "POWER-Z capture implementation",
    )

    return config


def validate_formal_authorization(root: Path) -> dict:
    protocols = root / "protocols"
    validation_path = protocols / "PI4B8_COMMISSIONING_VALIDATION.json"
    decision_path = protocols / "PI4B8_COMMISSIONING_DECISION.md"
    final_path = protocols / "PI4B8_FINAL_EXECUTION_MANIFEST.json"

    validation = load_json(validation_path)
    if validation.get("status") != "PASS":
        raise GateError("commissioning validation status is not PASS")
    if validation.get("marker") != "COMMISSIONING_ONLY_EXCLUDED_FROM_FORMAL_STATISTICS":
        raise GateError("commissioning exclusion marker mismatch")
    checks = validation.get("checks", {})
    required_true = (
        "windows_600",
        "timing_fields",
        "tp_fp_fn_tn_recomputable",
        "meter_monotonic",
        "ordering",
        "energy_reproducible",
        "run_id_binding",
        "label_leakage",
        "throttle_zero",
    )
    failed = [name for name in required_true if checks.get(name) is not True]
    if failed:
        raise GateError(f"commissioning checks failed or missing: {', '.join(failed)}")
    if checks.get("pre_roll_s", 0) < 5 or checks.get("post_roll_s", 0) < 5:
        raise GateError("commissioning pre/post roll is below 5 seconds")

    if not decision_path.is_file():
        raise GateError(f"missing commissioning decision: {decision_path}")
    decision = decision_path.read_text()
    if "COMMISSIONING PASS" not in decision:
        raise GateError("commissioning decision does not record PASS")
    if validation.get("run_id") not in decision:
        raise GateError("commissioning decision run_id mismatch")

    final = load_json(final_path)
    if final.get("status") != "PI4B8_FINAL_EXECUTION_MANIFEST":
        raise GateError("final execution manifest status mismatch")
    if final.get("commissioning_run_id") != validation.get("run_id"):
        raise GateError("final manifest commissioning run_id mismatch")
    require_hash(
        validation_path,
        final.get("commissioning_validation_sha256", ""),
        "commissioning validation",
    )
    require_hash(
        decision_path,
        final.get("commissioning_decision_sha256", ""),
        "commissioning decision",
    )
    require_hash(
        root / "online/pi4b8/pi4b8_commissioning_orchestrator.py",
        final.get("orchestrator_sha256", ""),
        "commissioning orchestrator",
    )
    require_hash(
        protocols / "PI4B8_FORMAL_RUNTIME_PACKAGE_MANIFEST_V4.json",
        final.get("runtime_package_manifest_sha256", ""),
        "formal runtime package manifest",
    )
    return final


def run_gate(root: Path, config_path: Path, formal: bool) -> dict:
    config = validate_lineage(root, config_path)
    if formal:
        validate_background_load_freeze(config)
        validate_formal_authorization(root)
        return {"status": "PI4B8_FINAL_FORMAL_GATE_PASS"}
    return {
        "status": "PI4B8_FORMAL_RUNNER_LINEAGE_GATE_PASS",
        "formal_execution": "not_authorized_by_this_invocation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/pi4b8_online_experiment_v1.json",
    )
    parser.add_argument("--commissioning", action="store_true")
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    try:
        result = run_gate(ROOT, args.config, args.formal)
    except GateError as exc:
        raise SystemExit(f"Pi4B8 gate failed: {exc}") from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
