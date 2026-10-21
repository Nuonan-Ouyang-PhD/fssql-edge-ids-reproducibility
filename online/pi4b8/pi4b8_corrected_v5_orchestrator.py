#!/usr/bin/env python3
"""Generate the hash-bound corrected V5 run plan without executing hardware.

The historical V4 launcher/orchestrator remains untouched.  This file closes
the worker/package CLI mismatch, but intentionally has no execution switch:
commissioning and final authorization artifacts must be frozen first.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

import numpy as np

from online.pi4b8.corrected_v5_contract import (
    CONDITIONS,
    PACKAGE_MANIFEST_NAME,
    POLICIES,
    SEEDS,
    ContractError,
    build_worker_argv,
    find_cell,
    sha256,
    validate_materialized_package,
)


def build_order() -> list[tuple[int, str, str]]:
    order: list[tuple[int, str, str]] = []
    for seed in SEEDS:
        rng = np.random.default_rng(20260910 + seed)
        block = [(policy, condition) for condition in CONDITIONS for policy in POLICIES]
        rng.shuffle(block)
        order.extend((seed, policy, condition) for policy, condition in block)
    return order


def build_plan(
    package_root: Path,
    manifest_path: Path,
    *,
    remote_package_root: Path,
    remote_output_root: Path,
) -> dict[str, Any]:
    remote_package_root = Path(remote_package_root)
    remote_output_root = Path(remote_output_root)
    if not remote_package_root.is_absolute() or not remote_output_root.is_absolute():
        raise ContractError("remote package/output roots must be absolute")
    if "v4" in remote_package_root.name.lower():
        raise ContractError("corrected orchestrator must not target historical V4")
    if remote_package_root == remote_output_root or remote_package_root in remote_output_root.parents:
        raise ContractError("remote output must be outside the immutable package")

    manifest = validate_materialized_package(package_root, manifest_path)
    artifacts = {entry["role"]: entry for entry in manifest["artifacts"]}
    runs = []
    for index, (seed, policy, condition) in enumerate(build_order(), 1):
        safe_policy = policy.lower().replace("-", "_")
        safe_condition = condition.replace("/", "_")
        run_id = f"pi4b8_corrected_v5_{index:02d}_{safe_policy}_{safe_condition}_s{seed}"
        output = remote_output_root / run_id
        argv = build_worker_argv(
            package_root,
            manifest_path,
            run_id=run_id,
            policy=policy,
            condition=condition,
            seed=seed,
            output=output,
            execution_root=remote_package_root,
        )
        cell = find_cell(manifest, condition, seed)
        q_role = f"q_table:{seed}" if policy == "FSSQL-R" else None
        runs.append(
            {
                "execution_index": index,
                "run_id": run_id,
                "policy": policy,
                "condition": condition,
                "seed": seed,
                "windows": 600,
                "records_per_window": 16,
                "runtime_roles": cell["runtime"],
                "scoring_roles": cell["scoring"],
                "q_table_role": q_role,
                "q_table_sha256": "" if q_role is None else artifacts[q_role]["sha256"],
                "remote_output": str(output),
                "worker_argv": argv,
                "worker_shell_command": shlex.join(argv),
            }
        )
    return {
        "status": "CORRECTED_V5_PLAN_ONLY_NOT_EXECUTION_AUTHORIZATION",
        "package_manifest": PACKAGE_MANIFEST_NAME,
        "package_manifest_sha256": sha256(manifest_path),
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "worker_sha256": artifacts["worker"]["sha256"],
        "remote_package_root": str(remote_package_root),
        "remote_output_root": str(remote_output_root),
        "official_test_access": "forbidden",
        "scoring_label_access": "post_action_only",
        "powerz_primary_boundary": (
            "start_ack_receive_mac_monotonic_to_end_ack_receive_mac_monotonic"
        ),
        "powerz_boundary_includes_post_decision_scoring_and_evidence_write": True,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--remote-package-root", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        plan = build_plan(
            args.package_root,
            args.package_manifest,
            remote_package_root=args.remote_package_root,
            remote_output_root=args.remote_output_root,
        )
    except ContractError as error:
        raise SystemExit(f"PI4B8_CORRECTED_V5_PLAN_BLOCKED: {error}") from error
    if args.plan_output.exists():
        raise SystemExit(f"refusing to overwrite plan: {args.plan_output}")
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.plan_output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print("PI4B8_CORRECTED_V5_PLAN_GATE_PASS_NOT_EXECUTION_AUTHORIZATION")
    print(f"runs={len(plan['runs'])} plan_sha256={sha256(args.plan_output)}")


if __name__ == "__main__":
    main()
