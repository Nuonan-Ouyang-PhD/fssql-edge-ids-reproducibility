#!/usr/bin/env python3
"""Build the frozen matched-seed Stress Extension V3 execution matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    config = json.loads(args.config.read_text())
    if config["status"] != "PREREGISTERED_BEFORE_STRESS_EXTENSION_V3_POLICY_EXECUTION":
        raise RuntimeError("Stress Extension V3 protocol is not preregistered")

    fields = [
        "global_order", "seed_block", "within_seed_order", "seed", "regime",
        "policy", "training_seed", "planned_run_id", "windows",
        "decision_window_ms", "n_records", "background", "attempt", "status",
    ]
    workload = config["workloads"][config["regimes"][0]]
    training_map = config["q_learning"]["evaluation_seed_to_training_seed"]
    global_order = 0
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for seed_block, seed in enumerate(config["evaluation_seeds"]):
            rng = np.random.default_rng(config["execution_order"]["base_seed"] + seed)
            for within_seed_order, policy_index in enumerate(rng.permutation(len(config["policies"]))):
                policy = config["policies"][int(policy_index)]
                global_order += 1
                writer.writerow({
                    "global_order": global_order,
                    "seed_block": seed_block,
                    "within_seed_order": within_seed_order,
                    "seed": seed,
                    "regime": config["regimes"][0],
                    "policy": policy,
                    "training_seed": training_map[str(seed)] if policy in ("Unshielded-Q", "FSSQL-R") else "",
                    "planned_run_id": f"pi3bplus_stress_v3_{slug(policy)}_s{seed}_a1",
                    "windows": config["windows_per_run"],
                    "decision_window_ms": config["decision_window_ms"],
                    "n_records": workload["n_records"],
                    "background": workload["background"],
                    "attempt": 1,
                    "status": "PLANNED",
                })

    expected = config["execution_order"].get("expected_matrix_sha256")
    if expected and sha256(args.output) != expected:
        raise RuntimeError("Stress Extension V3 matrix hash mismatch")
    print(f"wrote {global_order} preregistered runs; sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
