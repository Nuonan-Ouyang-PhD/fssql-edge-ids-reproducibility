#!/usr/bin/env python3
"""Open the sealed official test once and build matched feature-only traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.preprocessor_numpy import PortablePreprocessor  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Git must be clean before official-test opening")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.source_commit:
        raise RuntimeError("source commit does not equal clean HEAD")

    policy = json.loads(args.policy_config.read_text())
    if policy["official_test"]["status_at_preregistration"] != "sealed":
        raise RuntimeError("unexpected preregistration test status")
    q_manifest = ROOT / "q_training/Q_TABLE_MANIFEST.json"
    q = json.loads(q_manifest.read_text())
    if q["status"] != "Q_TRAINING_AND_STABILITY_DIAGNOSTIC_PASS":
        raise RuntimeError("Q tables are not frozen")
    test_path = ROOT / "data/raw/UNSW_NB15_testing-set.csv"
    dataset = json.loads((ROOT / "data/DATASET_SHA256.json").read_text())
    expected_test_hash = dataset["files"][test_path.name]["sha256"]
    if sha256(test_path) != expected_test_hash:
        raise RuntimeError("official-test source hash mismatch")
    parity_path = ROOT / "preprocessing/PREPROCESSOR_PORTABLE_PARITY.json"
    parity = json.loads(parity_path.read_text())
    portable_path = ROOT / parity["artifact"]
    if parity["status"] != "PASS" or sha256(portable_path) != parity["artifact_sha256"]:
        raise RuntimeError("portable preprocessor parity artifact mismatch")

    with test_path.open(newline="", encoding="utf-8-sig") as stream:
        test_rows = list(csv.DictReader(stream))
    preprocessor = PortablePreprocessor(portable_path)
    risk_names = json.loads((ROOT / "models/RISK_PROXY_SPEC.json").read_text())["features"]
    forbidden = set(policy["official_test"]["scheduler_forbidden_fields"])
    feature_names = preprocessor.numeric_columns + preprocessor.categorical_columns
    seeds = policy["evaluation_seeds"]
    windows = policy["windows_per_run"]
    selected_raw = []
    risk_features = []
    labels = []
    indices = []
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(len(test_rows))[:windows]
        for window_id, source_index in enumerate(order):
            source = test_rows[int(source_index)]
            if forbidden.intersection(feature_names):
                raise RuntimeError("forbidden scheduler field in feature schema")
            selected_raw.append({name: source[name] for name in feature_names})
            risk_features.append([float(source[name]) for name in risk_names])
            labels.append(int(source["label"]))
            indices.append({
                "array_index": len(indices),
                "seed": seed,
                "window_id": window_id,
                "source_row_index": int(source_index),
                "source_id": source["id"],
            })
    features = preprocessor.transform(selected_raw)

    args.output.mkdir(parents=True)
    raw_path = args.output / "trace_raw_features.csv"
    with raw_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=feature_names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected_raw)
    index_path = args.output / "trace_index.csv"
    with index_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(indices[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(indices)
    feature_path = args.output / "trace_features.npy"
    risk_path = args.output / "trace_risk_features.npy"
    label_path = args.output / "trace_labels.npy"
    np.save(feature_path, features, allow_pickle=False)
    np.save(risk_path, np.asarray(risk_features, dtype=np.float64), allow_pickle=False)
    np.save(label_path, np.asarray(labels, dtype=np.int8), allow_pickle=False)
    files = [raw_path, index_path, feature_path, risk_path, label_path]
    manifest = {
        "status": "OFFICIAL_TEST_OPENED_AND_MATCHED_TRACES_FROZEN",
        "opened_utc": utc_now(),
        "opening_source_commit": head,
        "policy_config_sha256": sha256(args.policy_config),
        "q_manifest_sha256": sha256(q_manifest),
        "official_test_source": str(test_path.relative_to(ROOT)),
        "official_test_source_sha256": expected_test_hash,
        "official_test_rows": len(test_rows),
        "seeds": seeds,
        "windows_per_seed": windows,
        "trace_rows": len(indices),
        "scheduler_forbidden_fields": sorted(forbidden),
        "scheduler_feature_fields": feature_names,
        "label_access_boundary": "Labels are stored in a separate array and are not passed to policy selection; used after selected-action inference for metrics only.",
        "artifact_sha256": {path.name: sha256(path) for path in files},
    }
    (args.output / "OFFICIAL_TEST_INPUT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
