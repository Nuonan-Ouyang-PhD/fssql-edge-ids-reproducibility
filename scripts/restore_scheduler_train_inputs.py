#!/usr/bin/env python3
"""Restore the frozen scheduler-development inputs from UNSW training only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RISK_COLUMNS = [
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "ct_state_ttl",
    "ct_dst_src_ltm",
    "is_sm_ips_ports",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"{role} hash mismatch: expected {expected}, observed {observed}"
        )


def restore(source: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite restored inputs: {output}")

    dataset_manifest = json.loads((ROOT / "data/DATASET_SHA256.json").read_text())
    expected_source = dataset_manifest["files"]["UNSW_NB15_training-set.csv"]
    require_hash(source, expected_source["sha256"], "official training CSV")

    lineage_path = (
        ROOT / "device_calibration/input_lineage/scheduler_train_6000_label_lineage.csv"
    )
    restoration_manifest = json.loads(
        (
            ROOT
            / "device_calibration/input_lineage/"
            "SCHEDULER_TRAIN_6000_LABEL_RESTORATION_SHA256.json"
        ).read_text()
    )
    expected_lineage = restoration_manifest["files"][
        "device_calibration/inputs/scheduler_train_6000_label_lineage.csv"
    ]["sha256"]
    require_hash(lineage_path, expected_lineage, "scheduler lineage")

    frame = pd.read_csv(source)
    lineage = pd.read_csv(lineage_path)
    if len(lineage) != 6000 or lineage["runtime_sequence"].tolist() != list(range(6000)):
        raise RuntimeError("scheduler lineage is not the ordered 6000-row trace")
    source_indices = lineage["source_row_index"].to_numpy(dtype=np.int64)
    if len(np.unique(source_indices)) != 6000 or np.any(source_indices < 0) or np.any(
        source_indices >= len(frame)
    ):
        raise RuntimeError("scheduler lineage is not a one-to-one training-row mapping")

    selected = frame.iloc[source_indices].reset_index(drop=True)
    if not np.array_equal(selected["id"].to_numpy(), lineage["source_id"].to_numpy()):
        raise RuntimeError("source ID lineage mismatch")
    if not np.array_equal(selected["label"].to_numpy(), lineage["label"].to_numpy()):
        raise RuntimeError("label lineage mismatch")
    if selected["attack_cat"].fillna("").astype(str).tolist() != lineage[
        "attack_cat"
    ].fillna("").astype(str).tolist():
        raise RuntimeError("attack-category lineage mismatch")

    split = pd.read_csv(ROOT / "data/SPLIT_MANIFEST.csv").iloc[source_indices]
    if not split["split"].eq("scheduler_train").all() or split[
        "feature_group_hex"
    ].astype(str).tolist() != lineage["feature_group_hex"].astype(str).tolist():
        raise RuntimeError("rows do not belong to the frozen scheduler_train split")

    raw = selected.drop(columns=["id", "attack_cat", "label"])
    from runtime.preprocessor_numpy import PortablePreprocessor

    preprocessor = PortablePreprocessor(ROOT / "preprocessing/preprocessor_portable.npz")
    features = preprocessor.transform(raw.to_dict(orient="records"))
    risk_features = selected[RISK_COLUMNS].to_numpy(dtype=np.float64)
    labels = selected["label"].to_numpy(dtype=np.int8)
    if not np.isin(labels, [0, 1]).all():
        raise RuntimeError("restored scoring labels are not binary")

    output.mkdir(parents=True)
    feature_path = output / "scheduler_train_6000_features.npy"
    risk_path = output / "scheduler_train_6000_risk_features.npy"
    raw_path = output / "scheduler_train_6000_raw_features.csv"
    label_path = output / "scheduler_train_6000_labels.npy"
    index_path = output / "SOURCE_INDEX.csv"
    np.save(feature_path, features, allow_pickle=False)
    np.save(risk_path, risk_features, allow_pickle=False)
    np.save(label_path, labels, allow_pickle=False)
    raw.to_csv(raw_path, index=False)
    with index_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sequence", "source_row_index", "feature_group_hex"])
        for sequence, row in lineage.iterrows():
            writer.writerow(
                [sequence, int(row["source_row_index"]), row["feature_group_hex"]]
            )

    old_input_manifest = json.loads(
        (ROOT / "device_calibration/input_lineage/MANIFEST.json").read_text()
    )
    for path in (feature_path, risk_path, raw_path, index_path):
        expected = old_input_manifest["files"][path.name]
        if path.stat().st_size != int(expected["size_bytes"]):
            raise RuntimeError(f"restored byte-size mismatch: {path.name}")
        require_hash(path, expected["sha256"], f"restored {path.name}")

    files = [feature_path, risk_path, raw_path, label_path, index_path]
    manifest = {
        "status": "SCHEDULER_TRAIN_6000_RESTORATION_PASS",
        "source": "official UNSW-NB15 training CSV only",
        "official_test_access": "not used",
        "rows": 6000,
        "one_to_one": True,
        "runtime_order_preserved": True,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        },
    }
    manifest_path = output / "RESTORATION_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/raw/UNSW_NB15_training-set.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "device_calibration/inputs"
    )
    args = parser.parse_args()
    print(json.dumps(restore(args.source.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
