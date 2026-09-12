#!/usr/bin/env python3
"""Create a seeded scheduler_train-only input bundle for device calibration."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
SPLITS = ROOT / "data" / "SPLIT_MANIFEST.csv"
PREPROCESSOR = ROOT / "preprocessing" / "preprocessor.joblib"
OUTPUT = ROOT / "device_calibration" / "inputs"
SEED = 20260909
ROWS = 6000
RISK_FEATURES = [
    "rate", "sttl", "dttl", "sload", "dload", "ct_state_ttl",
    "ct_dst_src_ltm", "is_sm_ips_ports",
]


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


if OUTPUT.exists():
    raise FileExistsError(f"refusing to overwrite calibration input bundle: {OUTPUT}")
OUTPUT.mkdir(parents=True)
frame = pd.read_csv(SOURCE)
splits = pd.read_csv(SPLITS)
available = np.flatnonzero(splits["split"].eq("scheduler_train").to_numpy())
rng = np.random.default_rng(SEED)
selected = rng.choice(available, size=ROWS, replace=False)
preprocessor = joblib.load(PREPROCESSOR)
features = preprocessor.transform(
    frame.iloc[selected].drop(columns=["id", "attack_cat", "label"])
).toarray().astype(np.float32)
risk = frame.iloc[selected][RISK_FEATURES].to_numpy(dtype=np.float64)

feature_path = OUTPUT / "scheduler_train_6000_features.npy"
risk_path = OUTPUT / "scheduler_train_6000_risk_features.npy"
index_path = OUTPUT / "SOURCE_INDEX.csv"
raw_path = OUTPUT / "scheduler_train_6000_raw_features.csv"
np.save(feature_path, features, allow_pickle=False)
np.save(risk_path, risk, allow_pickle=False)
frame.iloc[selected].drop(columns=["id", "attack_cat", "label"]).to_csv(
    raw_path, index=False
)
with index_path.open("x", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["sequence", "source_row_index", "feature_group_hex"])
    for sequence, source_index in enumerate(selected):
        writer.writerow([
            sequence,
            int(source_index),
            splits.iloc[source_index]["feature_group_hex"],
        ])

files = [feature_path, risk_path, raw_path, index_path]
manifest = {
    "status": "FROZEN_DEVICE_CALIBRATION_INPUT",
    "source_split": "scheduler_train only",
    "seed": SEED,
    "rows": ROWS,
    "official_test_access": "not read",
    "files": {
        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in files
    },
}
with (OUTPUT / "MANIFEST.json").open("x", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2)
    stream.write("\n")
print(json.dumps(manifest, indent=2))
