#!/usr/bin/env python3
"""Verify NumPy portable inference against frozen PyTorch artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.action_pool_numpy import NumpyActionModel  # noqa: E402
from scripts.train_action_pool import LucidRevision, TinyDLRevision  # noqa: E402


OUTPUT = ROOT / "models" / "PORTABLE_INFERENCE_PARITY.json"
if OUTPUT.exists():
    raise FileExistsError(f"refusing to overwrite parity result: {OUTPUT}")

frame = pd.read_csv(ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv")
splits = pd.read_csv(ROOT / "data" / "SPLIT_MANIFEST.csv")
validation = frame.loc[splits["split"].eq("validation")].iloc[:512]
preprocessor = joblib.load(ROOT / "preprocessing" / "preprocessor.joblib")
features = preprocessor.transform(
    validation.drop(columns=["id", "attack_cat", "label"])
).toarray().astype(np.float32)

checks = {}
for name, torch_model in [
    ("lucid_revision", LucidRevision(kernels=32, dropout=0.2)),
    ("tinydl_revision", TinyDLRevision(features.shape[1], dropout=0.1)),
]:
    torch_model.load_state_dict(torch.load(
        ROOT / "models" / "artifacts" / f"{name}.pt",
        weights_only=True,
    ))
    torch_model.eval()
    with torch.no_grad():
        expected = torch.sigmoid(torch_model(torch.from_numpy(features))).numpy()
    observed = NumpyActionModel(
        ROOT / "models" / "artifacts" / f"{name}.npz"
    ).predict_proba(features)
    difference = float(np.max(np.abs(expected - observed)))
    checks[name] = {
        "samples": len(features),
        "maximum_absolute_probability_difference": difference,
        "status": "PASS" if difference <= 1e-5 else "FAIL",
    }

for name in ("fisvdd_revision", "oi_svdd_as_elm_revision"):
    probabilities = NumpyActionModel(
        ROOT / "models" / "artifacts" / f"{name}.npz"
    ).predict_proba(features)
    checks[name] = {
        "samples": len(features),
        "finite_probabilities": bool(np.isfinite(probabilities).all()),
        "probability_range": [float(probabilities.min()), float(probabilities.max())],
        "status": "PASS" if np.isfinite(probabilities).all() else "FAIL",
    }

result = {
    "status": "PASS" if all(check["status"] == "PASS" for check in checks.values()) else "FAIL",
    "scope": "first 512 validation rows; official test not accessed",
    "checks": checks,
}
with OUTPUT.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps(result, indent=2))
