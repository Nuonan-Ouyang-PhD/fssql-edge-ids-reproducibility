#!/usr/bin/env python3
"""Validate the self-contained public technical release without raw datasets."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
errors: list[str] = []


def require(path: str) -> Path:
    candidate = ROOT / path
    if not candidate.is_file():
        errors.append(f"missing: {path}")
    return candidate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


for path in [
    "runtime/action_pool_numpy.py",
    "runtime/preprocessor_numpy.py",
    "runtime/risk_proxy_numpy.py",
    "online/pi4b8/pi4b8_online_worker.py",
    "online/pi4b8/powerz_hid_capture.py",
    "models/MODEL_SHA256.json",
    "preprocessing/preprocessor_portable.npz",
    "risk_proxy/risk_proxy.npz",
    "results/pi4b8/RUN_LEVEL_METRICS.csv",
    "results/official_test/OFFICIAL_TEST_RESULTS.csv",
]:
    require(path)

for source in list((ROOT / "runtime").glob("*.py")) + list((ROOT / "online").glob("**/*.py")):
    try:
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except SyntaxError as exc:
        errors.append(f"syntax: {source.relative_to(ROOT)}: {exc}")

q_paths = sorted((ROOT / "q_training/pi4b8_q_v1/tables").glob("fssql_r_seed_*.npz"))
if len(q_paths) != 5:
    errors.append(f"expected 5 Pi4B FSSQL-R Q tables, found {len(q_paths)}")
for path in q_paths:
    with np.load(path, allow_pickle=False) as artifact:
        if "q_values" not in artifact.files or artifact["q_values"].shape != (1280, 4):
            errors.append(f"invalid Q table: {path.relative_to(ROOT)}")

model_paths = sorted((ROOT / "models/artifacts").glob("*_revision.npz"))
if len(model_paths) != 4:
    errors.append(f"expected 4 portable detector artifacts, found {len(model_paths)}")
for path in model_paths:
    with np.load(path, allow_pickle=False) as artifact:
        if not artifact.files:
            errors.append(f"empty model artifact: {path.relative_to(ROOT)}")

with require("results/pi4b8/RUN_LEVEL_METRICS.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != 60:
    errors.append(f"expected 60 Pi4B run-level rows, found {len(rows)}")
if rows and len({(row["policy"], row["condition"], row["seed"]) for row in rows}) != 60:
    errors.append("Pi4B policy-condition-seed mapping is not one-to-one")

with require("results/official_test/OFFICIAL_TEST_RESULTS.csv").open(newline="", encoding="utf-8") as stream:
    official = list(csv.DictReader(stream))
if len(official) != 4 or any(int(row["rows"]) != 82332 for row in official):
    errors.append("official-test summary does not contain 4 detectors x 82,332 rows")

for forbidden in [
    ROOT / "data/raw/UNSW_NB15_training-set.csv",
    ROOT / "data/raw/UNSW_NB15_testing-set.csv",
    ROOT / ".formal_inputs",
]:
    if forbidden.exists():
        errors.append(f"non-public raw input unexpectedly present: {forbidden.relative_to(ROOT)}")

manifest = ROOT / "PUBLIC_RELEASE_MANIFEST_SHA256.txt"
if manifest.exists():
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file() or sha256(path) != digest:
            errors.append(f"checksum mismatch: {relative}")

if errors:
    print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
    raise SystemExit(1)

print(json.dumps({
    "status": "PUBLIC_TECHNICAL_RELEASE_PASS",
    "portable_models": len(model_paths),
    "pi4b_q_tables": len(q_paths),
    "pi4b_run_level_cells": len(rows),
    "official_test_detectors": len(official),
}, indent=2))

