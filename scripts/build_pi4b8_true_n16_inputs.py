#!/usr/bin/env python3
"""Build the frozen compact true-n=16 scheduler-development batch maps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (5101, 5102, 5103, 5104, 5105)
BASE_ROWS = 6000
RECORDS_PER_WINDOW = 16
TRAINING_WINDOWS = 6000
EVALUATION_WINDOWS = 600


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def batch_indices(seed: int, windows: int) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(BASE_ROWS)
    positions = np.arange(windows * RECORDS_PER_WINDOW, dtype=np.int64) % BASE_ROWS
    result = order[positions].reshape(windows, RECORDS_PER_WINDOW).astype(np.int32)
    if np.any(np.apply_along_axis(lambda row: len(np.unique(row)) != 16, 1, result)):
        raise RuntimeError(f"seed {seed} contains a repeated record within a window")
    return result


def build(source: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite true-n16 input freeze: {output}")
    source_manifest_path = source / "RESTORATION_MANIFEST.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text())
    if (
        source_manifest.get("status") != "SCHEDULER_TRAIN_6000_RESTORATION_PASS"
        or source_manifest.get("official_test_access") != "not used"
        or int(source_manifest.get("rows", 0)) != BASE_ROWS
    ):
        raise RuntimeError("scheduler-development source has not passed restoration")
    required = (
        "scheduler_train_6000_features.npy",
        "scheduler_train_6000_risk_features.npy",
        "scheduler_train_6000_raw_features.csv",
        "scheduler_train_6000_labels.npy",
    )
    source_hashes: dict[str, str] = {}
    for name in required:
        path = source / name
        expected = source_manifest["files"][name]["sha256"]
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"restored scheduler input hash mismatch: {name}")
        source_hashes[name] = observed

    output.mkdir(parents=True)
    files: dict[str, dict[str, object]] = {}
    training_by_seed: dict[str, str] = {}
    evaluation_by_seed: dict[str, str] = {}
    for seed in SEEDS:
        training = batch_indices(seed, TRAINING_WINDOWS)
        evaluation = batch_indices(seed, EVALUATION_WINDOWS)
        if not np.array_equal(training[:EVALUATION_WINDOWS], evaluation):
            raise RuntimeError(f"training/evaluation prefix mismatch for seed {seed}")
        if not np.all(np.bincount(training.ravel(), minlength=BASE_ROWS) == 16):
            raise RuntimeError(f"training schedule is not balanced for seed {seed}")
        train_name = f"training_batch_indices_seed_{seed}.npy"
        eval_name = f"evaluation_batch_indices_seed_{seed}.npy"
        np.save(output / train_name, training, allow_pickle=False)
        np.save(output / eval_name, evaluation, allow_pickle=False)
        training_by_seed[str(seed)] = train_name
        evaluation_by_seed[str(seed)] = eval_name

    np.save(
        output / "training_backlog_zero.npy",
        np.zeros(TRAINING_WINDOWS, dtype=np.float32),
        allow_pickle=False,
    )
    np.save(
        output / "evaluation_backlog_zero.npy",
        np.zeros(EVALUATION_WINDOWS, dtype=np.float32),
        allow_pickle=False,
    )
    for path in sorted(output.glob("*.npy")):
        array = np.load(path, allow_pickle=False)
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

    manifest = {
        "status": "PI4B8_TRUE_N16_INPUT_FREEZE_PASS",
        "official_test_access": "not used",
        "source_scope": "frozen scheduler_train development rows only",
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_files": source_hashes,
        "base_rows": BASE_ROWS,
        "records_per_window": RECORDS_PER_WINDOW,
        "training_windows": TRAINING_WINDOWS,
        "evaluation_windows": EVALUATION_WINDOWS,
        "seeds": list(SEEDS),
        "batch_rule": (
            "order=default_rng(seed).permutation(6000); "
            "index[w,j]=order[(16*w+j)%6000]"
        ),
        "condition_matching": "same seed schedule for n16/B0, n16/B1 and n16/B2",
        "backlog_semantics": "zero queued records in the stationary synchronous loop",
        "training_batch_indices_by_seed": training_by_seed,
        "evaluation_batch_indices_by_seed": evaluation_by_seed,
        "files": files,
    }
    manifest_path = output / "PI4B8_TRUE_N16_INPUT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=ROOT / "device_calibration/inputs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "device_calibration/inputs/pi4b8_true_n16_v1",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.source.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
