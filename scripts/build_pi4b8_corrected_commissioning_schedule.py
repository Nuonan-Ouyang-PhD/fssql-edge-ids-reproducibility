#!/usr/bin/env python3
"""Build the frozen non-formal true-n16 commissioning schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "device_calibration/inputs/pi4b8_true_n16_v1/commissioning_batch_indices_seed_5099.npy"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite commissioning schedule: {output}")
    freeze = json.loads(
        (ROOT / "protocols/PI4B8_TRUE_N16_INPUT_FREEZE_V1.json").read_text()
    )
    if (
        freeze.get("base_rows") != 6000
        or freeze.get("records_per_window") != 16
        or freeze.get("evaluation_windows") != 600
    ):
        raise SystemExit("true-n16 input freeze does not match commissioning scope")
    seed = 5099
    order = np.random.default_rng(seed).permutation(6000)
    indices = order[np.arange(600 * 16, dtype=np.int64) % 6000].reshape(600, 16)
    if any(len(set(row.tolist())) != 16 for row in indices):
        raise SystemExit("commissioning schedule repeats a row within a window")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, indices, allow_pickle=False)
    print("PI4B8_CORRECTED_COMMISSIONING_SCHEDULE_BUILD_PASS")
    print(f"seed={seed} shape={indices.shape} output={output}")


if __name__ == "__main__":
    main()
