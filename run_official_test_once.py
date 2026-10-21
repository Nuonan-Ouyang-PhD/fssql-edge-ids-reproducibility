#!/usr/bin/env python3
"""Run the frozen one-shot detector evaluation on the official test file."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from runtime.action_pool_numpy import NumpyActionModel
from runtime.preprocessor_numpy import PortablePreprocessor


ROOT = Path(__file__).resolve().parent
OFFICIAL_TEST_CSV = ROOT / "data/raw/UNSW_NB15_testing-set.csv"
MODEL_HYPERPARAMETERS = ROOT / "models/MODEL_HYPERPARAMETERS.json"
OUTPUT_DIR = ROOT / "results/official_test"
OPEN_LOCK = OUTPUT_DIR / "OFFICIAL_TEST_OPENED_ONCE.json"
DETECTORS = [
    ("FISVDD", "fisvdd_revision"),
    ("LUCID", "lucid_revision"),
    ("TinyDL", "tinydl_revision"),
    ("OI-SVDD+AS-ELM", "oi_svdd_as_elm_revision"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_detector_specs(path: Path = MODEL_HYPERPARAMETERS) -> list[tuple[str, str, float]]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "FROZEN_AFTER_VALIDATION":
        raise RuntimeError("model hyperparameters are not frozen after validation")
    actions = payload.get("actions")
    if not isinstance(actions, dict):
        raise RuntimeError("model hyperparameters are missing actions")

    specs = []
    for detector, artifact in DETECTORS:
        record = actions.get(detector)
        if not isinstance(record, dict) or "decision_threshold_selected_on_validation" not in record:
            raise RuntimeError(f"missing frozen validation threshold for {detector}")
        threshold = float(record["decision_threshold_selected_on_validation"])
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise RuntimeError(f"invalid frozen validation threshold for {detector}: {threshold}")
        specs.append((detector, artifact, threshold))
    return specs


def main() -> None:
    if OPEN_LOCK.exists():
        raise RuntimeError(
            "official test was already opened; this one-shot evaluator is permanently locked"
        )
    specs = load_detector_specs()
    rows = list(csv.DictReader(OFFICIAL_TEST_CSV.open(encoding="utf-8-sig", newline="")))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    features = [
        {key: value for key, value in row.items() if key not in ("id", "attack_cat", "label")}
        for row in rows
    ]
    transformed = PortablePreprocessor(
        ROOT / "preprocessing/preprocessor_portable.npz"
    ).transform(features)

    results = []
    for detector, artifact, threshold in specs:
        model = NumpyActionModel(ROOT / f"models/artifacts/{artifact}.npz")
        scores = model.predict_proba(transformed)
        predictions = (scores >= threshold).astype(np.int8)
        tp = int(((predictions == 1) & (labels == 1)).sum())
        fp = int(((predictions == 1) & (labels == 0)).sum())
        fn = int(((predictions == 0) & (labels == 1)).sum())
        tn = int(((predictions == 0) & (labels == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        results.append({
            "detector": detector,
            "threshold_frozen": threshold,
            "rows": len(rows),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "auroc": float(roc_auc_score(labels, scores)),
            "pr_auc": float(average_precision_score(labels, scores)),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "OFFICIAL_TEST_RESULTS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    validation = {
        "status": "PASS_ONE_SHOT_COMPLETE",
        "rows_evaluated": len(rows),
        "detectors": results,
        "test_sha256": sha256(OFFICIAL_TEST_CSV),
        "threshold_source": str(MODEL_HYPERPARAMETERS.relative_to(ROOT)),
        "threshold_source_sha256": sha256(MODEL_HYPERPARAMETERS),
        "official_test": "opened_once",
        "no_tuning": True,
        "development_phase": "closed",
    }
    (OUTPUT_DIR / "OFFICIAL_TEST_VALIDATION.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    summary = (
        "# Official UNSW-NB15 one-shot final evaluation\n\n"
        "All 82,332 official test rows were transformed with the frozen portable "
        "preprocessor and evaluated once with frozen validation thresholds. No retraining, "
        "threshold selection, Q/guard/workload/policy changes were performed. These results "
        "are independent detector-generalization evidence and are separate from Pi4B8 "
        "scheduler-development online results.\n\n"
        + "\n".join(
            f"- {result['detector']}: F1={result['f1']:.6f}, "
            f"precision={result['precision']:.6f}, recall={result['recall']:.6f}, "
            f"FPR={result['fpr']:.6f}, AUROC={result['auroc']:.6f}, "
            f"PR-AUC={result['pr_auc']:.6f}"
            for result in results
        )
        + "\n"
    )
    (OUTPUT_DIR / "OFFICIAL_TEST_SUMMARY.md").write_text(summary)
    print("OFFICIAL_TEST_ONE_SHOT_PASS", len(rows))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
