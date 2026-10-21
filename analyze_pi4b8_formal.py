#!/usr/bin/env python3
"""Retain provenance analysis for historical V4 Pi4B runs.

The V4 technical-validity labels do not certify corrected true-n=16 execution.
They must not be used as the results of a future corrected Pi4B campaign.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from online.pi4b8.run_pi4b8_formal_matrix import (
    RunValidationError,
    validate_power_capture,
)


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "evidence/pi4b8_formal_matrix"
OUT = ROOT / "analysis/pi4b8_formal_v1"


class EvidenceError(RuntimeError):
    """A raw historical artifact does not satisfy its technical validity rules."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise EvidenceError(f"missing CSV: {path}")
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def json_object(path: Path) -> dict:
    if not path.is_file():
        raise EvidenceError(f"missing JSON: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"malformed JSON: {path}") from exc


def validated_energy(
    powerz_path: Path,
    handshake_path: Path,
    validation_path: Path,
    *,
    expected_run_id: str,
    minimum_roll_s: float = 5.0,
) -> dict:
    validation = json_object(validation_path)
    if validation.get("status") != "VALID_COMPLETE":
        raise EvidenceError(f"{expected_run_id}: run validation is not VALID_COMPLETE")
    if validation.get("run_id") != expected_run_id:
        raise EvidenceError(f"{expected_run_id}: run validation binding mismatch")
    if validation.get("powerz", {}).get("sha256") != sha256(powerz_path):
        raise EvidenceError(f"{expected_run_id}: POWER-Z hash mismatch")
    if validation.get("handshake_sha256") != sha256(handshake_path):
        raise EvidenceError(f"{expected_run_id}: handshake hash mismatch")

    handshake = json_object(handshake_path)
    if handshake.get("status") != "VALID_COMPLETE":
        raise EvidenceError(f"{expected_run_id}: handshake is not VALID_COMPLETE")
    try:
        result = validate_power_capture(
            powerz_path,
            handshake,
            run_id=expected_run_id,
            minimum_roll_s=minimum_roll_s,
        )
    except RunValidationError as exc:
        raise EvidenceError(str(exc)) from exc
    recorded_energy = validation.get("powerz", {}).get("energy_J")
    if recorded_energy is None or not np.isclose(
        result["energy_J"], float(recorded_energy), rtol=1e-12, atol=1e-9
    ):
        raise EvidenceError(f"{expected_run_id}: energy is not independently reproducible")
    return result


def accepted_cells(root: Path, base: Path) -> list[dict]:
    completeness = json_object(root / "PI4B8_FORMAL_EVIDENCE_COMPLETENESS_AFTER_RECOVERY.json")
    selected = []
    for item in completeness.get("runs", []):
        run_id = item["run_id"]
        attempt_id = run_id + "_attempt2" if item["execution_order"] in (40, 51) else run_id
        directory = base / attempt_id
        window_files = list(directory.glob("*/windows.csv"))
        if len(window_files) != 1:
            raise EvidenceError(
                f"{attempt_id}: expected one windows.csv, found {len(window_files)}"
            )
        selected.append(
            {
                "cell_run_id": run_id,
                "accepted_attempt_id": attempt_id,
                "execution_order": item["execution_order"],
                "policy": item["policy"],
                "condition": item["condition"],
                "seed": item["seed"],
                "pi_raw": str(window_files[0]),
                "powerz": str(directory / "POWERZ_raw.csv"),
                "handshake": str(directory / "HANDSHAKE.json"),
                "validation": str(directory / "RUN_VALIDATION.json"),
            }
        )
    if len(selected) != 60:
        raise EvidenceError(f"expected 60 accepted cells, found {len(selected)}")
    return selected


def bootstrap_ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(20260911)
    means = [np.mean(rng.choice(values, len(values), replace=True)) for _ in range(10000)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> None:
    selected = accepted_cells(ROOT, BASE)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "HISTORICAL_V4_PROVENANCE_ONLY",
        "accepted_cell_count": 60,
        "selection_rule": (
            "original attempt for 58 valid cells; _attempt2 for execution orders "
            "40 and 51; never mix attempts. These are historical V4 attempts, "
            "not corrected true-n=16 evidence."
        ),
        "runs": selected,
    }
    manifest_path = OUT / "ACCEPTED_RUN_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "ACCEPTED_RUN_MANIFEST_SHA256.txt").write_text(
        sha256(manifest_path) + "\n"
    )

    metrics = []
    for accepted in selected:
        raw = csv_rows(Path(accepted["pi_raw"]))
        if len(raw) != 600:
            raise EvidenceError(
                f"{accepted['accepted_attempt_id']}: expected 600 windows, found {len(raw)}"
            )
        values = lambda key: np.array([float(row[key]) for row in raw])
        counts = {key: sum(int(row[key]) for row in raw) for key in ("tp", "fp", "fn", "tn")}
        tp, fp, fn, tn = [counts[key] for key in ("tp", "fp", "fn", "tn")]
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        actions = {}
        for row in raw:
            actions[row["selected_action"]] = actions.get(row["selected_action"], 0) + 1
        e2e = values("e2e_ms")
        controller = values("controller_ms")
        detector = values("detector_inference_ms")
        temperature = values("temperature_C")
        switches = sum(
            row["previous_action"] not in ("", "None")
            and row["previous_action"] != row["selected_action"]
            for row in raw
        )
        power = validated_energy(
            Path(accepted["powerz"]),
            Path(accepted["handshake"]),
            Path(accepted["validation"]),
            expected_run_id=accepted["accepted_attempt_id"],
        )
        energy = power["energy_J"]
        metrics.append(
            {
                **accepted,
                **counts,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fpr": fp / (fp + tn) if fp + tn else 0,
                "cumulative_reward": float(values("reward").sum()),
                "mean_reward": float(values("reward").mean()),
                "p50_e2e": float(np.percentile(e2e, 50)),
                "p95_e2e": float(np.percentile(e2e, 95)),
                "p99_e2e": float(np.percentile(e2e, 99)),
                "max_e2e": float(e2e.max()),
                "mean_controller": float(controller.mean()),
                "p95_controller": float(np.percentile(controller, 95)),
                "p99_controller": float(np.percentile(controller, 99)),
                "mean_detector": float(detector.mean()),
                "switch_count": int(switches),
                "switch_rate": switches / len(raw),
                "switch_cost": float(values("reward_switch").sum()),
                "latency_violations": int((e2e > 40).sum()),
                "thermal_violations": int((temperature > 82).sum()),
                "safe_set_mean": float(values("safe_set_size").mean()),
                "mean_temperature": float(temperature.mean()),
                "max_temperature": float(temperature.max()),
                "mean_power_W": power["mean_power_W"],
                "peak_power_W": power["peak_power_W"],
                "total_energy_J": energy,
                "total_energy_Wh": energy / 3600,
                "J_per_window": energy / len(raw),
                "J_per_1000_records": energy * 1000 / len(raw),
                "J_per_correct_attack": energy / tp if tp else float("nan"),
                "action_composition": actions,
            }
        )

    excluded = {"pi_raw", "powerz", "handshake", "validation", "action_composition"}
    fields = [key for key in metrics[0] if key not in excluded] + ["action_composition"]
    with (OUT / "RUN_LEVEL_METRICS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    key: json.dumps(metric[key], sort_keys=True)
                    if key == "action_composition"
                    else metric[key]
                    for key in fields
                }
            )

    keys = [
        "f1", "precision", "recall", "fpr", "cumulative_reward", "mean_reward",
        "p50_e2e", "p95_e2e", "p99_e2e", "max_e2e", "mean_controller",
        "p95_controller", "p99_controller", "mean_detector", "switch_count",
        "switch_rate", "switch_cost", "latency_violations", "thermal_violations",
        "safe_set_mean", "mean_temperature", "max_temperature", "mean_power_W",
        "peak_power_W", "total_energy_J", "total_energy_Wh", "J_per_window",
        "J_per_1000_records", "J_per_correct_attack",
    ]
    pairs = []
    for condition in ("n16/B0", "n16/B1", "n16/B2"):
        for policy_a, policy_b in (
            ("FSSQL-R", "Safe-Greedy"),
            ("FSSQL-R", "Threshold"),
            ("FSSQL-R", "Static-Light"),
            ("Safe-Greedy", "Threshold"),
        ):
            for key in keys:
                a_values = np.array(
                    [m[key] for m in metrics if m["policy"] == policy_a and m["condition"] == condition]
                )
                b_values = np.array(
                    [m[key] for m in metrics if m["policy"] == policy_b and m["condition"] == condition]
                )
                difference = a_values - b_values
                pairs.append(
                    {
                        "metric": key,
                        "condition": condition,
                        "policy_a": policy_a,
                        "policy_b": policy_b,
                        "a_values": a_values.tolist(),
                        "b_values": b_values.tolist(),
                        "paired_difference_values": difference.tolist(),
                        "mean_difference": float(np.mean(difference)),
                        "sd_difference": float(np.std(difference, ddof=1)),
                        "median_difference": float(np.median(difference)),
                        "bootstrap_ci95_mean_difference": bootstrap_ci(difference),
                        "paired_t_p": float(ttest_rel(a_values, b_values).pvalue),
                        "wilcoxon_p_exact": float(wilcoxon(difference).pvalue)
                        if np.any(difference)
                        else 1.0,
                        "paired_cohen_dz": float(
                            np.mean(difference) / np.std(difference, ddof=1)
                        )
                        if np.std(difference, ddof=1) > 0
                        else float("nan"),
                    }
                )
    (OUT / "PAIRED_SEED_RESULTS.json").write_text(
        json.dumps(pairs, indent=2, allow_nan=True) + "\n"
    )
    (OUT / "ANALYSIS_README.md").write_text(
        "Historical V4 provenance analysis only. VALID_COMPLETE denotes the "
        "technical acquisition classification used at the time; it does not make "
        "these runs corrected true-n=16 evidence. The official test was sealed "
        "during V4 collection and was subsequently opened once for the frozen "
        "detector-generalization evaluation. Do not use these outputs as results "
        "from a future corrected Pi4B campaign.\n"
    )
    print(
        "PI4B8_HISTORICAL_V4_PROVENANCE_ANALYSIS_COMPLETE",
        len(metrics),
        "runs",
        len(pairs),
        "paired results",
    )


if __name__ == "__main__":
    try:
        main()
    except EvidenceError as exc:
        raise SystemExit(f"Pi4B8 analysis blocked: {exc}") from exc
