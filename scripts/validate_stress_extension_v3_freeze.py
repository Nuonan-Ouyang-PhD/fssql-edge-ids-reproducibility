#!/usr/bin/env python3
"""Validate the Stress Extension V3 preregistration without reading test labels."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/pi3b_stress_extension_v3.json"
MATRIX_PATH = ROOT / "configs/pi3b_stress_extension_v3_matrix.csv"
OUTPUT_PATH = ROOT / "provenance/STRESS_EXTENSION_V3_FREEZE_VALIDATION.json"
sys.path.insert(0, str(ROOT / "runtime"))
from action_pool_numpy import NumpyActionModel  # noqa: E402
from preprocessor_numpy import PortablePreprocessor  # noqa: E402
from risk_proxy_numpy import RiskProxy  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


config = json.loads(CONFIG_PATH.read_text())
matrix_rows = list(csv.DictReader(MATRIX_PATH.open(newline="", encoding="utf-8")))
require(
    config["status"] == "PREREGISTERED_BEFORE_STRESS_EXTENSION_V3_POLICY_EXECUTION",
    "config is not preregistered",
)
require(
    config["policies"]
    == ["Static-Light", "Round-Robin", "Safe-Greedy", "Threshold", "Unshielded-Q", "FSSQL-R"],
    "policy set mismatch",
)
require(config["regimes"] == ["binding_resource"], "regime mismatch")
require(config["windows_per_run"] == 600 and config["decision_window_ms"] == 100.0, "run length mismatch")
require(config["constraints"] == {"latency_cap_ms": 40.0, "thermal_cap_c": 82.0, "emergency_stop_c": 58.0}, "constraint mismatch")
workload = config["workloads"]["binding_resource"]
require(workload["n_records"] == 16 and workload["background"] == "B0", "selected workload mismatch")

for record in config["frozen_inputs"].values():
    require(sha256(ROOT / record["path"]) == record["sha256"], f"frozen input mismatch: {record['path']}")
for name in ("collector", "campaign_runner"):
    require(
        sha256(ROOT / config["runtime"][name]) == config["runtime"][f"{name}_sha256"],
        f"runtime hash mismatch: {name}",
    )
require(
    sha256(ROOT / config["execution_order"]["matrix_generator"])
    == config["execution_order"]["matrix_generator_sha256"],
    "matrix generator hash mismatch",
)
require(sha256(MATRIX_PATH) == config["execution_order"]["expected_matrix_sha256"], "matrix hash mismatch")

selection_path = ROOT / config["frozen_inputs"]["binding_selection_validation"]["path"]
selection = json.loads(selection_path.read_text())
require(selection["status"] == "BINDING_STRESS_V3_INDEPENDENT_VALIDATION_PASS", "binding selection not validated")
require(selection["selected_candidate"] == {"candidate_index": 4, "n_records": 16, "background": "B0"}, "binding selection mismatch")
selected_result = selection["candidate_results"][-1]
require(selected_result["qualified"] is True and selected_result["binding_fraction"] == 1.0, "selected candidate not qualifying")
require(selected_result["safe_set_size_counts"] == {"0": 0, "1": 0, "2": 0, "3": 1800, "4": 0}, "selected safe-set distribution mismatch")
require(selected_result["distinct_admissible_actions"] == config["guard_lock"]["expected_development_safe_set"], "selected admissible set mismatch")

expected_runs = len(config["policies"]) * len(config["evaluation_seeds"])
require(len(matrix_rows) == expected_runs == 60, "matrix row count mismatch")
require([int(row["global_order"]) for row in matrix_rows] == list(range(1, 61)), "global order mismatch")
observed_pairs = Counter((int(row["seed"]), row["policy"]) for row in matrix_rows)
expected_pairs = Counter((seed, policy) for seed in config["evaluation_seeds"] for policy in config["policies"])
require(observed_pairs == expected_pairs, "matched seed/policy coverage mismatch")
training_map = config["q_learning"]["evaluation_seed_to_training_seed"]
for seed_block, seed in enumerate(config["evaluation_seeds"]):
    block = matrix_rows[seed_block * len(config["policies"]):(seed_block + 1) * len(config["policies"])]
    expected_order = np.random.default_rng(config["execution_order"]["base_seed"] + seed).permutation(len(config["policies"]))
    require(
        [row["policy"] for row in block] == [config["policies"][int(index)] for index in expected_order],
        f"counterbalance order mismatch: seed {seed}",
    )
    for within_seed_order, row in enumerate(block):
        require(int(row["seed_block"]) == seed_block and int(row["within_seed_order"]) == within_seed_order, f"block order mismatch: seed {seed}")
        require(row["regime"] == "binding_resource" and row["n_records"] == "16" and row["background"] == "B0", f"workload row mismatch: seed {seed}")
        require(row["windows"] == "600" and row["decision_window_ms"] == "100.0", f"duration row mismatch: seed {seed}")
        expected_training_seed = str(training_map[str(seed)]) if row["policy"] in ("Unshielded-Q", "FSSQL-R") else ""
        require(row["training_seed"] == expected_training_seed, f"training seed mismatch: seed {seed}/{row['policy']}")
        require(row["attempt"] == "1" and row["status"] == "PLANNED", f"attempt status mismatch: seed {seed}/{row['policy']}")

q_config_path = ROOT / config["q_learning"]["config"]
q_manifest_path = ROOT / config["q_learning"]["manifest"]
require(sha256(q_config_path) == config["q_learning"]["config_sha256"], "Q config hash mismatch")
require(sha256(q_manifest_path) == config["q_learning"]["manifest_sha256"], "Q manifest hash mismatch")
q_manifest = json.loads(q_manifest_path.read_text())
require(q_manifest["status"] == "Q_TRAINING_AND_STABILITY_DIAGNOSTIC_PASS", "Q manifest status mismatch")
for relative, expected_hash in q_manifest["artifact_sha256"].items():
    require(sha256(ROOT / "q_training" / relative) == expected_hash, f"Q artifact mismatch: {relative}")

input_manifest_path = ROOT / config["official_test"]["input_manifest"]
require(sha256(input_manifest_path) == config["official_test"]["accepted_input_manifest_sha256"], "official input manifest mismatch")
input_manifest = json.loads(input_manifest_path.read_text())
require(input_manifest["status"] == "OFFICIAL_TEST_OPENED_AND_MATCHED_TRACES_FROZEN", "official input status mismatch")
require(input_manifest["policy_config_sha256"] == config["official_test"]["formal_matrix_v1_config_sha256"], "official trace lineage mismatch")
input_dir = input_manifest_path.parent
for relative, expected_hash in input_manifest["artifact_sha256"].items():
    require(sha256(input_dir / relative) == expected_hash, f"official trace artifact mismatch: {relative}")

trace_index = list(csv.DictReader((input_dir / "trace_index.csv").open(newline="", encoding="utf-8")))
for seed in config["evaluation_seeds"]:
    indices = [int(row["array_index"]) for row in trace_index if int(row["seed"]) == seed]
    require(len(indices) == len(set(indices)) == 600, f"trace index count mismatch: seed {seed}")
    batches = [
        [indices[(window * 16 + offset) % 600] for offset in range(16)]
        for window in range(600)
    ]
    require(Counter(index for batch in batches for index in batch) == Counter({index: 16 for index in indices}), f"batch multiplicity mismatch: seed {seed}")
raw_fields = next(csv.reader((input_dir / "trace_raw_features.csv").open(newline="", encoding="utf-8")))
require(not set(config["official_test"]["scheduler_forbidden_fields"]) & set(raw_fields), "forbidden scheduler field present")

raw_path = ROOT / config["frozen_inputs"]["scheduler_raw"]["path"]
features_path = ROOT / config["frozen_inputs"]["scheduler_features"]["path"]
risk_features_path = ROOT / config["frozen_inputs"]["scheduler_risk_features"]["path"]
raw_rows = list(csv.DictReader(raw_path.open(newline="", encoding="utf-8")))[:16]
features = np.load(features_path, allow_pickle=False)[:16]
risk_features = np.load(risk_features_path, allow_pickle=False)[:16]
transformed = PortablePreprocessor(ROOT / config["frozen_inputs"]["preprocessor"]["path"]).transform(raw_rows)
require(np.array_equal(transformed, features), "scheduler-train 16-record preprocessing parity mismatch")
risks = RiskProxy(ROOT / config["frozen_inputs"]["risk_proxy"]["path"]).predict(risk_features)
require(len(risks) == 16 and np.isfinite(risks).all() and np.isfinite(np.mean(risks)), "scheduler-train batch risk smoke failed")
model_manifest = json.loads((ROOT / config["frozen_inputs"]["model_manifest"]["path"]).read_text())
model_output_ranges = {}
for action in config["action_order"]:
    model_path = ROOT / f"models/artifacts/{action}.npz"
    require(sha256(model_path) == model_manifest["files"][str(model_path.relative_to(ROOT))]["sha256"], f"model artifact mismatch: {action}")
    probabilities = NumpyActionModel(model_path).predict_proba(transformed)
    require(len(probabilities) == 16 and np.isfinite(probabilities).all(), f"16-record model smoke failed: {action}")
    model_output_ranges[action] = [float(np.min(probabilities)), float(np.max(probabilities))]

require(len(config["required_window_fields"]) == len(set(config["required_window_fields"])) == 61, "window schema mismatch")
validation = {
    "status": "STRESS_EXTENSION_V3_FREEZE_VALIDATION_PASS",
    "config_sha256": sha256(CONFIG_PATH),
    "matrix_sha256": sha256(MATRIX_PATH),
    "runs": 60,
    "windows": 36000,
    "records_evaluated": 576000,
    "matrix_coverage": "6 policies x 10 matched seeds, exactly once",
    "selected_workload": {"n_records": 16, "background": "B0"},
    "selected_development_safe_set": selected_result["distinct_admissible_actions"],
    "selected_development_safe_set_size_counts": selected_result["safe_set_size_counts"],
    "q_artifact_hashes_reproduced": len(q_manifest["artifact_sha256"]),
    "official_trace_artifact_hashes_reproduced_without_loading_labels": len(input_manifest["artifact_sha256"]),
    "official_test_labels_loaded": False,
    "scheduler_train_batch_smoke": {
        "records": 16,
        "preprocessing_exact_parity": True,
        "finite_risk_scores": True,
        "finite_model_probabilities": True,
        "model_probability_ranges": model_output_ranges,
    },
    "runtime_sha256": {
        name: config["runtime"][f"{name}_sha256"]
        for name in ("collector", "campaign_runner")
    },
}
OUTPUT_PATH.write_text(json.dumps(validation, indent=2) + "\n")
print(json.dumps(validation, indent=2))
