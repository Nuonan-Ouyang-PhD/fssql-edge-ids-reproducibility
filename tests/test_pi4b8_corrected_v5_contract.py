import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from online.pi4b8 import corrected_v5_contract as contract
from online.pi4b8 import pi4b8_corrected_v5_orchestrator as orchestrator


def make_package(root: Path):
    package = root / "package"
    package.mkdir(parents=True)
    role_paths = {
        "worker": "pi4b8_online_worker.py",
        "runtime_contract": "runtime/corrected_v5_contract.py",
        "runtime_helper:action_pool_numpy": "runtime/action_pool_numpy.py",
        "runtime_helper:preprocessor_numpy": "runtime/preprocessor_numpy.py",
        "runtime_helper:risk_proxy_numpy": "runtime/risk_proxy_numpy.py",
        "runtime_config": "config/runtime.json",
        "model_hyperparameters": "config/MODEL_HYPERPARAMETERS.json",
        "validation_metrics": "config/VALIDATION_METRICS.csv",
        "q_config": "config/q.json",
        "q_manifest": "config/q_manifest.json",
        "preprocessor": "preprocessing/preprocessor_portable.npz",
        "risk_proxy": "risk_proxy/risk_proxy.npz",
        "guard:latency_predictor": "guard/latency_predictor.csv",
        "guard:thermal_predictor": "guard/thermal_predictor.json",
        "guard:margins": "guard/guard_margins.json",
        "powerz:capture": "powerz/powerz_hid_capture.py",
        "powerz:validator": "powerz/powerz_evidence_validator.py",
        "powerz:specification": "powerz/POWER_Z_HID_ACQUISITION_V1.md",
        "base:raw_features": "inputs/base/raw_features.csv",
        "base:features": "inputs/base/features.npy",
        "base:risk_features": "inputs/base/risk_features.npy",
        "base:backlog": "inputs/base/backlog.npy",
        "base:labels": "scoring/base/labels.npy",
    }
    for action in contract.ACTIONS:
        role_paths[f"model:{action}"] = f"models/{action}.npz"
    for seed in contract.SEEDS:
        role_paths[f"q_table:{seed}"] = f"q_tables/fssql_r_seed_{seed}.npz"
        role_paths[f"batch_indices:{seed}"] = f"inputs/batches/seed_{seed}.npy"
    role_paths[
        f"commissioning_batch_indices:{contract.COMMISSIONING_SEED}"
    ] = f"inputs/commissioning/seed_{contract.COMMISSIONING_SEED}.npy"

    artifacts = []
    for role, relative in role_paths.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if role == "runtime_config":
            path.write_text(
                json.dumps(
                    {
                        "action_order": list(contract.ACTIONS),
                        "constraints": {
                            "latency_cap_ms": 40.0,
                            "thermal_cap_c": 82.0,
                        },
                        "state": {
                            "risk_bin_upper_edges": [0.25, 0.5, 0.75],
                            "temperature_C_upper_edges": [35.0, 45.0, 55.0],
                            "energy_residual_upper_edges": [0.25, 0.5, 0.75],
                            "backlog_upper_edges": [0.5, 2.0, 8.0],
                            "number_of_states": 1280,
                        },
                        "energy_state_initialization": 1.0,
                        "reward": {
                            "energy_weight": 0.1,
                            "backlog_weight": 0.05,
                            "switch_weight": 0.02,
                            "backlog_normalizer": 256.0,
                        },
                        "validation_action_quality": {
                            action: {"tpr": 0.8, "fpr": 0.2}
                            for action in contract.ACTIONS
                        },
                        "normalized_energy_cost_by_condition": {
                            condition: {action: 0.3 for action in contract.ACTIONS}
                            for condition in contract.CONDITIONS
                        },
                        "energy_state_decrement_by_condition": {
                            condition: {action: 0.0001 for action in contract.ACTIONS}
                            for condition in contract.CONDITIONS
                        },
                        "workloads": {
                            "n16/B0": {"background_duty": 0.0},
                            "n16/B1": {
                                "background_duty": 0.5,
                                "pinned_cpu": 3,
                                "period_ms": 100.0,
                            },
                            "n16/B2": {
                                "background_duty": 1.0,
                                "pinned_cpu": 3,
                                "period_ms": 100.0,
                            },
                        },
                        "threshold_controller": {
                            "T_hi_C": 80.0,
                            "T_lo_C": 70.0,
                            "q_hi_records": 512.0,
                            "q_lo_records": 256.0,
                            "p_hi": 0.7,
                            "light_action": contract.ACTIONS[0],
                            "heavy_action": contract.ACTIONS[2],
                        },
                    }
                )
            )
        elif role == "model_hyperparameters":
            names = {
                "fisvdd_revision": "FISVDD",
                "lucid_revision": "LUCID",
                "tinydl_revision": "TinyDL",
                "oi_svdd_as_elm_revision": "OI-SVDD+AS-ELM",
            }
            path.write_text(
                json.dumps(
                    {
                        "actions": {
                            name: {"decision_threshold_selected_on_validation": 0.5}
                            for name in names.values()
                        }
                    }
                )
            )
        elif role == "guard:margins":
            path.write_text(
                json.dumps(
                    {
                        "fixed_latency_buffer_ms": 1.0,
                        "epsilon_tau_ms_by_action": {
                            action: 2.0 for action in contract.ACTIONS
                        },
                        "epsilon_T_C_shared": 1.0,
                        "fixed_thermal_buffer_c": 1.0,
                    }
                )
            )
        elif role.startswith("batch_indices:") or role.startswith(
            "commissioning_batch_indices:"
        ):
            np.save(path, np.arange(9600, dtype=np.int64).reshape(600, 16) % 6000)
        elif role.startswith("q_table:"):
            np.savez(path, q_values=np.tile([9.0, 100.0, 8.0, 7.0], (1280, 1)))
        else:
            path.write_bytes((role + "\n").encode())
        artifacts.append(
            {
                "role": role,
                "source_path": f"sources/{relative}",
                "package_path": relative,
                "bytes": path.stat().st_size,
                "sha256": contract.sha256(path),
            }
        )

    cells = []
    for condition in contract.CONDITIONS:
        for seed in contract.SEEDS:
            cells.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "runtime": {
                        "raw_features": "base:raw_features",
                        "features": "base:features",
                        "risk_features": "base:risk_features",
                        "backlog": "base:backlog",
                        "batch_indices": f"batch_indices:{seed}",
                    },
                    "scoring": {"labels": "base:labels"},
                }
            )
    shared = {
        "official_test_access": "forbidden",
        "source_commit": "0" * 40,
        "base_records": 6000,
        "windows_per_run": 600,
        "records_per_window": 16,
        "conditions": list(contract.CONDITIONS),
        "seeds": list(contract.SEEDS),
        "artifacts": artifacts,
        "cells": cells,
        "commissioning_cell_required": True,
        "commissioning_preregistration_sha256": "f" * 64,
        "commissioning_cell": {
            "scientific_use": "COMMISSIONING_ONLY_EXCLUDED_FROM_FORMAL_STATISTICS",
            "policy": contract.COMMISSIONING_POLICY,
            "condition": contract.COMMISSIONING_CONDITION,
            "seed": contract.COMMISSIONING_SEED,
            "runtime": {
                "raw_features": "base:raw_features",
                "features": "base:features",
                "risk_features": "base:risk_features",
                "backlog": "base:backlog",
                "batch_indices": (
                    f"commissioning_batch_indices:{contract.COMMISSIONING_SEED}"
                ),
            },
            "scoring": {"labels": "base:labels"},
        },
    }
    source = {
        **shared,
        "schema_version": contract.SOURCE_SCHEMA,
        "status": contract.SOURCE_STATUS,
    }
    source_path = package / contract.SOURCE_MANIFEST_NAME
    source_path.write_text(json.dumps(source, sort_keys=True))
    manifest = {
        **shared,
        "schema_version": contract.PACKAGE_SCHEMA,
        "status": contract.PACKAGE_STATUS,
        "source_manifest_sha256": contract.sha256(source_path),
        "execution_authorized": False,
    }
    manifest_path = package / contract.PACKAGE_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    return package, manifest_path, manifest


class PackageContractTests(unittest.TestCase):
    def test_powerz_validator_is_required_at_fixed_package_path(self):
        with tempfile.TemporaryDirectory() as directory:
            package, _, manifest = make_package(Path(directory))
            manifest["artifacts"] = [
                artifact
                for artifact in manifest["artifacts"]
                if artifact["role"] != "powerz:validator"
            ]
            with self.assertRaisesRegex(
                contract.ContractError, "powerz:validator"
            ):
                contract.validate_package_manifest(manifest)
            source = json.loads(
                (package / contract.SOURCE_MANIFEST_NAME).read_text()
            )
            source["artifacts"] = [
                artifact
                for artifact in source["artifacts"]
                if artifact["role"] != "powerz:validator"
            ]
            with self.assertRaisesRegex(
                contract.ContractError, "powerz:validator"
            ):
                contract.validate_source_manifest(source)

            _, _, manifest = make_package(Path(directory) / "wrong-path")
            artifact = next(
                item
                for item in manifest["artifacts"]
                if item["role"] == "powerz:validator"
            )
            artifact["package_path"] = "powerz/other_validator.py"
            with self.assertRaisesRegex(
                contract.ContractError, "must use package path"
            ):
                contract.validate_package_manifest(manifest)

    def test_plan_has_60_cells_and_never_targets_v4(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, manifest_path, _ = make_package(root)
            plan = orchestrator.build_plan(
                package,
                manifest_path,
                remote_package_root=Path(
                    "/home/pi/tsusc_major_revision/pi4b8_corrected_runtime_v5"
                ),
                remote_output_root=Path(
                    "/home/pi/tsusc_major_revision/pi4b8_corrected_runs_v5"
                ),
            )
            self.assertEqual(len(plan["runs"]), 60)
            self.assertEqual(
                len(
                    {
                        (run["policy"], run["condition"], run["seed"])
                        for run in plan["runs"]
                    }
                ),
                60,
            )
            self.assertEqual(
                {run["seed"] for run in plan["runs"]}, set(contract.SEEDS)
            )
            self.assertTrue(
                all(run["seed"] != contract.COMMISSIONING_SEED for run in plan["runs"])
            )
            self.assertTrue(
                all("formal_runtime_v4" not in run["worker_shell_command"] for run in plan["runs"])
            )
            self.assertEqual(
                plan["status"], "CORRECTED_V5_PLAN_ONLY_NOT_EXECUTION_AUTHORIZATION"
            )

    def test_full_package_and_worker_cli_are_hash_and_seed_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, manifest_path, _ = make_package(root)
            output = root / "runs" / "run-1"
            command = contract.build_worker_argv(
                package,
                manifest_path,
                run_id="run-1",
                policy="FSSQL-R",
                condition="n16/B2",
                seed=5103,
                output=output,
            )
            self.assertEqual(command[:2], ["python3", "-B"])
            self.assertIn("--batch-indices", command)
            self.assertIn(str((package / "inputs/batches/seed_5103.npy").resolve()), command)
            self.assertIn(str((package / "scoring/base/labels.npy").resolve()), command)
            self.assertIn(str((package / "q_tables/fssql_r_seed_5103.npz").resolve()), command)
            self.assertNotIn(str((package / "q_tables/fssql_r_seed_5102.npz").resolve()), command)

    def test_non_fssql_cli_does_not_receive_q_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, manifest_path, _ = make_package(root)
            command = contract.build_worker_argv(
                package,
                manifest_path,
                run_id="run-2",
                policy="Threshold",
                condition="n16/B0",
                seed=5101,
                output=root / "runs/run-2",
            )
            self.assertNotIn("--q-table", command)
            self.assertNotIn("--q-config", command)
            self.assertNotIn("--q-manifest", command)

    def test_commissioning_is_hash_bound_and_not_a_formal_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, manifest_path, manifest = make_package(root)
            command = contract.build_worker_argv(
                package,
                manifest_path,
                run_id="commissioning-1",
                policy=contract.COMMISSIONING_POLICY,
                condition=contract.COMMISSIONING_CONDITION,
                seed=contract.COMMISSIONING_SEED,
                output=root / "runs/commissioning-1",
            )
            self.assertIn(
                str(
                    (
                        package
                        / f"inputs/commissioning/seed_{contract.COMMISSIONING_SEED}.npy"
                    ).resolve()
                ),
                command,
            )
            self.assertEqual(len(manifest["cells"]), 15)
            with self.assertRaisesRegex(contract.ContractError, "no unique corrected cell"):
                contract.build_worker_argv(
                    package,
                    manifest_path,
                    run_id="bad-commissioning",
                    policy="Threshold",
                    condition=contract.COMMISSIONING_CONDITION,
                    seed=contract.COMMISSIONING_SEED,
                    output=root / "runs/bad-commissioning",
                )

    def test_commissioning_requires_prereg_hash_and_distinct_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, manifest = make_package(Path(directory))
            manifest.pop("commissioning_preregistration_sha256")
            with self.assertRaisesRegex(
                contract.ContractError, "commissioning_preregistration_sha256"
            ):
                contract.validate_package_manifest(manifest)

            _, _, manifest = make_package(Path(directory) / "alias")
            manifest["commissioning_cell"]["runtime"][
                "batch_indices"
            ] = "batch_indices:5101"
            with self.assertRaisesRegex(contract.ContractError, "distinct"):
                contract.validate_package_manifest(manifest)

            _, _, manifest = make_package(Path(directory) / "label-leak")
            manifest["commissioning_cell"]["runtime"]["risk_features"] = "base:labels"
            with self.assertRaisesRegex(contract.ContractError, "leaks scoring"):
                contract.validate_package_manifest(manifest)

    def test_package_must_match_frozen_source_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            package, manifest_path, manifest = make_package(Path(directory))
            manifest["commissioning_preregistration_sha256"] = "e" * 64
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            with self.assertRaisesRegex(contract.ContractError, "package/source"):
                contract.validate_materialized_package(package, manifest_path)

    def test_hash_change_and_cross_condition_batch_change_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, manifest_path, manifest = make_package(root)
            (package / "config/runtime.json").write_text("changed")
            with self.assertRaisesRegex(contract.ContractError, "byte-size mismatch|hash mismatch"):
                contract.validate_materialized_package(package, manifest_path)

            package, manifest_path, manifest = make_package(root / "second")
            target = next(
                cell
                for cell in manifest["cells"]
                if cell["condition"] == "n16/B1" and cell["seed"] == 5101
            )
            target["runtime"]["batch_indices"] = "batch_indices:5102"
            with self.assertRaisesRegex(contract.ContractError, "reuse identical batch"):
                contract.validate_package_manifest(manifest)

    def test_labels_cannot_be_reused_as_runtime_input(self):
        with tempfile.TemporaryDirectory() as directory:
            package, _, manifest = make_package(Path(directory))
            _ = package
            manifest["cells"][0]["runtime"]["risk_features"] = "base:labels"
            with self.assertRaisesRegex(contract.ContractError, "leaks scoring"):
                contract.validate_package_manifest(manifest)

    def test_required_runtime_dependency_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, manifest = make_package(Path(directory))
            manifest["runtime_dependencies_required"] = True
            with self.assertRaisesRegex(contract.ContractError, "requires runtime dependencies"):
                contract.validate_package_manifest(manifest)
            manifest["artifacts"].append(
                {
                    "role": "runtime_dependencies",
                    "package_path": "config/PI4B8_RUNTIME_DEPENDENCIES_V1.json",
                    "bytes": 1,
                    "sha256": "0" * 64,
                }
            )
            contract.validate_package_manifest(manifest)


class EvidenceContractTests(unittest.TestCase):
    def test_worker_output_requires_600_by_16_and_ack_hash_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, manifest_path, _ = make_package(root)
            path = root / "windows.csv"
            fields = [
                "run_id",
                "window_id",
                "policy",
                "condition",
                "seed",
                "records_processed",
                "batch_indices",
                "state_id",
                "selected_action",
                "selected_action_index",
                "previous_action",
                "previous_action_index",
                "switched",
                "safe_set_mask",
                "safe_set_size",
                "safe_action_indices",
                "fallback",
                "q_values_for_state",
                "q_masked_candidate_indices",
                "q_masked_candidate_values",
                "q_masked_argmax_index",
                "q_masked_argmax_rule",
                "predicted_latency_by_action_ms",
                "latency_headroom_by_action_ms",
                "predicted_thermal_by_action_C",
                "thermal_headroom_by_action_C",
                *contract.CORRECTED_TIMING_FIELDS,
                "temperature_start_C",
                "risk",
                "energy_state",
                "backlog_records",
                "threshold_queue_load_records",
                "background_load_pid",
                "background_load_alive",
                "background_pinned_cpu",
                "background_cpu_affinity",
                "background_duty",
                "background_period_ms",
                "selected_decision_threshold",
                "prediction_scores",
                "prediction_bits",
                "labels_scoring_only",
                "tp",
                "fp",
                "fn",
                "tn",
                "reward",
                "reward_total",
                "reward_security",
                "reward_energy",
                "reward_backlog",
                "reward_switch",
                "throttled_start",
                "throttled_end",
                "cpu_frequency_start_mhz_min",
                "cpu_frequency_end_mhz_min",
            ]
            predictions = [0, 1] * 8
            labels = [0, 1] * 8
            schedule = np.load(package / "inputs/batches/seed_5104.npy")
            action_map = {action: 10.0 for action in contract.ACTIONS}
            latency_headroom = {action: 27.0 for action in contract.ACTIONS}
            thermal_map = {action: 40.0 for action in contract.ACTIONS}
            thermal_headroom = {action: 40.0 for action in contract.ACTIONS}
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for window in range(600):
                    row = {field: 0.0 for field in fields}
                    previous = 4 if window == 0 else 0
                    energy = 1.0 - window * 0.0001
                    state_components = (
                        2,
                        1,
                        3,
                        0,
                        previous,
                    )
                    state_id = 0
                    for value, cardinality in zip(
                        state_components, (4, 4, 4, 4, 5)
                    ):
                        state_id = state_id * cardinality + value
                    row.update(
                        {
                            "run_id": "run-1",
                            "window_id": window,
                            "policy": "Safe-Greedy",
                            "condition": "n16/B1",
                            "seed": 5104,
                            "records_processed": 16,
                            "batch_indices": json.dumps(schedule[window].tolist()),
                            "state_id": state_id,
                            "selected_action": contract.ACTIONS[0],
                            "selected_action_index": 0,
                            "previous_action": "" if window == 0 else contract.ACTIONS[0],
                            "previous_action_index": previous,
                            "switched": 0,
                            "safe_set_mask": "1111",
                            "safe_set_size": 4,
                            "safe_action_indices": json.dumps([0, 1, 2, 3]),
                            "fallback": 0,
                            "q_values_for_state": "",
                            "q_masked_candidate_indices": "",
                            "q_masked_candidate_values": "",
                            "q_masked_argmax_index": "",
                            "q_masked_argmax_rule": "",
                            "predicted_latency_by_action_ms": json.dumps(action_map),
                            "latency_headroom_by_action_ms": json.dumps(latency_headroom),
                            "predicted_thermal_by_action_C": json.dumps(thermal_map),
                            "thermal_headroom_by_action_C": json.dumps(thermal_headroom),
                            "temperature_start_C": 40.0,
                            "risk": 0.5,
                            "energy_state": energy,
                            "backlog_records": 0.0,
                            "threshold_queue_load_records": 16.0,
                            "background_load_pid": 123,
                            "background_load_alive": 1,
                            "background_pinned_cpu": 3,
                            "background_cpu_affinity": json.dumps([3]),
                            "background_duty": 0.5,
                            "background_period_ms": 100.0,
                            "selected_decision_threshold": 0.5,
                            "prediction_scores": json.dumps([0.2, 0.8] * 8),
                            "prediction_bits": json.dumps(predictions),
                            "labels_scoring_only": json.dumps(labels),
                            "tp": 8,
                            "fp": 0,
                            "fn": 0,
                            "tn": 8,
                            "reward": 0.27,
                            "reward_total": 0.27,
                            "reward_security": 0.3,
                            "reward_energy": -0.03,
                            "reward_backlog": 0.0,
                            "reward_switch": 0.0,
                            "throttled_start": "throttled=0x0",
                            "throttled_end": "throttled=0x0",
                            "cpu_frequency_start_mhz_min": 1800.0,
                            "cpu_frequency_end_mhz_min": 1800.0,
                        }
                    )
                    writer.writerow(row)
            package_hash = contract.sha256(manifest_path)
            start_ack = {
                "event": "RUN_START_ACK",
                "run_id": "run-1",
                "records_per_window": 16,
                "pi_monotonic": 1.0,
                "package_manifest_sha256": package_hash,
                "q_table_sha256": "",
            }
            end_ack = {
                "event": "RUN_END_ACK",
                "run_id": "run-1",
                "pi_monotonic": 2.0,
                "windows_completed": 600,
                "records_completed": 9600,
                "raw_artifact_sha256": contract.sha256(path),
                "package_manifest_sha256": package_hash,
            }
            result = contract.validate_worker_output(
                path,
                package_root=package,
                package_manifest_path=manifest_path,
                run_id="run-1",
                policy="Safe-Greedy",
                condition="n16/B1",
                seed=5104,
                start_ack=start_ack,
                end_ack=end_ack,
                expected_package_manifest_sha256=package_hash,
                expected_q_table_sha256="",
            )
            self.assertEqual(result["records"], 9600)
            self.assertEqual(result["first_window_temperature_start_C"], 40.0)

            original_text = path.read_text()
            original_rows = list(csv.DictReader(original_text.splitlines()))
            output_fields = list(original_rows[0])

            def write_rows(changed_rows):
                with path.open("w", newline="") as stream:
                    output = csv.DictWriter(stream, fieldnames=output_fields)
                    output.writeheader()
                    output.writerows(changed_rows)

            tampered = [dict(row) for row in original_rows]
            wrong_batch = json.loads(tampered[0]["batch_indices"])
            wrong_batch[0] += 1
            tampered[0]["batch_indices"] = json.dumps(wrong_batch)
            write_rows(tampered)
            with self.assertRaisesRegex(contract.ContractError, "hash-bound schedule"):
                contract.validate_worker_output(
                    path,
                    package_root=package,
                    package_manifest_path=manifest_path,
                    run_id="run-1",
                    policy="Safe-Greedy",
                    condition="n16/B1",
                    seed=5104,
                    start_ack=start_ack,
                    end_ack=end_ack,
                    expected_package_manifest_sha256=package_hash,
                    expected_q_table_sha256="",
                )

            tampered = [dict(row) for row in original_rows]
            tampered[0]["reward_total"] = "999"
            write_rows(tampered)
            with self.assertRaisesRegex(contract.ContractError, "reward_total"):
                contract.validate_worker_output(
                    path,
                    package_root=package,
                    package_manifest_path=manifest_path,
                    run_id="run-1",
                    policy="Safe-Greedy",
                    condition="n16/B1",
                    seed=5104,
                    start_ack=start_ack,
                    end_ack=end_ack,
                    expected_package_manifest_sha256=package_hash,
                    expected_q_table_sha256="",
                )

            tampered = [dict(row) for row in original_rows]
            tampered[0]["background_cpu_affinity"] = json.dumps([2])
            write_rows(tampered)
            with self.assertRaisesRegex(contract.ContractError, "background-load"):
                contract.validate_worker_output(
                    path,
                    package_root=package,
                    package_manifest_path=manifest_path,
                    run_id="run-1",
                    policy="Safe-Greedy",
                    condition="n16/B1",
                    seed=5104,
                    start_ack=start_ack,
                    end_ack=end_ack,
                    expected_package_manifest_sha256=package_hash,
                    expected_q_table_sha256="",
                )

            # The hash-bound Q table has its global maximum at unsafe action 1.
            # The logged masked choice over [0, 2, 3] must still reproduce action 0.
            fssql_rows = [dict(row) for row in original_rows]
            q_values = [9.0, 100.0, 8.0, 7.0]
            for row in fssql_rows:
                latency = json.loads(row["predicted_latency_by_action_ms"])
                headroom = json.loads(row["latency_headroom_by_action_ms"])
                latency[contract.ACTIONS[1]] = 100.0
                headroom[contract.ACTIONS[1]] = -63.0
                row.update(
                    {
                        "policy": "FSSQL-R",
                        "safe_set_mask": "1011",
                        "safe_set_size": 3,
                        "safe_action_indices": json.dumps([0, 2, 3]),
                        "predicted_latency_by_action_ms": json.dumps(latency),
                        "latency_headroom_by_action_ms": json.dumps(headroom),
                        "q_values_for_state": json.dumps(q_values),
                        "q_masked_candidate_indices": json.dumps([0, 2, 3]),
                        "q_masked_candidate_values": json.dumps([9.0, 8.0, 7.0]),
                        "q_masked_argmax_index": 0,
                        "q_masked_argmax_rule": "maximum_Q_then_lowest_action_index",
                    }
                )
            write_rows(fssql_rows)
            q_path = package / "q_tables/fssql_r_seed_5104.npz"
            start_ack["q_table_sha256"] = contract.sha256(q_path)
            end_ack["raw_artifact_sha256"] = contract.sha256(path)
            fssql_result = contract.validate_worker_output(
                path,
                package_root=package,
                package_manifest_path=manifest_path,
                run_id="run-1",
                policy="FSSQL-R",
                condition="n16/B1",
                seed=5104,
                start_ack=start_ack,
                end_ack=end_ack,
                expected_package_manifest_sha256=package_hash,
                expected_q_table_sha256=contract.sha256(q_path),
            )
            self.assertEqual(fssql_result["windows"], 600)

            fallback_rows = [dict(row) for row in fssql_rows]
            for row in fallback_rows:
                row.update(
                    {
                        "safe_set_mask": "0000",
                        "safe_set_size": 0,
                        "safe_action_indices": json.dumps([]),
                        "fallback": 1,
                        "predicted_latency_by_action_ms": json.dumps(
                            {action: 100.0 for action in contract.ACTIONS}
                        ),
                        "latency_headroom_by_action_ms": json.dumps(
                            {action: -63.0 for action in contract.ACTIONS}
                        ),
                        "q_masked_candidate_indices": json.dumps([0]),
                        "q_masked_candidate_values": json.dumps([9.0]),
                    }
                )
            write_rows(fallback_rows)
            end_ack["raw_artifact_sha256"] = contract.sha256(path)
            fallback_result = contract.validate_worker_output(
                path,
                package_root=package,
                package_manifest_path=manifest_path,
                run_id="run-1",
                policy="FSSQL-R",
                condition="n16/B1",
                seed=5104,
                start_ack=start_ack,
                end_ack=end_ack,
                expected_package_manifest_sha256=package_hash,
                expected_q_table_sha256=contract.sha256(q_path),
            )
            self.assertEqual(fallback_result["windows"], 600)

            static_rows = [dict(row) for row in original_rows]
            for row in static_rows:
                row["policy"] = "Static-Light"
            write_rows(static_rows)
            start_ack["q_table_sha256"] = ""
            end_ack["raw_artifact_sha256"] = contract.sha256(path)
            static_result = contract.validate_worker_output(
                path,
                package_root=package,
                package_manifest_path=manifest_path,
                run_id="run-1",
                policy="Static-Light",
                condition="n16/B1",
                seed=5104,
                start_ack=start_ack,
                end_ack=end_ack,
                expected_package_manifest_sha256=package_hash,
                expected_q_table_sha256="",
            )
            self.assertEqual(static_result["windows"], 600)
            static_rows[0]["selected_action_index"] = 1
            static_rows[0]["selected_action"] = contract.ACTIONS[1]
            write_rows(static_rows)
            with self.assertRaisesRegex(contract.ContractError, "Static-Light"):
                contract.validate_worker_output(
                    path,
                    package_root=package,
                    package_manifest_path=manifest_path,
                    run_id="run-1",
                    policy="Static-Light",
                    condition="n16/B1",
                    seed=5104,
                    start_ack=start_ack,
                    end_ack=end_ack,
                    expected_package_manifest_sha256=package_hash,
                    expected_q_table_sha256="",
                )

            threshold_rows = [dict(row) for row in original_rows]
            for row in threshold_rows:
                row["policy"] = "Threshold"
            write_rows(threshold_rows)
            end_ack["raw_artifact_sha256"] = contract.sha256(path)
            threshold_result = contract.validate_worker_output(
                path,
                package_root=package,
                package_manifest_path=manifest_path,
                run_id="run-1",
                policy="Threshold",
                condition="n16/B1",
                seed=5104,
                start_ack=start_ack,
                end_ack=end_ack,
                expected_package_manifest_sha256=package_hash,
                expected_q_table_sha256="",
            )
            self.assertEqual(threshold_result["windows"], 600)
            threshold_rows[0]["selected_action_index"] = 1
            threshold_rows[0]["selected_action"] = contract.ACTIONS[1]
            write_rows(threshold_rows)
            with self.assertRaisesRegex(contract.ContractError, "Threshold action"):
                contract.validate_worker_output(
                    path,
                    package_root=package,
                    package_manifest_path=manifest_path,
                    run_id="run-1",
                    policy="Threshold",
                    condition="n16/B1",
                    seed=5104,
                    start_ack=start_ack,
                    end_ack=end_ack,
                    expected_package_manifest_sha256=package_hash,
                    expected_q_table_sha256="",
                )

            path.write_text(original_text)
            start_ack["q_table_sha256"] = ""
            end_ack["raw_artifact_sha256"] = contract.sha256(path)
            start_ack["package_manifest_sha256"] = "b" * 64
            with self.assertRaisesRegex(contract.ContractError, "ACK/output binding"):
                contract.validate_worker_output(
                    path,
                    package_root=package,
                    package_manifest_path=manifest_path,
                    run_id="run-1",
                    policy="Safe-Greedy",
                    condition="n16/B1",
                    seed=5104,
                    start_ack=start_ack,
                    end_ack=end_ack,
                    expected_package_manifest_sha256=package_hash,
                    expected_q_table_sha256="",
                )

    def test_power_integration_uses_ack_boundaries_without_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            meter = Path(directory) / "meter.csv"
            with meter.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("sample_monotonic", "power_W"))
                writer.writeheader()
                for second in range(21):
                    writer.writerow({"sample_monotonic": second, "power_W": 2.0})
            handshake = {
                "run_id": "run-1",
                "meter_capture_start_mac_monotonic": -1.0,
                "start_request_mac_monotonic": 4.0,
                "start_ack_receive_mac_monotonic": 5.0,
                "end_ack_receive_mac_monotonic": 15.0,
                "meter_capture_end_mac_monotonic": 21.0,
                "pi_start_ack": {
                    "event": "RUN_START_ACK",
                    "run_id": "run-1",
                    "pi_monotonic": 100.0,
                },
                "pi_end_ack": {
                    "event": "RUN_END_ACK",
                    "run_id": "run-1",
                    "pi_monotonic": 110.0,
                },
            }
            result = contract.validate_power_boundary(meter, handshake, run_id="run-1")
            self.assertEqual(result["energy_J"], 20.0)
            self.assertTrue(result["boundary_includes_post_decision_scoring_and_evidence_write"])
            handshake["end_ack_receive_mac_monotonic"] = 20.5
            with self.assertRaisesRegex(contract.ContractError, "outside POWER-Z"):
                contract.validate_power_boundary(meter, handshake, run_id="run-1")


if __name__ == "__main__":
    unittest.main()
