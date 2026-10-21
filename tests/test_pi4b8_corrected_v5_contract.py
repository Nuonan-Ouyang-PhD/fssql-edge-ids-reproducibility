import csv
import json
import tempfile
import unittest
from pathlib import Path

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
            path = Path(directory) / "windows.csv"
            fields = [
                "run_id",
                "window_id",
                "policy",
                "condition",
                "seed",
                "records_processed",
                *contract.CORRECTED_TIMING_FIELDS,
                "prediction_scores",
                "prediction_bits",
                "labels_scoring_only",
                "tp",
                "fp",
                "fn",
                "tn",
                "throttled_start",
                "throttled_end",
                "cpu_frequency_start_mhz_min",
                "cpu_frequency_end_mhz_min",
            ]
            predictions = [0, 1] * 8
            labels = [0, 1] * 8
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for window in range(600):
                    row = {field: 0.0 for field in fields}
                    row.update(
                        {
                            "run_id": "run-1",
                            "window_id": window,
                            "policy": "Safe-Greedy",
                            "condition": "n16/B1",
                            "seed": 5104,
                            "records_processed": 16,
                            "prediction_scores": json.dumps([0.2, 0.8] * 8),
                            "prediction_bits": json.dumps(predictions),
                            "labels_scoring_only": json.dumps(labels),
                            "tp": 8,
                            "fp": 0,
                            "fn": 0,
                            "tn": 8,
                            "throttled_start": "throttled=0x0",
                            "throttled_end": "throttled=0x0",
                            "cpu_frequency_start_mhz_min": 1800.0,
                            "cpu_frequency_end_mhz_min": 1800.0,
                        }
                    )
                    writer.writerow(row)
            package_hash = "a" * 64
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
            start_ack["package_manifest_sha256"] = "b" * 64
            with self.assertRaisesRegex(contract.ContractError, "ACK/output binding"):
                contract.validate_worker_output(
                    path,
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
