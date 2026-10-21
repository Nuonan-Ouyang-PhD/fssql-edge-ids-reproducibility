import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from online.pi4b8 import pi4b8_corrected_v5_execution_orchestrator as execution


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sixty_run_plan(package_hash: str) -> dict:
    runs = []
    index = 0
    for seed in range(5101, 5106):
        for condition in ("n16/B0", "n16/B1", "n16/B2"):
            for policy in ("FSSQL-R", "Safe-Greedy", "Threshold", "Static-Light"):
                index += 1
                runs.append(
                    {
                        "execution_index": index,
                        "run_id": f"run-{index}",
                        "policy": policy,
                        "condition": condition,
                        "seed": seed,
                    }
                )
    return {
        "status": "CORRECTED_V5_PLAN_ONLY_NOT_EXECUTION_AUTHORIZATION",
        "package_manifest_sha256": package_hash,
        "remote_package_root": "/remote/package-v5",
        "remote_output_root": "/remote/runs-v5",
        "runs": runs,
    }


class CorrectedExecutionAuthorizationTests(unittest.TestCase):
    def _fixture(self, root: Path):
        package_root = root / "package"
        authorization = root / "authorization"
        package_root.mkdir()
        authorization.mkdir()
        manifest_path = package_root / "PI4B8_CORRECTED_RUNTIME_PACKAGE_MANIFEST_V5.json"
        manifest_path.write_text("{}\n", encoding="utf-8")
        for relative in (
            "pi4b8_online_worker.py",
            "powerz/powerz_hid_capture.py",
            "powerz/POWER_Z_HID_ACQUISITION_V1.md",
            execution.RUNTIME_DEPENDENCY_PATH,
        ):
            path = package_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative, encoding="utf-8")
        dependency_path = package_root / execution.RUNTIME_DEPENDENCY_PATH
        dependency_path.write_text(
            json.dumps(
                {
                    "status": "FROZEN_BEFORE_CORRECTED_PI4B8_COMMISSIONING",
                    "official_test_access": "forbidden",
                    "pi_worker": {
                        "python_implementation": "cpython",
                        "python_version": "3.13.5",
                        "numpy_version": "2.2.4",
                        "system": "Linux",
                        "machine": "aarch64",
                        "cpu_governor_required": "performance",
                        "minimum_frequency_mhz": 1350.0,
                    },
                    "mac_orchestrator_and_powerz": {
                        "python_implementation": "cpython",
                        "python_version": "3.9.6",
                        "numpy_version": "2.0.2",
                        "hidapi_version": "0.15.0",
                        "system": "Darwin",
                        "machine": "arm64",
                        "python_executable": ".venv/bin/python",
                    },
                }
            ),
            encoding="utf-8",
        )
        package = {
            "source_manifest_sha256": "a" * 64,
            "artifacts": [
                {
                    "role": "worker",
                    "package_path": "pi4b8_online_worker.py",
                    "sha256": digest(package_root / "pi4b8_online_worker.py"),
                },
                {
                    "role": "powerz:capture",
                    "package_path": "powerz/powerz_hid_capture.py",
                    "sha256": digest(package_root / "powerz/powerz_hid_capture.py"),
                },
                {
                    "role": "powerz:specification",
                    "package_path": "powerz/POWER_Z_HID_ACQUISITION_V1.md",
                    "sha256": digest(
                        package_root / "powerz/POWER_Z_HID_ACQUISITION_V1.md"
                    ),
                },
                {
                    "role": execution.RUNTIME_DEPENDENCY_ROLE,
                    "package_path": execution.RUNTIME_DEPENDENCY_PATH,
                    "sha256": digest(dependency_path),
                },
                {
                    "role": "scheduler_batch_indices_commissioning_seed_5099",
                    "package_path": "inputs/batch_indices_seed_5099.npy",
                    "sha256": "4" * 64,
                },
            ],
            "commissioning_cell": {
                "policy": "Static-Light",
                "condition": "n16/B0",
                "seed": 5099,
                "runtime": {
                    "batch_indices": (
                        "scheduler_batch_indices_commissioning_seed_5099"
                    )
                },
            },
        }
        validation_path = authorization / execution.COMMISSIONING_VALIDATION
        validation = {
            "status": "PASS",
            "marker": execution.COMMISSIONING_MARKER,
            "run_id": "commissioning-v5",
            "policy": "Static-Light",
            "condition": "n16/B0",
            "seed": 5099,
            "checks": {
                "windows_600": True,
                "records_9600": True,
                "worker_output_valid": True,
                "scoring_reproducible": True,
                "label_isolation": True,
                "powerz_valid": True,
                "meter_monotonic": True,
                "energy_reproducible": True,
                "run_id_binding": True,
                "hash_binding": True,
                "throttle_zero": True,
                "frequency_invariant": True,
                "governor_performance": True,
                "runtime_versions_exact": True,
                "pre_roll_s": 5.1,
                "post_roll_s": 5.2,
            },
        }
        decision_path = authorization / execution.COMMISSIONING_DECISION
        decision_path.write_text(
            "CORRECTED V5 COMMISSIONING PASS commissioning-v5\n"
            + execution.COMMISSIONING_MARKER,
            encoding="utf-8",
        )
        plan_path = authorization / execution.FORMAL_PLAN
        plan = sixty_run_plan(digest(manifest_path))
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        preflight_path = authorization / execution.RUNTIME_PREFLIGHT
        preflight = {
            "status": "PASS",
            "dependency_manifest_sha256": digest(dependency_path),
            "remote_package": {
                "status": "PASS",
                "remote_package_root": "/remote/package-v5",
                "package_manifest_sha256": digest(manifest_path),
                "artifacts": 38,
                "files": 40,
            },
            "pi_worker": {
                "python_implementation": "cpython",
                "python_version": "3.13.5",
                "numpy_version": "2.2.4",
                "system": "Linux",
                "machine": "aarch64",
                "governor_paths": [
                    f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor"
                    for i in range(4)
                ],
                "governors": ["performance"] * 4,
            },
            "mac_orchestrator_and_powerz": {
                "python_implementation": "cpython",
                "python_version": "3.9.6",
                "numpy_version": "2.0.2",
                "hidapi_version": "0.15.0",
                "system": "Darwin",
                "machine": "arm64",
            },
        }
        preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
        validation.update(
            {
                "package_manifest_sha256": digest(manifest_path),
                "source_manifest_sha256": "a" * 64,
                "worker_sha256": package["artifacts"][0]["sha256"],
                "orchestrator_sha256": digest(Path(execution.__file__)),
                "runtime_preflight_sha256": digest(preflight_path),
                "runtime_dependencies_sha256": digest(dependency_path),
                "commissioning_batch_indices_sha256": "4" * 64,
            }
        )
        validation_path.write_text(json.dumps(validation), encoding="utf-8")
        final_path = authorization / execution.FINAL_MANIFEST
        final = {
            "status": "PI4B8_CORRECTED_V5_FINAL_EXECUTION_MANIFEST",
            "execution_authorized": True,
            "official_test_access": "forbidden",
            "commissioning_run_id": "commissioning-v5",
            "commissioning_validation_sha256": digest(validation_path),
            "commissioning_decision_sha256": digest(decision_path),
            "formal_plan_sha256": digest(plan_path),
            "orchestrator_sha256": digest(Path(execution.__file__)),
            "runtime_package_manifest_sha256": digest(manifest_path),
            "source_manifest_sha256": "a" * 64,
            "worker_sha256": package["artifacts"][0]["sha256"],
            "powerz_capture_sha256": package["artifacts"][1]["sha256"],
            "powerz_specification_sha256": package["artifacts"][2]["sha256"],
            "runtime_preflight_sha256": digest(preflight_path),
            "runtime_dependencies_sha256": digest(dependency_path),
            "commissioning_batch_indices_sha256": "4" * 64,
        }
        final_path.write_text(json.dumps(final), encoding="utf-8")
        return package_root, manifest_path, authorization, package, plan

    def test_formal_gate_requires_exact_corrected_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package_root, manifest_path, authorization, package, plan = self._fixture(root)
            with mock.patch.object(
                execution, "validate_materialized_package", return_value=package
            ), mock.patch.object(execution, "build_plan", return_value=plan):
                receipt = execution.validate_formal_authorization(
                    package_root, manifest_path, authorization
                )
                self.assertEqual(
                    receipt["status"], "PI4B8_CORRECTED_V5_FINAL_FORMAL_GATE_PASS"
                )
                self.assertEqual(receipt["runs"], 60)

                decision = authorization / execution.COMMISSIONING_DECISION
                decision.write_text("modified", encoding="utf-8")
                with self.assertRaisesRegex(execution.ExecutionError, "decision"):
                    execution.validate_formal_authorization(
                        package_root, manifest_path, authorization
                    )

    def test_formal_gate_fails_closed_without_final_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package_root, manifest_path, authorization, package, plan = self._fixture(root)
            (authorization / execution.FINAL_MANIFEST).unlink()
            with mock.patch.object(
                execution, "validate_materialized_package", return_value=package
            ), mock.patch.object(execution, "build_plan", return_value=plan):
                with self.assertRaisesRegex(execution.ExecutionError, "missing required"):
                    execution.validate_formal_authorization(
                        package_root, manifest_path, authorization
                    )

    def test_plan_must_equal_recomputed_package_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package_root, manifest_path, authorization, _, plan = self._fixture(root)
            plan["runs"][0]["seed"] = 9999
            (authorization / execution.FORMAL_PLAN).write_text(
                json.dumps(plan), encoding="utf-8"
            )
            expected = sixty_run_plan(digest(manifest_path))
            with mock.patch.object(execution, "build_plan", return_value=expected):
                with self.assertRaisesRegex(execution.ExecutionError, "differs"):
                    execution.validate_frozen_plan(
                        package_root,
                        manifest_path,
                        authorization / execution.FORMAL_PLAN,
                    )


class CommissioningArtifactTests(unittest.TestCase):
    def test_remote_package_preflight_requires_local_manifest_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            package_root = Path(directory)
            manifest = package_root / "PI4B8_CORRECTED_RUNTIME_PACKAGE_MANIFEST_V5.json"
            manifest.write_text("{}\n", encoding="utf-8")
            observed = {
                "status": "PASS",
                "remote_package_root": "/remote/package-v5",
                "package_manifest_sha256": digest(manifest),
                "artifacts": 38,
                "files": 40,
            }
            completed = mock.Mock(
                returncode=0, stdout=json.dumps(observed), stderr=""
            )
            with mock.patch.object(execution.subprocess, "run", return_value=completed):
                receipt = execution.validate_remote_package_preflight(
                    package_root,
                    remote_package_root=Path("/remote/package-v5"),
                    ssh_target="pi@host",
                )
            self.assertEqual(receipt["status"], "PASS")

            observed["package_manifest_sha256"] = "0" * 64
            completed.stdout = json.dumps(observed)
            with mock.patch.object(execution.subprocess, "run", return_value=completed):
                with self.assertRaisesRegex(execution.ExecutionError, "differs"):
                    execution.validate_remote_package_preflight(
                        package_root,
                        remote_package_root=Path("/remote/package-v5"),
                        ssh_target="pi@host",
                    )

    def test_commissioning_rejects_formal_seed(self):
        with self.assertRaisesRegex(execution.ExecutionError, "non-formal cell"):
            execution.validate_commissioning_identity(
                "Static-Light", "n16/B0", 5101
            )
        execution.validate_commissioning_identity(
            "Static-Light", "n16/B0", 5099
        )

    def test_commissioning_output_is_explicitly_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            authorization = Path(directory)
            result = {
                "run_id": "commissioning-v5",
                "policy": "Static-Light",
                "condition": "n16/B0",
                "seed": 5099,
                "pi": {"windows": 600, "records": 9600, "sha256": "b" * 64},
                "powerz": {
                    "sha256": "c" * 64,
                    "pre_roll_s": 5.1,
                    "post_roll_s": 5.2,
                },
                "run_manifest_sha256": "d" * 64,
            }
            preflight_path = authorization / execution.RUNTIME_PREFLIGHT
            preflight = {
                "status": "PASS",
                "dependency_manifest_sha256": "3" * 64,
                "pi_worker": {"governors": ["performance"] * 4},
            }
            preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
            validation_path, decision_path = execution.write_commissioning_artifacts(
                authorization,
                result,
                package_manifest_sha256="e" * 64,
                source_manifest_sha256="f" * 64,
                worker_sha256="1" * 64,
                orchestrator_sha256="2" * 64,
                runtime_preflight=preflight,
                commissioning_batch_indices_sha256="4" * 64,
            )
            validation = json.loads(validation_path.read_text())
            self.assertEqual(validation["status"], "PASS")
            self.assertEqual(validation["marker"], execution.COMMISSIONING_MARKER)
            self.assertEqual(
                validation["commissioning_batch_indices_sha256"], "4" * 64
            )
            self.assertIn(execution.COMMISSIONING_MARKER, decision_path.read_text())

    def test_remote_preflight_rejects_governor_and_numpy_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            package_root = Path(directory)
            dependency_path = package_root / execution.RUNTIME_DEPENDENCY_PATH
            dependency_path.parent.mkdir(parents=True)
            dependency_path.write_text(
                json.dumps(
                    {
                        "status": "FROZEN_BEFORE_CORRECTED_PI4B8_COMMISSIONING",
                        "official_test_access": "forbidden",
                        "pi_worker": {
                            "python_implementation": "cpython",
                            "python_version": "3.13.5",
                            "numpy_version": "2.2.4",
                            "system": "Linux",
                            "machine": "aarch64",
                            "cpu_governor_required": "performance",
                            "minimum_frequency_mhz": 1350.0,
                        },
                        "mac_orchestrator_and_powerz": {
                            "python_implementation": "cpython",
                            "python_version": "3.9.6",
                            "numpy_version": "2.0.2",
                            "hidapi_version": "0.15.0",
                            "system": "Darwin",
                            "machine": "arm64",
                            "python_executable": ".venv/bin/python",
                        },
                    }
                ),
                encoding="utf-8",
            )
            package = {
                "artifacts": [
                    {
                        "role": execution.RUNTIME_DEPENDENCY_ROLE,
                        "package_path": execution.RUNTIME_DEPENDENCY_PATH,
                        "sha256": digest(dependency_path),
                    }
                ]
            }
            observed = {
                "python_implementation": "cpython",
                "python_version": "3.13.5",
                "numpy_version": "2.2.4",
                "system": "Linux",
                "machine": "aarch64",
                "governor_paths": [
                    f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor"
                    for i in range(4)
                ],
                "governors": ["ondemand"] * 4,
            }
            completed = mock.Mock(returncode=0, stdout=json.dumps(observed), stderr="")
            with mock.patch.object(execution.subprocess, "run", return_value=completed):
                with self.assertRaisesRegex(execution.ExecutionError, "governor"):
                    execution.validate_remote_runtime_preflight(
                        package_root, package, ssh_target="pi@host"
                    )

            observed["governors"] = ["performance"] * 4
            observed["numpy_version"] = "2.0.2"
            completed.stdout = json.dumps(observed)
            with mock.patch.object(execution.subprocess, "run", return_value=completed):
                with self.assertRaisesRegex(execution.ExecutionError, "numpy_version"):
                    execution.validate_remote_runtime_preflight(
                        package_root, package, ssh_target="pi@host"
                    )

            observed["numpy_version"] = "2.2.4"
            completed.stdout = json.dumps(observed)
            with mock.patch.object(execution.subprocess, "run", return_value=completed):
                receipt = execution.validate_remote_runtime_preflight(
                    package_root, package, ssh_target="pi@host"
                )
            self.assertEqual(receipt["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
