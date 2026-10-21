import hashlib
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from online.pi4b8 import pi4b8_corrected_v5_execution_orchestrator as execution


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependency_document() -> dict:
    return {
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
        "device_identity": {
            "hostname": "pi4b8g",
            "model": "Raspberry Pi 4 Model B Rev 1.5",
            "cpu_serial": "100000005368e39d",
            "mem_total_kib": 8007464,
        },
        "cooldown": {
            "idle_samples": 30,
            "idle_interval_s": 1.0,
            "maximum_start_above_idle_median_c": 2.0,
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
        "powerz_device": {
            "vendor_id_hex": "0x5fc9",
            "product_id_hex": "0x0063",
            "serial": "075356",
        },
    }


def cooldown_payload(*, final_temperature: float = 40.0) -> tuple[dict, list]:
    dependency = dependency_document()
    identity = {**dependency["device_identity"], "boot_id": "boot-1"}
    samples = []
    for index in range(30):
        samples.append(
            {
                "sample_index": index,
                "pi_utc": f"2026-01-01T00:00:{index:02d}+00:00",
                "pi_monotonic": 100.0 + index,
                "temperature_c": final_temperature if index == 29 else 40.0,
                "frequency_mhz": [1800.0] * 4,
                "governors": ["performance"] * 4,
                "throttled": "throttled=0x0",
                "throttle_returncode": 0,
            }
        )
    return identity, samples


def cooldown_receipt(*, final_temperature: float = 40.0) -> dict:
    identity, samples = cooldown_payload(final_temperature=final_temperature)
    return execution.validate_cooldown_payload(
        identity, samples, dependency_document()
    )


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
            "powerz/powerz_evidence_validator.py",
            "powerz/POWER_Z_HID_ACQUISITION_V1.md",
            "runtime/corrected_v5_contract.py",
            execution.RUNTIME_DEPENDENCY_PATH,
        ):
            path = package_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative, encoding="utf-8")
        dependency_path = package_root / execution.RUNTIME_DEPENDENCY_PATH
        dependency_path.write_text(
            json.dumps(dependency_document()),
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
                    "role": "powerz:validator",
                    "package_path": "powerz/powerz_evidence_validator.py",
                    "sha256": digest(
                        package_root / "powerz/powerz_evidence_validator.py"
                    ),
                },
                {
                    "role": "runtime_contract",
                    "package_path": "runtime/corrected_v5_contract.py",
                    "sha256": digest(
                        package_root / "runtime/corrected_v5_contract.py"
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
        commissioning_evidence = root / "evidence" / "commissioning-v5"
        commissioning_evidence.mkdir(parents=True)
        commissioning_files = {
            "pi_raw_path": commissioning_evidence / "pi_windows.csv",
            "powerz_raw_path": commissioning_evidence / "POWERZ_raw.csv",
            "handshake_path": commissioning_evidence / "HANDSHAKE.json",
            "cooldown_gate_path": commissioning_evidence / execution.COOLDOWN_EVIDENCE,
            "run_manifest_path": commissioning_evidence / "RUN_MANIFEST.json",
        }
        for field, path in commissioning_files.items():
            path.write_text(field, encoding="utf-8")
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
                "cooldown_gate": True,
                "device_identity_bound": True,
                "actual_start_temperature_gate": True,
                "pre_roll_s": 5.1,
                "post_roll_s": 5.2,
            },
            "evidence_directory": str(commissioning_evidence.resolve()),
            **{field: str(path.resolve()) for field, path in commissioning_files.items()},
            "pi_raw_sha256": digest(commissioning_files["pi_raw_path"]),
            "powerz_raw_sha256": digest(commissioning_files["powerz_raw_path"]),
            "handshake_sha256": digest(commissioning_files["handshake_path"]),
            "cooldown_gate_sha256": digest(
                commissioning_files["cooldown_gate_path"]
            ),
            "run_manifest_sha256": digest(
                commissioning_files["run_manifest_path"]
            ),
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
            "local_validation_code": {
                "runtime_contract": package["artifacts"][4]["sha256"],
                "powerz:validator": package["artifacts"][3]["sha256"],
            },
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
                "device_identity": dependency_document()["device_identity"],
                "boot_id": "boot-1",
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
                "powerz_validator_sha256": package["artifacts"][3]["sha256"],
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
            "powerz_validator_sha256": package["artifacts"][3]["sha256"],
            "commissioning_batch_indices_sha256": "4" * 64,
            "commissioning_pi_raw_sha256": validation["pi_raw_sha256"],
            "commissioning_powerz_raw_sha256": validation["powerz_raw_sha256"],
            "commissioning_handshake_sha256": validation["handshake_sha256"],
            "commissioning_cooldown_gate_sha256": validation[
                "cooldown_gate_sha256"
            ],
            "commissioning_run_manifest_sha256": validation[
                "run_manifest_sha256"
            ],
        }
        final_path.write_text(json.dumps(final), encoding="utf-8")
        return package_root, manifest_path, authorization, package, plan

    def test_formal_gate_requires_exact_corrected_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package_root, manifest_path, authorization, package, plan = self._fixture(root)
            validation = json.loads(
                (authorization / execution.COMMISSIONING_VALIDATION).read_text()
            )
            reopened = {
                field: validation[field]
                for field in (
                    "pi_raw_sha256",
                    "powerz_raw_sha256",
                    "handshake_sha256",
                    "cooldown_gate_sha256",
                    "run_manifest_sha256",
                )
            }
            with mock.patch.object(
                execution, "validate_materialized_package", return_value=package
            ), mock.patch.object(
                execution, "build_plan", return_value=plan
            ), mock.patch.object(
                execution, "revalidate_complete_cell", return_value=reopened
            ) as revalidate:
                receipt = execution.validate_formal_authorization(
                    package_root, manifest_path, authorization
                )
                self.assertEqual(
                    receipt["status"], "PI4B8_CORRECTED_V5_FINAL_FORMAL_GATE_PASS"
                )
                self.assertEqual(receipt["runs"], 60)
                revalidate.assert_called_once()

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

    def test_final_manifest_is_created_from_commissioning_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package_root, manifest_path, authorization, package, plan = self._fixture(root)
            (authorization / execution.FINAL_MANIFEST).unlink()
            with mock.patch.object(execution, "build_plan", return_value=plan):
                path = execution.write_final_execution_manifest(
                    package_root, manifest_path, authorization, package
                )
            final = json.loads(path.read_text())
            validation = json.loads(
                (authorization / execution.COMMISSIONING_VALIDATION).read_text()
            )
            self.assertTrue(final["execution_authorized"])
            self.assertEqual(
                final["commissioning_powerz_raw_sha256"],
                validation["powerz_raw_sha256"],
            )
            self.assertEqual(
                final["powerz_validator_sha256"],
                package["artifacts"][3]["sha256"],
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
            evidence = authorization / "evidence"
            evidence.mkdir()
            (evidence / "pi_windows.csv").write_text("pi", encoding="utf-8")
            (evidence / "POWERZ_raw.csv").write_text("meter", encoding="utf-8")
            (evidence / "HANDSHAKE.json").write_text("{}", encoding="utf-8")
            (evidence / "RUN_MANIFEST.json").write_text("{}", encoding="utf-8")
            cooldown = cooldown_receipt()
            (evidence / execution.COOLDOWN_EVIDENCE).write_text(
                json.dumps(cooldown), encoding="utf-8"
            )
            result = {
                "run_id": "commissioning-v5",
                "policy": "Static-Light",
                "condition": "n16/B0",
                "seed": 5099,
                "pi": {
                    "windows": 600,
                    "records": 9600,
                    "sha256": "b" * 64,
                    "first_window_temperature_start_C": 40.0,
                },
                "powerz": {
                    "sha256": "c" * 64,
                    "pre_roll_s": 5.1,
                    "post_roll_s": 5.2,
                },
                "run_manifest_sha256": digest(evidence / "RUN_MANIFEST.json"),
                "evidence_directory": str(evidence),
            }
            preflight_path = authorization / execution.RUNTIME_PREFLIGHT
            preflight = {
                "status": "PASS",
                "dependency_manifest_sha256": "3" * 64,
                "pi_worker": {
                    "governors": ["performance"] * 4,
                    "device_identity": dependency_document()["device_identity"],
                    "boot_id": "boot-1",
                },
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
                powerz_validator_sha256="5" * 64,
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
                json.dumps(dependency_document()),
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
                **dependency_document()["device_identity"],
                "boot_id": "boot-1",
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

            observed["hostname"] = "wrong-device"
            completed.stdout = json.dumps(observed)
            with mock.patch.object(execution.subprocess, "run", return_value=completed):
                with self.assertRaisesRegex(execution.ExecutionError, "identity"):
                    execution.validate_remote_runtime_preflight(
                        package_root, package, ssh_target="pi@host"
                    )


class EvidenceHardeningTests(unittest.TestCase):
    def test_local_validation_code_must_match_packaged_hashes(self):
        package = {
            "artifacts": [
                {
                    "role": "runtime_contract",
                    "sha256": digest(
                        execution.ROOT / "online/pi4b8/corrected_v5_contract.py"
                    ),
                },
                {
                    "role": "powerz:validator",
                    "sha256": digest(
                        execution.ROOT
                        / "online/pi4b8/powerz_evidence_validator.py"
                    ),
                },
            ]
        }
        observed = execution.validate_local_validation_code_binding(package)
        self.assertEqual(observed["runtime_contract"], package["artifacts"][0]["sha256"])
        package["artifacts"][1]["sha256"] = "0" * 64
        with self.assertRaisesRegex(execution.ExecutionError, "differs"):
            execution.validate_local_validation_code_binding(package)

    def test_ack_wait_has_finite_timeout(self):
        read_fd, write_fd = os.pipe()
        try:
            with os.fdopen(read_fd, "r", encoding="utf-8") as stream:
                with self.assertRaisesRegex(execution.ExecutionError, "timeout"):
                    execution.read_ack(
                        stream,
                        "RUN_START_ACK",
                        "run-1",
                        timeout_s=0.01,
                    )
        finally:
            os.close(write_fd)

    def test_powerz_ready_is_identity_and_path_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            meter_path = Path(directory) / "meter.csv"
            read_fd, write_fd = os.pipe()
            os.write(
                write_fd,
                (
                    json.dumps(
                        {
                            "event": "POWERZ_READY",
                            "output": str(meter_path),
                            "serial": "wrong",
                        }
                    )
                    + "\n"
                ).encode(),
            )
            os.close(write_fd)
            with os.fdopen(read_fd, "r", encoding="utf-8") as stream:
                with self.assertRaisesRegex(execution.ExecutionError, "serial"):
                    execution.read_powerz_ready(
                        stream,
                        meter_path=meter_path,
                        serial="075356",
                        timeout_s=0.1,
                    )

    def test_cooldown_rejects_hot_start_and_throttle(self):
        identity, hot_samples = cooldown_payload(final_temperature=43.0)
        with self.assertRaisesRegex(execution.ExecutionError, "temperature"):
            execution.validate_cooldown_payload(
                identity, hot_samples, dependency_document()
            )

        identity, samples = cooldown_payload()
        samples[10]["throttled"] = "throttled=0x1"
        with self.assertRaisesRegex(execution.ExecutionError, "throttle"):
            execution.validate_cooldown_payload(
                identity, samples, dependency_document()
            )

        receipt = cooldown_receipt()
        self.assertEqual(
            execution.validate_actual_run_start_temperature(
                {"first_window_temperature_start_C": 42.0},
                receipt,
                dependency_document(),
            ),
            42.0,
        )
        with self.assertRaisesRegex(execution.ExecutionError, "first-window"):
            execution.validate_actual_run_start_temperature(
                {"first_window_temperature_start_C": 42.1},
                receipt,
                dependency_document(),
            )

    def test_failed_attempt_writes_hashed_invalid_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package_root = root / "package"
            package_root.mkdir()
            manifest_path = package_root / execution.PACKAGE_MANIFEST_NAME
            manifest_path.write_text("{}\n", encoding="utf-8")
            evidence_root = root / "evidence"
            run = {
                "run_id": "failed-run",
                "policy": "Static-Light",
                "condition": "n16/B0",
                "seed": 5099,
                "remote_output": "/remote/failed-run",
                "worker_argv": ["python3", "worker.py"],
                "q_table_sha256": "",
            }
            package = {
                "artifacts": [
                    {
                        "role": "powerz:capture",
                        "package_path": "powerz/capture.py",
                    }
                ]
            }
            remote_listing = mock.Mock(
                returncode=0,
                stdout=json.dumps(
                    {"root": "/remote/failed-run", "files": []}
                ),
                stderr="",
            )
            with mock.patch.object(
                execution, "validate_materialized_package", return_value=package
            ), mock.patch.object(
                execution,
                "runtime_dependency_contract",
                return_value=(Path("dependency.json"), dependency_document(), "a" * 64),
            ), mock.patch.object(
                execution,
                "collect_idle_cooldown_gate",
                side_effect=execution.ExecutionError("cooldown failed"),
            ), mock.patch.object(
                execution.subprocess, "run", return_value=remote_listing
            ):
                with self.assertRaisesRegex(execution.ExecutionError, "cooldown failed"):
                    execution.execute_cell(
                        package_root,
                        manifest_path,
                        run,
                        evidence_root=evidence_root,
                        ssh_target="pi@host",
                        mac_python="python3",
                    )
            local_run = evidence_root / "failed-run"
            manifest = json.loads((local_run / "RUN_MANIFEST.json").read_text())
            self.assertEqual(manifest["validity_status"], "INVALID_TECHNICAL")
            self.assertEqual(manifest["reason"], "cooldown failed")
            self.assertTrue(manifest["artifacts"])
            for record in manifest["artifacts"]:
                self.assertEqual(digest(local_run / record["path"]), record["sha256"])

    def test_existing_incomplete_cell_is_never_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_root = root / "evidence"
            evidence_root.mkdir()
            (evidence_root / "run-1").mkdir()
            (evidence_root / execution.FORMAL_REGISTRY).write_text(
                json.dumps(
                    {
                        "utc": "2026-01-01T00:00:00+00:00",
                        "event": "EXECUTION_STARTED",
                        "execution_index": 1,
                        "run_id": "run-1",
                        "policy": "Static-Light",
                        "condition": "n16/B0",
                        "seed": 5101,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            package_root = root / "package"
            package_root.mkdir()
            manifest_path = package_root / execution.PACKAGE_MANIFEST_NAME
            manifest_path.write_text("{}\n", encoding="utf-8")
            plan = {
                "remote_package_root": "/remote/package",
                "remote_output_root": "/remote/output",
                "runs": [
                    {
                        "execution_index": 1,
                        "run_id": "run-1",
                        "policy": "Static-Light",
                        "condition": "n16/B0",
                        "seed": 5101,
                    }
                ],
            }
            receipt = {
                "status": "PI4B8_CORRECTED_V5_FINAL_FORMAL_GATE_PASS",
                "final_manifest_sha256": "a" * 64,
                "formal_plan_sha256": "b" * 64,
                "runs": 60,
                "plan": plan,
            }
            args = types.SimpleNamespace(
                package_root=package_root,
                package_manifest=manifest_path,
                authorization_dir=root / "authorization",
                evidence_root=evidence_root,
                remote_package_root=Path("/remote/package"),
                remote_output_root=Path("/remote/output"),
                ssh_target="pi@host",
                mac_python="python3",
            )
            live_preflight = {
                "status": "PASS",
                "pi_worker": {"boot_id": "boot-1"},
            }
            with mock.patch.object(
                execution, "validate_formal_authorization", return_value=receipt
            ), mock.patch.object(
                execution, "validate_materialized_package", return_value={}
            ), mock.patch.object(
                execution,
                "runtime_dependency_contract",
                return_value=(Path("dependency.json"), dependency_document(), "a" * 64),
            ), mock.patch.object(
                execution, "validate_runtime_preflight", return_value=live_preflight
            ), mock.patch.object(
                execution,
                "revalidate_complete_cell",
                side_effect=execution.ExecutionError("not complete"),
            ), mock.patch.object(execution, "execute_cell") as execute:
                with self.assertRaisesRegex(
                    execution.ExecutionError, "replacement authorization"
                ):
                    execution.formal_run(args)
            execute.assert_not_called()
            registry = (
                evidence_root / execution.FORMAL_REGISTRY
            ).read_text(encoding="utf-8")
            self.assertIn("BLOCKED_EXISTING_ATTEMPT_NOT_VALID_COMPLETE", registry)

    def test_registry_rejects_duplicate_start_and_unplanned_run(self):
        run = {
            "execution_index": 1,
            "run_id": "run-1",
            "policy": "Static-Light",
            "condition": "n16/B0",
            "seed": 5101,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.jsonl"
            start = {
                "event": "EXECUTION_STARTED",
                **run,
            }
            complete = {
                "event": "VALID_COMPLETE",
                "execution_index": 1,
                "run_id": "run-1",
            }
            path.write_text(
                json.dumps(start) + "\n" + json.dumps(complete) + "\n",
                encoding="utf-8",
            )
            receipt = execution.validate_formal_registry(path, [run])
            self.assertTrue(receipt["execution_started_once"])

            path.write_text(
                json.dumps(start)
                + "\n"
                + json.dumps(start)
                + "\n"
                + json.dumps(complete)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(execution.ExecutionError, "more than once"):
                execution.validate_formal_registry(path, [run])

            unplanned = {**start, "run_id": "run-2"}
            path.write_text(json.dumps(unplanned) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(execution.ExecutionError, "unplanned"):
                execution.validate_formal_registry(path, [run])


if __name__ == "__main__":
    unittest.main()
