import csv
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import analyze_pi4b8_formal as analysis
from online.pi4b8 import pi4b8_formal_runner as gate
from online.pi4b8 import run_pi4b8_formal_matrix as matrix


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_meter(path, times):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "timestamp_utc",
                "sample_monotonic",
                "voltage_V",
                "current_A",
                "power_W",
                "raw_hex",
            ),
        )
        writer.writeheader()
        for value in times:
            writer.writerow(
                {
                    "timestamp_utc": "2026-09-11T00:00:00+00:00",
                    "sample_monotonic": value,
                    "voltage_V": 5.0,
                    "current_A": 0.4,
                    "power_W": 2.0,
                    "raw_hex": "00",
                }
            )


def handshake(run_id="run-1"):
    return {
        "run_id": run_id,
        "meter_capture_start_mac_monotonic": -1.0,
        "start_request_mac_monotonic": 4.0,
        "start_ack_receive_mac_monotonic": 5.0,
        "end_ack_receive_mac_monotonic": 15.0,
        "meter_capture_end_mac_monotonic": 21.0,
        "pi_start_ack": {
            "event": "RUN_START_ACK",
            "run_id": run_id,
            "pi_utc": 1.0,
            "pi_monotonic": 10.0,
        },
        "pi_end_ack": {
            "event": "RUN_END_ACK",
            "run_id": run_id,
            "pi_utc": 2.0,
            "pi_monotonic": 20.0,
            "windows_completed": 2,
        },
        "status": "VALID_COMPLETE",
    }


class FormalGateTests(unittest.TestCase):
    def test_historical_v4_matrix_launcher_is_disabled(self):
        with self.assertRaisesRegex(matrix.RunValidationError, "historical V4 launcher disabled"):
            matrix.main()

    def test_pi4b_background_labels_fail_closed_without_frozen_load_commands(self):
        config = {
            "workload": {
                "background_robustness": [
                    {"n_records": 16, "background": "B1"},
                    {"n_records": 16, "background": "B2"},
                ]
            }
        }
        with self.assertRaisesRegex(gate.GateError, "B1/B2 formal execution blocked"):
            gate.validate_background_load_freeze(config)

    def test_formal_authorization_checks_status_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocols = root / "protocols"
            online = root / "online/pi4b8"
            protocols.mkdir(parents=True)
            online.mkdir(parents=True)
            validation_path = protocols / "PI4B8_COMMISSIONING_VALIDATION.json"
            decision_path = protocols / "PI4B8_COMMISSIONING_DECISION.md"
            runtime_path = protocols / "PI4B8_FORMAL_RUNTIME_PACKAGE_MANIFEST_V4.json"
            orchestrator_path = online / "pi4b8_commissioning_orchestrator.py"
            final_path = protocols / "PI4B8_FINAL_EXECUTION_MANIFEST.json"
            validation = {
                "status": "PASS",
                "marker": "COMMISSIONING_ONLY_EXCLUDED_FROM_FORMAL_STATISTICS",
                "run_id": "commissioning-1",
                "checks": {
                    "windows_600": True,
                    "timing_fields": True,
                    "tp_fp_fn_tn_recomputable": True,
                    "meter_monotonic": True,
                    "ordering": True,
                    "energy_reproducible": True,
                    "run_id_binding": True,
                    "label_leakage": True,
                    "throttle_zero": True,
                    "pre_roll_s": 5.0,
                    "post_roll_s": 5.0,
                },
            }
            validation_path.write_text(json.dumps(validation))
            decision_path.write_text("COMMISSIONING PASS commissioning-1\n")
            runtime_path.write_text("{}\n")
            orchestrator_path.write_text("# frozen\n")
            final_path.write_text(
                json.dumps(
                    {
                        "status": "PI4B8_FINAL_EXECUTION_MANIFEST",
                        "commissioning_run_id": "commissioning-1",
                        "commissioning_validation_sha256": sha256(validation_path),
                        "commissioning_decision_sha256": sha256(decision_path),
                        "orchestrator_sha256": sha256(orchestrator_path),
                        "runtime_package_manifest_sha256": sha256(runtime_path),
                    }
                )
            )
            gate.validate_formal_authorization(root)
            decision_path.write_text("modified\n")
            with self.assertRaisesRegex(gate.GateError, "commissioning decision"):
                gate.validate_formal_authorization(root)

    def test_launcher_requires_explicit_final_gate_pass(self):
        passed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='{"status":"PI4B8_FINAL_FORMAL_GATE_PASS"}', stderr=""
        )
        with mock.patch.object(matrix.subprocess, "run", return_value=passed) as called:
            self.assertEqual(
                matrix.run_formal_gate("python3")["status"],
                "PI4B8_FINAL_FORMAL_GATE_PASS",
            )
        self.assertIn("--formal", called.call_args.args[0])

        wrong = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='{"status":"LINEAGE_ONLY"}', stderr=""
        )
        with mock.patch.object(matrix.subprocess, "run", return_value=wrong):
            with self.assertRaises(matrix.RunValidationError):
                matrix.run_formal_gate("python3")


class RawEvidenceValidationTests(unittest.TestCase):
    def test_power_capture_and_analysis_use_strict_ack_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meter_path = root / "POWERZ_raw.csv"
            handshake_path = root / "HANDSHAKE.json"
            validation_path = root / "RUN_VALIDATION.json"
            write_meter(meter_path, range(21))
            bound = handshake()
            result = matrix.validate_power_capture(meter_path, bound, run_id="run-1")
            self.assertEqual(result["energy_J"], 20.0)
            self.assertEqual(result["pre_roll_s"], 5.0)
            self.assertEqual(result["post_roll_s"], 5.0)
            handshake_path.write_text(json.dumps(bound))
            validation_path.write_text(
                json.dumps(
                    {
                        "status": "VALID_COMPLETE",
                        "run_id": "run-1",
                        "powerz": {
                            "sha256": sha256(meter_path),
                            "energy_J": result["energy_J"],
                        },
                        "handshake_sha256": sha256(handshake_path),
                    }
                )
            )
            reproduced = analysis.validated_energy(
                meter_path,
                handshake_path,
                validation_path,
                expected_run_id="run-1",
            )
            self.assertEqual(reproduced["energy_J"], 20.0)

    def test_power_capture_rejects_nonmonotonic_outside_and_short_roll(self):
        cases = [
            ([0, 1, 1, 20], handshake(), "strictly monotonic"),
            ([6, 7, 14, 15], handshake(), "outside POWER-Z"),
            ([1, 5, 15, 19], handshake(), "insufficient POWER-Z roll"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, (times, bound, message) in enumerate(cases):
                with self.subTest(index=index):
                    path = Path(directory) / f"meter-{index}.csv"
                    write_meter(path, times)
                    with self.assertRaisesRegex(matrix.RunValidationError, message):
                        matrix.validate_power_capture(path, bound, run_id="run-1")

    def test_windows_validation_checks_rows_hash_identity_and_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windows.csv"
            fields = [
                "window_id", "policy", "condition", "seed", "predicted_latency_ms",
                "latency_headroom_ms", *matrix.TIMING_FIELDS, "temperature_C",
                "cpu_frequency_mhz_min", "throttled", "prediction",
                "label_scoring_only", "tp", "fp", "fn", "tn",
            ]
            rows = []
            for window_id, prediction, label in ((0, 1, 1), (1, 0, 0)):
                row = {field: 0 for field in fields}
                row.update(
                    {
                        "window_id": window_id,
                        "policy": "Threshold",
                        "condition": "n16/B0",
                        "seed": 5101,
                        "predicted_latency_ms": 1.0,
                        "latency_headroom_ms": 39.0,
                        "temperature_C": 40.0,
                        "cpu_frequency_mhz_min": 1800.0,
                        "throttled": "throttled=0x0",
                        "prediction": prediction,
                        "label_scoring_only": label,
                        "tp": int(prediction == 1 and label == 1),
                        "fp": int(prediction == 1 and label == 0),
                        "fn": int(prediction == 0 and label == 1),
                        "tn": int(prediction == 0 and label == 0),
                    }
                )
                rows.append(row)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            end_ack = {"windows_completed": 2, "raw_artifact_sha256": sha256(path)}
            result = matrix.validate_windows(
                path,
                run_id="run-1",
                policy="Threshold",
                condition="n16/B0",
                seed=5101,
                windows=2,
                min_frequency_mhz=1350,
                end_ack=end_ack,
            )
            self.assertEqual(result["rows"], 2)

            text = path.read_text().replace("throttled=0x0", "throttled=0x1", 1)
            path.write_text(text)
            end_ack["raw_artifact_sha256"] = sha256(path)
            with self.assertRaisesRegex(matrix.RunValidationError, "throttle/undervoltage"):
                matrix.validate_windows(
                    path,
                    run_id="run-1",
                    policy="Threshold",
                    condition="n16/B0",
                    seed=5101,
                    windows=2,
                    min_frequency_mhz=1350,
                    end_ack=end_ack,
                )


if __name__ == "__main__":
    unittest.main()
