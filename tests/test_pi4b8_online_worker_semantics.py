import csv
import inspect
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np

from online.pi4b8 import pi4b8_online_worker as worker
from runtime.action_pool_numpy import NumpyActionModel
from runtime.risk_proxy_numpy import RiskProxy
from scripts.train_q_tables import state_id as training_state_id


ROOT = Path(__file__).resolve().parents[1]
ACTIONS = [
    "fisvdd_revision",
    "lucid_revision",
    "tinydl_revision",
    "oi_svdd_as_elm_revision",
]


def q_config():
    return json.loads((ROOT / "configs/pi4b8_q_training_prereg_v1.json").read_text())


def corrected_reward_config():
    return json.loads(
        (ROOT / "configs/pi4b8_corrected_runtime_v5.json").read_text()
    )["reward"]


def corrected_state_config():
    state = q_config()["state"]
    state["previous_action_values"] = [*ACTIONS, "none"]
    return state


def threshold_config():
    # Sentinel values exercise the already preregistered branch ordering.  They
    # are test inputs, not proposed Pi4 thresholds.
    return {
        "T_hi_C": 80.0,
        "T_lo_C": 70.0,
        "q_hi_records": 256.0,
        "q_lo_records": 64.0,
        "p_hi": 0.70,
        "heavy_action": "tinydl_revision",
        "light_action": "fisvdd_revision",
    }


def action_quality():
    return {
        "fisvdd_revision": {"tpr": 0.10, "fpr": 0.90},
        "lucid_revision": {"tpr": 0.90, "fpr": 0.10},
        "tinydl_revision": {"tpr": 0.60, "fpr": 0.20},
        "oi_svdd_as_elm_revision": {"tpr": 0.50, "fpr": 0.30},
    }


class FrozenThresholdTests(unittest.TestCase):
    def test_thresholds_are_loaded_from_model_hyperparameters(self):
        thresholds = worker.load_model_thresholds(
            ROOT / "models/MODEL_HYPERPARAMETERS.json"
        )
        self.assertEqual(
            thresholds,
            {
                "fisvdd_revision": 0.05,
                "lucid_revision": 0.05,
                "tinydl_revision": 0.31499999999999995,
                "oi_svdd_as_elm_revision": 0.065,
            },
        )

    def test_missing_threshold_fails_closed_instead_of_defaulting_to_point_five(self):
        payload = json.loads(
            (ROOT / "models/MODEL_HYPERPARAMETERS.json").read_text()
        )
        del payload["actions"]["TinyDL"]["decision_threshold_selected_on_validation"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "MODEL_HYPERPARAMETERS.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises((KeyError, RuntimeError, ValueError)):
                worker.load_model_thresholds(path)

    def test_model_without_serialized_threshold_has_no_point_five_fallback(self):
        artifact = ROOT / "models/artifacts/tinydl_revision.npz"
        with self.assertRaisesRegex(RuntimeError, "bind the frozen validation threshold"):
            _ = NumpyActionModel(artifact).threshold
        self.assertEqual(
            NumpyActionModel(artifact, decision_threshold=0.315).threshold,
            0.315,
        )


class WindowBatchTests(unittest.TestCase):
    def test_flat_records_are_reshaped_into_true_sixteen_record_windows(self):
        flat = np.arange(2 * 16 * 3, dtype=np.float32).reshape(32, 3)
        windows = worker.reshape_window_records(
            flat, windows=2, records_per_window=16
        )
        self.assertEqual(windows.shape, (2, 16, 3))
        np.testing.assert_array_equal(windows[0], flat[:16])
        np.testing.assert_array_equal(windows[1], flat[16:])

    def test_one_record_per_window_input_is_rejected_for_n16(self):
        one_record_per_window = np.zeros((600, 191), dtype=np.float32)
        with self.assertRaises((RuntimeError, ValueError)):
            worker.reshape_window_records(
                one_record_per_window, windows=600, records_per_window=16
            )

    def test_batch_indices_must_match_the_frozen_seed_schedule(self):
        order = np.random.default_rng(5101).permutation(6000)
        expected = order[np.arange(600 * 16) % 6000].reshape(600, 16)
        observed = worker.validate_batch_indices(
            expected,
            windows=600,
            records_per_window=16,
            base_rows=6000,
            seed=5101,
        )
        np.testing.assert_array_equal(observed, expected)
        changed = expected.copy()
        changed[0, 0], changed[0, 1] = changed[0, 1], changed[0, 0]
        with self.assertRaisesRegex(RuntimeError, "frozen seed schedule"):
            worker.validate_batch_indices(
                changed,
                windows=600,
                records_per_window=16,
                base_rows=6000,
                seed=5101,
            )

    def test_raw_runtime_rows_reject_scoring_and_audit_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            path.write_text("feature,label\n1,0\n")
            with self.assertRaisesRegex(RuntimeError, "scoring/audit"):
                worker.read_raw_rows(path, windows=1, records_per_window=1)

    def test_raw_runtime_rows_reject_case_variant_label_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            path.write_text("feature,Scoring_Label\n1,0\n")
            with self.assertRaisesRegex(RuntimeError, "scoring/audit"):
                worker.read_raw_rows(path, windows=1, records_per_window=1)

    def test_raw_runtime_rows_reject_lineage_group_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            path.write_text("feature,feature_group_hex\n1,deadbeef\n")
            with self.assertRaisesRegex(RuntimeError, "scoring/audit"):
                worker.read_raw_rows(path, windows=1, records_per_window=1)


class RuntimeConfigurationGateTests(unittest.TestCase):
    def valid_config(self):
        return {
            "status": "FROZEN_BEFORE_CORRECTED_PI4B8_EXECUTION",
            "official_test_access": "forbidden",
            "action_order": ACTIONS,
            "action_roles": {
                "light": "fisvdd_revision",
                "heavy": "tinydl_revision",
            },
            "policies": ["FSSQL-R", "Safe-Greedy", "Threshold", "Static-Light"],
            "constraints": {"latency_cap_ms": 40.0, "thermal_cap_c": 82.0},
            "decision_window_ms": 100.0,
            "minimum_frequency_mhz": 1350.0,
            "energy_boundary": "mac_start_ack_to_end_ack",
            "background_load_lifetime": "before_start_ack_through_end_ack",
            "timing_boundaries": {
                "controller": "after_preprocessing_through_action_binding",
                "e2e": "preprocessing_start_through_post_action_telemetry",
                "switching": "resident_model_action_binding_only",
            },
            "workloads": {
                "n16/B0": {
                    "n_records": 16,
                    "background": "B0",
                    "busy_cpu_count": 0,
                    "background_duty": 0.0,
                    "background_stabilization_seconds": 0.5,
                }
            },
            "risk_aggregation": "mean_of_16_frozen_proxy_probabilities",
            "backlog_source": "frozen_runtime_input_records",
            "energy_state_initialization": 1.0,
            "energy_state_reset": "once_at_run_start",
            "threshold_queue_value": "n_records_plus_queued_backlog",
            "state": corrected_state_config(),
            "validation_action_quality": action_quality(),
            "validation_action_quality_source": {
                "path": "models/VALIDATION_METRICS.csv",
                "sha256": "7b2db09489ebc2c2e02c7604b0a7749a3a218d8d0b6abf547123f963affa1755",
                "derivation": (
                    "TPR=recall; validation prevalence is solved from "
                    "accuracy/precision/recall; FPR=FP/(1-prevalence)"
                ),
            },
            "reward": corrected_reward_config(),
            "normalized_energy_cost_by_condition": {
                "n16/B0": {action: 1.0 for action in ACTIONS}
            },
            "energy_state_decrement_by_condition": {
                "n16/B0": {action: 0.001 for action in ACTIONS}
            },
            "threshold_controller": {"uses_shield": False, **threshold_config()},
        }

    def test_unfrozen_risk_backlog_or_threshold_semantics_fail_closed(self):
        for field in ("risk_aggregation", "backlog_source", "threshold_controller"):
            with self.subTest(field=field):
                config = self.valid_config()
                del config[field]
                with self.assertRaises((KeyError, RuntimeError, TypeError, ValueError)):
                    worker.validate_runtime_config(config, "n16/B0", "Threshold")

    def test_runtime_config_requires_official_test_to_remain_forbidden(self):
        config = self.valid_config()
        config["official_test_access"] = "allowed"
        with self.assertRaises(RuntimeError):
            worker.validate_runtime_config(config, "n16/B0", "Static-Light")

    def test_background_label_without_executable_load_spec_fails_closed(self):
        config = self.valid_config()
        config["workloads"]["n16/B1"] = {
            "n_records": 16,
            "background": "B1",
            "background_duty": 0.5,
            "background_stabilization_seconds": 0.5,
        }
        with self.assertRaisesRegex(RuntimeError, "background-load mismatch"):
            worker.validate_runtime_config(config, "n16/B1", "Threshold")

    def test_background_load_matches_frozen_calibration_semantics(self):
        worker.validate_background_load(
            "n16/B2",
            {
                "background": "B2",
                "busy_cpu_count": 1,
                "background_duty": 1.0,
                "pinned_cpu": 3,
                "period_ms": 100.0,
                "background_stabilization_seconds": 0.5,
            },
        )

    def test_background_stabilization_must_equal_frozen_value(self):
        workload = {
            "background": "B2",
            "busy_cpu_count": 1,
            "background_duty": 1.0,
            "pinned_cpu": 3,
            "period_ms": 100.0,
            "background_stabilization_seconds": 0.6,
        }
        with self.assertRaisesRegex(RuntimeError, "frozen 0.5"):
            worker.validate_background_load("n16/B2", workload)

    def test_background_process_affinity_is_observed_from_the_os(self):
        with mock.patch.object(
            worker.os, "sched_getaffinity", return_value={3}, create=True
        ):
            self.assertEqual(worker.process_affinity(4312), [3])

    def test_background_load_fails_closed_when_observed_affinity_is_not_cpu3(self):
        class FakeEvent:
            def set(self):
                pass

            def wait(self, timeout=None):
                return True

        class FakeProcess:
            pid = 4312

            def __init__(self, **_kwargs):
                self.running = False

            def start(self):
                self.running = True

            def is_alive(self):
                return self.running

            def join(self, timeout=None):
                self.running = False

            def terminate(self):
                self.running = False

        class FakeContext:
            def Event(self):
                return FakeEvent()

            def Process(self, **kwargs):
                return FakeProcess(**kwargs)

        workload = {
            "busy_cpu_count": 1,
            "background_duty": 0.5,
            "period_ms": 100.0,
            "pinned_cpu": 3,
        }
        with (
            mock.patch.object(worker.mp, "get_context", return_value=FakeContext()),
            mock.patch.object(worker.time, "sleep"),
            mock.patch.object(
                worker.os, "sched_getaffinity", return_value={3}, create=True
            ),
        ):
            load = worker.BackgroundLoad(workload)
            load.start()
            self.assertEqual(load.pid, 4312)
            self.assertTrue(load.alive)
            self.assertEqual(load.affinity, [3])
            load.stop()
        with (
            mock.patch.object(worker.mp, "get_context", return_value=FakeContext()),
            mock.patch.object(worker.time, "sleep"),
            mock.patch.object(
                worker.os, "sched_getaffinity", return_value={2}, create=True
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "CPU-affinity mismatch"):
                worker.BackgroundLoad(workload).start()

    def test_state_metadata_must_match_numeric_action_encoding(self):
        state = corrected_state_config()
        worker.validate_state_definition(state)
        state["previous_action_values"] = ["none", *ACTIONS]
        with self.assertRaisesRegex(RuntimeError, "state encoding"):
            worker.validate_state_definition(state)

    def test_caps_timing_and_threshold_ranges_fail_closed(self):
        cases = (
            ("constraints", {"latency_cap_ms": 50.0, "thermal_cap_c": 82.0}),
            ("decision_window_ms", 200.0),
            ("minimum_frequency_mhz", 1000.0),
        )
        for field, value in cases:
            with self.subTest(field=field):
                config = self.valid_config()
                config[field] = value
                with self.assertRaises(RuntimeError):
                    worker.validate_runtime_config(config, "n16/B0", "Threshold")
        config = self.valid_config()
        config["threshold_controller"]["p_hi"] = 1.1
        with self.assertRaisesRegex(RuntimeError, "p_hi"):
            worker.validate_runtime_config(config, "n16/B0", "Threshold")

    def test_validation_metrics_are_the_reward_quality_source(self):
        observed = worker.load_validation_action_quality(
            ROOT / "models/VALIDATION_METRICS.csv"
        )
        expected = json.loads(
            (ROOT / "configs/pi4b8_corrected_runtime_v5.json").read_text()
        )["validation_action_quality"]
        for action in ACTIONS:
            for metric in ("tpr", "fpr", "f1"):
                self.assertAlmostEqual(observed[action][metric], expected[action][metric])

    def test_reward_formula_and_label_isolation_are_frozen(self):
        config = self.valid_config()
        config["reward"]["ground_truth_label_use"] = "allowed"
        with self.assertRaisesRegex(RuntimeError, "security/label"):
            worker.validate_runtime_config(config, "n16/B0", "Safe-Greedy")

    def test_legacy_single_record_q_lineage_is_rejected_for_true_n16(self):
        config = self.valid_config()
        old_q_config_path = ROOT / "configs/pi4b8_q_training_prereg_v1.json"
        old_manifest = json.loads(
            (ROOT / "q_training/pi4b8_q_v1/Q_TABLE_MANIFEST.json").read_text()
        )
        with self.assertRaisesRegex(RuntimeError, "not trained for true n=16"):
            worker.validate_fssql_q_lineage(
                config,
                json.loads(old_q_config_path.read_text()),
                old_q_config_path,
                old_manifest,
                ROOT / "q_training/pi4b8_q_v1/tables/fssql_r_seed_5101.npz",
                5101,
                "n16/B0",
            )


class RiskAndStateTests(unittest.TestCase):
    def test_worker_binds_the_frozen_risk_proxy_implementation(self):
        self.assertIs(worker.RiskProxy, RiskProxy)
        proxy = worker.RiskProxy(ROOT / "risk_proxy/risk_proxy.npz")
        risk_features = np.zeros((16, 8), dtype=np.float64)
        probabilities = proxy.predict(risk_features)
        self.assertEqual(probabilities.shape, (16,))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((0.0 <= probabilities) & (probabilities <= 1.0)).all())

    def test_online_state_encoding_matches_numeric_q_training_encoding(self):
        config = q_config()
        config["state"] = corrected_state_config()
        cases = [
            (0.10, 34.0, 1.0, 0.0, 4),
            (0.50, 45.0, 0.50, 2.0, 0),
            (0.99, 60.0, 0.01, 12.0, 3),
        ]
        for values in cases:
            with self.subTest(values=values):
                self.assertEqual(
                    worker.state_id(config, *values),
                    training_state_id(config, *values),
                )

    def test_state_rejects_nonfinite_or_out_of_domain_inputs(self):
        config = q_config()
        config["state"] = corrected_state_config()
        invalid_cases = (
            (float("nan"), 40.0, 1.0, 0.0, 4),
            (0.5, float("inf"), 1.0, 0.0, 4),
            (0.5, 40.0, 1.1, 0.0, 4),
            (0.5, 40.0, 1.0, -1.0, 4),
            (0.5, 40.0, 1.0, 0.0, 1.5),
        )
        for values in invalid_cases:
            with self.subTest(values=values):
                with self.assertRaises(RuntimeError):
                    worker.state_id(config, *values)

    def test_unaggregated_risk_vector_fails_closed(self):
        # No Pi4 window-level risk aggregation rule is frozen.  The worker must
        # not silently replace RiskProxy with an implicit mean.
        with self.assertRaises((TypeError, RuntimeError, ValueError)):
            worker.state_id(
                q_config(), np.array([0.1, 0.9]), 40.0, 1.0, 0.0, 4
            )


class ThermalAndGuardTests(unittest.TestCase):
    def test_thermal_prediction_uses_n16_and_background_duty(self):
        thermal = json.loads(
            (
                ROOT
                / "calibration/pi4b8/frozen_guard_fit_margin_v1_run2/thermal_predictor.json"
            ).read_text()
        )
        action = "tinydl_revision"
        temperature = 40.0
        latency_hat = 8.0
        n_records = 16
        background_duty = 1.0
        numeric = np.asarray(
            [temperature, np.log2(n_records), background_duty, latency_hat],
            dtype=np.float64,
        )
        mean = np.asarray(thermal["continuous_mean"], dtype=np.float64)
        scale = np.asarray(thermal["continuous_scale"], dtype=np.float64)
        indicators = np.asarray([0.0, 1.0, 0.0])
        expected = float(
            np.concatenate(([1.0], (numeric - mean) / scale, indicators))
            @ np.asarray(thermal["coefficients"], dtype=np.float64)
        )
        observed = worker.thermal_prediction(
            thermal,
            action,
            temperature,
            latency_hat,
            n_records,
            background_duty,
        )
        self.assertAlmostEqual(observed, expected, places=12)
        b0 = worker.thermal_prediction(
            thermal, action, temperature, latency_hat, n_records, 0.0
        )
        self.assertNotEqual(observed, b0)

    def test_safe_set_requires_both_latency_and_thermal_headroom(self):
        latency = dict(zip(ACTIONS, [5.0, 38.0, 5.0, 5.0]))
        thermal = dict(zip(ACTIONS, [79.0, 70.0, 80.0, 79.0]))
        margins = {
            "epsilon_tau_ms_by_action": {
                "fisvdd_revision": 0.0,
                "lucid_revision": 1.5,
                "tinydl_revision": 0.0,
                "oi_svdd_as_elm_revision": 0.0,
            },
            "fixed_latency_buffer_ms": 1.0,
            "epsilon_T_C_shared": 1.0,
            "fixed_thermal_buffer_c": 1.5,
        }
        safe = worker.build_safe_set(
            ACTIONS,
            latency,
            thermal,
            margins,
            latency_cap_ms=40.0,
            thermal_cap_c=82.0,
        )
        # LUCID fails latency (38+1.5+1 > 40); TinyDL fails thermal
        # (80+1+1.5 > 82).  FISVDD and OI satisfy both constraints.
        self.assertEqual(safe, [0, 3])

    def test_guard_rejects_nonfinite_predictions_and_negative_margins(self):
        margins = {
            "epsilon_tau_ms_by_action": {action: 0.0 for action in ACTIONS},
            "fixed_latency_buffer_ms": 1.0,
            "epsilon_T_C_shared": 1.0,
            "fixed_thermal_buffer_c": 1.0,
        }
        with self.assertRaisesRegex(RuntimeError, "invalid guard input"):
            worker.build_safe_set(
                ACTIONS,
                [1.0, float("nan"), 1.0, 1.0],
                [40.0] * 4,
                margins,
                40.0,
                82.0,
            )
        with self.assertRaisesRegex(RuntimeError, "invalid guard input"):
            worker.build_safe_set(
                ACTIONS, [1.0] * 4, [40.0, -1.0, 40.0, 40.0], margins, 40.0, 82.0
            )
        margins["epsilon_tau_ms_by_action"]["lucid_revision"] = -0.1
        with self.assertRaisesRegex(RuntimeError, "invalid guard input"):
            worker.build_safe_set(
                ACTIONS, [1.0] * 4, [40.0] * 4, margins, 40.0, 82.0
            )
        margins["epsilon_tau_ms_by_action"]["lucid_revision"] = 0.0
        margins["fixed_thermal_buffer_c"] = -0.5
        with self.assertRaisesRegex(RuntimeError, "caps and margins"):
            worker.build_safe_set(
                ACTIONS, [1.0] * 4, [40.0] * 4, margins, 40.0, 82.0
            )

    def test_latency_predictor_rejects_duplicate_or_nonfinite_cells(self):
        header = "action,workload_bin,background,tau_hat_ms\n"
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.csv"
            duplicate.write_text(
                header
                + "fisvdd_revision,n16,B0,1.0\n"
                + "fisvdd_revision,n16,B0,2.0\n"
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                worker.read_latency_predictor(duplicate)
            nonfinite = Path(directory) / "nonfinite.csv"
            nonfinite.write_text(header + "fisvdd_revision,n16,B0,nan\n")
            with self.assertRaisesRegex(RuntimeError, "invalid latency"):
                worker.read_latency_predictor(nonfinite)


class PolicySelectionTests(unittest.TestCase):
    def test_masked_argmax_counterexample(self):
        q_row = np.asarray([10.0, 1.0, 9.0, 0.0])
        self.assertEqual(worker.masked_argmax(q_row, [1, 2]), 2)

    def test_masked_argmax_empty_safe_set_fails_closed(self):
        with self.assertRaises((RuntimeError, ValueError)):
            worker.masked_argmax(np.zeros(4), [])

    def test_masked_argmax_rejects_complex_q_values(self):
        with self.assertRaisesRegex(RuntimeError, "invalid Q row"):
            worker.masked_argmax(np.asarray([1.0 + 2j, 0.0, 0.0, 0.0]), [0, 1])

    def test_threshold_controller_uses_temperature_backlog_and_hysteresis(self):
        config = threshold_config()
        self.assertEqual(
            worker.threshold_action(81.0, 0.0, 0.99, "lucid_revision", config),
            "fisvdd_revision",
        )
        self.assertEqual(
            worker.threshold_action(60.0, 300.0, 0.99, "lucid_revision", config),
            "fisvdd_revision",
        )
        self.assertEqual(
            worker.threshold_action(60.0, 0.0, 0.80, "lucid_revision", config),
            "tinydl_revision",
        )
        self.assertEqual(
            worker.threshold_action(75.0, 100.0, 0.80, "lucid_revision", config),
            "lucid_revision",
        )
        self.assertEqual(
            worker.threshold_action(75.0, 100.0, 0.80, None, config),
            "fisvdd_revision",
        )

    def test_missing_pi4_threshold_config_and_backlog_fail_closed(self):
        with self.assertRaises((KeyError, RuntimeError, TypeError, ValueError)):
            worker.threshold_action(40.0, 0.0, 0.8, None, None)
        with self.assertRaises((KeyError, RuntimeError, TypeError, ValueError)):
            worker.threshold_action(40.0, None, 0.8, None, threshold_config())

    def test_static_light_is_fisvdd_action_zero(self):
        self.assertEqual(worker.STATIC_LIGHT_ACTION_INDEX, 0)
        self.assertEqual(worker.ACTIONS[worker.STATIC_LIGHT_ACTION_INDEX], "fisvdd_revision")
        action = worker.select_policy_action(
            "Static-Light",
            safe_idx=[0, 1, 2, 3],
            risk=0.5,
            temperature=40.0,
            backlog=0.0,
            threshold_queue_load=16.0,
            previous_action=4,
            threshold_config=threshold_config(),
            q_row=np.zeros(4),
            reward_config=corrected_reward_config(),
            quality=action_quality(),
            energy_cost=np.full(4, 1.0),
        )
        self.assertEqual(action, 0)

    def test_safe_greedy_utility_depends_on_candidate_action(self):
        common = dict(
            safe_idx=[0, 1],
            risk=0.80,
            temperature=40.0,
            backlog=0.0,
            threshold_queue_load=16.0,
            previous_action=4,
            threshold_config=threshold_config(),
            q_row=np.zeros(4),
            reward_config=corrected_reward_config(),
            quality=action_quality(),
            energy_cost=np.full(4, 1.0),
        )
        self.assertEqual(worker.select_policy_action("Safe-Greedy", **common), 1)
        reversed_quality = action_quality()
        reversed_quality["fisvdd_revision"] = {"tpr": 0.99, "fpr": 0.01}
        reversed_quality["lucid_revision"] = {"tpr": 0.01, "fpr": 0.99}
        common["quality"] = reversed_quality
        self.assertEqual(worker.select_policy_action("Safe-Greedy", **common), 0)

    def test_fssql_uses_masked_argmax(self):
        action = worker.select_policy_action(
            "FSSQL-R",
            safe_idx=[1, 2],
            risk=0.5,
            temperature=40.0,
            backlog=0.0,
            threshold_queue_load=16.0,
            previous_action=4,
            threshold_config=threshold_config(),
            q_row=np.asarray([10.0, 1.0, 9.0, 0.0]),
            reward_config=corrected_reward_config(),
            quality=action_quality(),
            energy_cost=np.full(4, 1.0),
        )
        self.assertEqual(action, 2)

    def test_fssql_q_evidence_reproduces_masked_selection(self):
        evidence = worker.q_selection_evidence(
            "FSSQL-R",
            np.asarray([10.0, 1.0, 9.0, 0.0]),
            [1, 2],
            2,
        )
        self.assertEqual(
            json.loads(evidence["q_values_for_state"]), [10.0, 1.0, 9.0, 0.0]
        )
        self.assertEqual(json.loads(evidence["q_masked_candidate_indices"]), [1, 2])
        self.assertEqual(
            json.loads(evidence["q_masked_candidate_values"]), [1.0, 9.0]
        )
        self.assertEqual(evidence["q_masked_argmax_index"], 2)
        self.assertEqual(
            evidence["q_masked_argmax_rule"],
            "maximum_Q_then_lowest_action_index",
        )

    def test_fssql_q_evidence_rejects_an_unreproducible_selected_action(self):
        with self.assertRaisesRegex(RuntimeError, "not reproducible"):
            worker.q_selection_evidence(
                "FSSQL-R", np.asarray([10.0, 1.0, 9.0, 0.0]), [1, 2], 1
            )

    def test_safe_greedy_rejects_out_of_range_safe_index(self):
        with self.assertRaisesRegex(RuntimeError, "invalid action index"):
            worker.select_policy_action(
                "Safe-Greedy",
                safe_idx=[4],
                risk=0.5,
                temperature=40.0,
                backlog=0.0,
                threshold_queue_load=16.0,
                previous_action=4,
                threshold_config=threshold_config(),
                q_row=None,
                reward_config=corrected_reward_config(),
                quality=action_quality(),
                energy_cost=np.ones(4),
            )


class ScoringAndRewardTests(unittest.TestCase):
    def test_nonbinary_labels_are_rejected_before_integer_cast(self):
        scores = np.asarray([0.1, 0.9])
        for labels in (
            np.asarray([0.0, 0.5]),
            np.asarray([0.0, 1.9]),
            np.asarray([0, 256], dtype=np.int64),
        ):
            with self.subTest(labels=labels.tolist()):
                with self.assertRaisesRegex(RuntimeError, "non-binary"):
                    worker.score_predictions(scores, labels, 0.5)

    def test_vector_scoring_covers_all_sixteen_records(self):
        probabilities = np.asarray([0.9, 0.9, 0.1, 0.1] * 4)
        labels = np.asarray([1, 0, 1, 0] * 4, dtype=np.int8)
        result = worker.score_predictions(probabilities, labels, threshold=0.5)
        self.assertEqual(
            {name: int(result[name]) for name in ("tp", "fp", "fn", "tn")},
            {"tp": 4, "fp": 4, "fn": 4, "tn": 4},
        )
        self.assertEqual(len(result["predictions"]), 16)

    def test_scoring_rejects_misaligned_labels(self):
        with self.assertRaises((RuntimeError, ValueError)):
            worker.score_predictions(np.zeros(16), np.zeros(15), threshold=0.5)

    def test_scoring_rejects_complex_values_and_invalid_threshold(self):
        with self.assertRaisesRegex(RuntimeError, "real-valued"):
            worker.score_predictions(
                np.asarray([0.1 + 1j, 0.9]), np.asarray([0, 1]), threshold=0.5
            )
        with self.assertRaisesRegex(RuntimeError, "threshold"):
            worker.score_predictions(
                np.asarray([0.1, 0.9]), np.asarray([0, 1]), threshold=float("nan")
            )

    def test_reward_components_match_frozen_q_training_formula(self):
        quality = action_quality()
        energy_cost = np.asarray([1.0, 0.5, 0.8, 0.9])
        parts = worker.reward_components(
            corrected_reward_config(),
            risk=0.75,
            action_index=1,
            backlog=128.0,
            previous_action=0,
            quality=quality,
            energy_cost=energy_cost,
        )
        self.assertAlmostEqual(parts["security"], 0.65)
        self.assertAlmostEqual(parts["energy"], -0.05)
        self.assertAlmostEqual(parts["backlog"], -0.025)
        self.assertAlmostEqual(parts["switch"], -0.02)
        self.assertAlmostEqual(parts["total"], 0.555)

    def test_reward_function_has_no_ground_truth_label_parameter(self):
        parameters = inspect.signature(worker.reward_components).parameters
        self.assertNotIn("label", parameters)
        self.assertNotIn("labels", parameters)

    def test_main_loads_scoring_labels_only_after_detector_prediction(self):
        events = []

        class FakePreprocessor:
            def __init__(self, _path):
                pass

            def transform(self, _rows):
                events.append("preprocess")
                return frozen_features[batch_indices[0]]

        class FakeRiskProxy:
            def __init__(self, _path):
                pass

            def predict(self, values):
                events.append("risk_proxy")
                self.assertEqual(values.shape, (16, 8))
                return np.full(16, 0.75)

            def assertEqual(self, left, right):
                if left != right:
                    raise AssertionError(f"{left!r} != {right!r}")

        class FakeModel:
            def __init__(self, _path, decision_threshold=None):
                pass

            def predict_proba(self, values):
                events.append("prediction")
                if values.shape != (16, 3):
                    raise AssertionError(f"model received {values.shape}, not (16, 3)")
                return np.linspace(0.1, 0.9, 16)

        frozen_features = np.arange(6000 * 3, dtype=np.float32).reshape(6000, 3)
        risk_features = np.zeros((6000, 8), dtype=np.float64)
        labels = np.asarray([0, 1] * 3000, dtype=np.int8)
        batch_indices = (
            np.random.default_rng(9999).permutation(6000)[:16].reshape(1, 16)
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset_root = root / "assets"
            (asset_root / "guard").mkdir(parents=True)
            (asset_root / "models").mkdir()
            (asset_root / "preprocessing").mkdir()
            (asset_root / "risk_proxy").mkdir()
            (asset_root / "config").mkdir()
            (asset_root / "config/VALIDATION_METRICS.csv").write_bytes(
                (ROOT / "models/VALIDATION_METRICS.csv").read_bytes()
            )

            with (asset_root / "guard/latency_predictor.csv").open("w") as stream:
                stream.write("action,workload_bin,background,tau_hat_ms\n")
                for action in ACTIONS:
                    stream.write(f"{action},n16,B0,1.0\n")
            (asset_root / "guard/thermal_predictor.json").write_text(
                (
                    ROOT
                    / "calibration/pi4b8/frozen_guard_fit_margin_v1_run2/thermal_predictor.json"
                ).read_text()
            )
            (asset_root / "guard/guard_margins.json").write_text(
                (
                    ROOT
                    / "calibration/pi4b8/frozen_guard_fit_margin_v1_run2/guard_margins.json"
                ).read_text()
            )

            runtime_config = {
                "status": "FROZEN_BEFORE_CORRECTED_PI4B8_EXECUTION",
                "official_test_access": "forbidden",
                "action_order": ACTIONS,
                "action_roles": {
                    "light": "fisvdd_revision",
                    "heavy": "tinydl_revision",
                },
                "policies": ["Static-Light"],
                "workloads": {
                    "n16/B0": {
                        "n_records": 16,
                        "background": "B0",
                        "busy_cpu_count": 0,
                        "background_duty": 0.0,
                        "background_stabilization_seconds": 0.5,
                    }
                },
                "risk_aggregation": "mean_of_16_frozen_proxy_probabilities",
                "backlog_source": "frozen_runtime_input_records",
                "energy_state_initialization": 1.0,
                "energy_state_reset": "once_at_run_start",
                "threshold_queue_value": "n_records_plus_queued_backlog",
                "state": corrected_state_config(),
                "validation_action_quality": json.loads(
                    (ROOT / "configs/pi4b8_corrected_runtime_v5.json").read_text()
                )["validation_action_quality"],
                "validation_action_quality_source": {
                    "path": "models/VALIDATION_METRICS.csv",
                    "sha256": "7b2db09489ebc2c2e02c7604b0a7749a3a218d8d0b6abf547123f963affa1755",
                    "derivation": (
                        "TPR=recall; validation prevalence is solved from "
                        "accuracy/precision/recall; FPR=FP/(1-prevalence)"
                    ),
                },
                "reward": corrected_reward_config(),
                "normalized_energy_cost_by_condition": {
                    "n16/B0": {action: 1.0 for action in ACTIONS}
                },
                "energy_state_decrement_by_condition": {
                    "n16/B0": {action: 0.001 for action in ACTIONS}
                },
                "threshold_controller": {"uses_shield": False, **threshold_config()},
                "decision_window_ms": 100.0,
                "minimum_frequency_mhz": 1350.0,
                "constraints": {"latency_cap_ms": 40.0, "thermal_cap_c": 82.0},
                "energy_boundary": "mac_start_ack_to_end_ack",
                "background_load_lifetime": "before_start_ack_through_end_ack",
                "timing_boundaries": {
                    "controller": "after_preprocessing_through_action_binding",
                    "e2e": "preprocessing_start_through_post_action_telemetry",
                    "switching": "resident_model_action_binding_only",
                },
            }
            runtime_path = root / "runtime.json"
            runtime_path.write_text(json.dumps(runtime_config))
            q_manifest_path = root / "q_manifest.json"
            q_manifest_path.write_text(
                json.dumps(
                    {
                        "validation_action_quality": action_quality(),
                        "n1_B0_latency_hat_ms": {
                            action: float(index + 1)
                            for index, action in enumerate(ACTIONS)
                        },
                    }
                )
            )
            for name in (
                "package.json",
                "raw.csv",
                "features.npy",
                "risk.npy",
                "backlog.npy",
                "batch_indices.npy",
                "labels.npy",
            ):
                (root / name).touch()

            args = Namespace(
                asset_root=asset_root,
                package_manifest=root / "package.json",
                runtime_config=runtime_path,
                q_config=None,
                q_manifest=None,
                model_hyperparameters=ROOT / "models/MODEL_HYPERPARAMETERS.json",
                raw_features=root / "raw.csv",
                features=root / "features.npy",
                risk_features=root / "risk.npy",
                backlog=root / "backlog.npy",
                batch_indices=root / "batch_indices.npy",
                labels=root / "labels.npy",
                policy="Static-Light",
                condition="n16/B0",
                seed=9999,
                run_id="unit_label_order",
                q_table=None,
                output=root / "output",
                windows=1,
            )

            real_np_load = np.load

            def tracked_load(path, *load_args, **load_kwargs):
                path = Path(path)
                if path == args.features:
                    return frozen_features
                if path == args.risk_features:
                    return risk_features
                if path == args.backlog:
                    return np.asarray([0.0])
                if path == args.batch_indices:
                    return batch_indices
                if path == args.labels:
                    events.append("label_access")
                    if "prediction" not in events:
                        raise AssertionError("labels were read before detector prediction")
                    return labels
                return real_np_load(path, *load_args, **load_kwargs)

            with (
                mock.patch.object(worker, "parse_args", return_value=args),
                mock.patch.object(worker, "read_raw_base_rows", return_value=[{}] * 6000),
                mock.patch.object(
                    worker,
                    "validate_worker_cli_binding",
                    return_value={
                        "base_records": 6000,
                        "source_manifest_sha256": "0" * 64,
                    },
                ),
                mock.patch.object(worker, "PortablePreprocessor", FakePreprocessor),
                mock.patch.object(worker, "RiskProxy", FakeRiskProxy),
                mock.patch.object(worker, "NumpyActionModel", FakeModel),
                mock.patch.object(worker.np, "load", side_effect=tracked_load),
                mock.patch.object(worker, "temperature_c", return_value=40.0),
                mock.patch.object(worker, "frequencies_mhz", return_value=[1800.0]),
                mock.patch.object(worker, "throttled", return_value=("throttled=0x0", 0)),
                mock.patch("builtins.print"),
            ):
                worker.main()

            with (args.output / "windows.csv").open(newline="") as stream:
                timing_row = next(csv.DictReader(stream))

        self.assertLess(events.index("prediction"), events.index("label_access"))
        independently_measured = (
            "preprocessing_ms",
            "risk_proxy_ms",
            "telemetry_ms",
            "state_construction_ms",
            "latency_guard_ms",
            "thermal_guard_ms",
            "admissible_set_ms",
            "guard_total_ms",
            "policy_selection_ms",
            "detector_inference_ms",
            "post_action_telemetry_ms",
            "controller_ms",
            "e2e_ms",
            "scoring_ms",
        )
        for field in independently_measured:
            with self.subTest(field=field):
                self.assertGreater(float(timing_row[field]), 0.0)
        self.assertGreater(float(timing_row["e2e_ms"]), float(timing_row["controller_ms"]))
        self.assertGreater(
            float(timing_row["guard_total_ms"]),
            float(timing_row["admissible_set_ms"]),
        )
        self.assertEqual(
            json.loads(timing_row["batch_indices"]), batch_indices[0].tolist()
        )
        self.assertEqual(len(json.loads(timing_row["batch_indices"])), 16)
        self.assertEqual(
            int(timing_row["state_id"]),
            worker.state_id(
                runtime_config, 0.75, 40.0, 1.0, 0.0, worker.NO_PREVIOUS_ACTION
            ),
        )
        self.assertEqual(int(timing_row["selected_action_index"]), 0)
        self.assertEqual(
            int(timing_row["previous_action_index"]), worker.NO_PREVIOUS_ACTION
        )
        self.assertEqual(int(timing_row["switched"]), 0)
        safe_indices = json.loads(timing_row["safe_action_indices"])
        self.assertEqual(
            timing_row["safe_set_mask"],
            "".join("1" if index in safe_indices else "0" for index in range(4)),
        )
        self.assertEqual(int(timing_row["safe_set_size"]), len(safe_indices))
        self.assertEqual(timing_row["q_values_for_state"], "")
        self.assertEqual(timing_row["q_masked_candidate_indices"], "")
        self.assertAlmostEqual(float(timing_row["selected_decision_threshold"]), 0.05)
        self.assertEqual(len(json.loads(timing_row["prediction_scores"])), 16)
        self.assertEqual(len(json.loads(timing_row["prediction_bits"])), 16)
        self.assertEqual(len(json.loads(timing_row["labels_scoring_only"])), 16)
        self.assertEqual(
            sum(int(timing_row[name]) for name in ("tp", "fp", "fn", "tn")),
            16,
        )
        reward_sum = sum(
            float(timing_row[name])
            for name in (
                "reward_security",
                "reward_energy",
                "reward_backlog",
                "reward_switch",
            )
        )
        self.assertAlmostEqual(float(timing_row["reward"]), reward_sum)
        self.assertAlmostEqual(float(timing_row["reward_total"]), reward_sum)
        self.assertEqual(json.loads(timing_row["background_cpu_affinity"]), [])
        self.assertEqual(int(timing_row["background_load_alive"]), 1)


if __name__ == "__main__":
    unittest.main()
