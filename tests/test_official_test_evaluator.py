import copy
import importlib.util
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import run_official_test_once as evaluator


TRACE_OPENER_PATH = Path(__file__).resolve().parents[1] / "scripts/open_official_test.py"
trace_spec = importlib.util.spec_from_file_location("open_official_test", TRACE_OPENER_PATH)
trace_opener = importlib.util.module_from_spec(trace_spec)
assert trace_spec.loader is not None
trace_spec.loader.exec_module(trace_opener)


class OfficialTestEvaluatorConfigurationTests(unittest.TestCase):
    def test_frozen_thresholds_are_loaded_from_model_hyperparameters(self):
        thresholds = {
            detector: threshold
            for detector, _, threshold in evaluator.load_detector_specs()
        }
        self.assertEqual(
            thresholds,
            {
                "FISVDD": 0.05,
                "LUCID": 0.05,
                "TinyDL": 0.31499999999999995,
                "OI-SVDD+AS-ELM": 0.065,
            },
        )

    def test_threshold_value_is_configuration_driven(self):
        payload = json.loads(evaluator.MODEL_HYPERPARAMETERS.read_text())
        payload = copy.deepcopy(payload)
        payload["actions"]["TinyDL"]["decision_threshold_selected_on_validation"] = 0.42
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "MODEL_HYPERPARAMETERS.json"
            path.write_text(json.dumps(payload))
            thresholds = {
                detector: threshold
                for detector, _, threshold in evaluator.load_detector_specs(path)
            }
        self.assertEqual(thresholds["TinyDL"], 0.42)

    def test_missing_or_invalid_threshold_fails_closed(self):
        payload = json.loads(evaluator.MODEL_HYPERPARAMETERS.read_text())
        cases = []
        missing = copy.deepcopy(payload)
        del missing["actions"]["LUCID"]["decision_threshold_selected_on_validation"]
        cases.append(missing)
        invalid = copy.deepcopy(payload)
        invalid["actions"]["LUCID"]["decision_threshold_selected_on_validation"] = 1.5
        cases.append(invalid)

        with tempfile.TemporaryDirectory() as directory:
            for index, case in enumerate(cases):
                with self.subTest(index=index):
                    path = Path(directory) / f"case_{index}.json"
                    path.write_text(json.dumps(case))
                    with self.assertRaises(RuntimeError):
                        evaluator.load_detector_specs(path)

    def test_outputs_are_scoped_to_results_official_test(self):
        self.assertEqual(
            evaluator.OUTPUT_DIR,
            evaluator.ROOT / "results" / "official_test",
        )

    def test_existing_open_once_lock_prevents_a_second_evaluation(self):
        self.assertTrue(evaluator.OPEN_LOCK.is_file())
        with self.assertRaisesRegex(RuntimeError, "already opened"):
            evaluator.main()

    def test_matched_trace_opener_obeys_the_same_one_shot_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "OFFICIAL_TEST_OPENED_ONCE.json"
            lock.write_text("{}\n")
            with mock.patch.object(trace_opener, "OPEN_LOCK", lock):
                with self.assertRaisesRegex(RuntimeError, "already opened"):
                    trace_opener.main()


if __name__ == "__main__":
    unittest.main()
