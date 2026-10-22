import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from device_calibration import fit_guard_predictors as fitter


ROOT = Path(__file__).resolve().parents[1]
COLLECTION_CONFIG = ROOT / "calibration/pi4b8/PI4B8_CALIBRATION_CONFIG.json"
PREDICTOR_CONFIG = ROOT / "configs/pi4b8_guard_predictor_protocol_v1.json"
FIT_SPLIT = ROOT / "calibration/pi4b8/formal_fit_20260910_v3"
MARGIN_SPLIT = ROOT / "calibration/pi4b8/formal_margin_20260910"
FROZEN_OUTPUT = ROOT / "calibration/pi4b8/frozen_guard_fit_margin_v1_run2"
PI3_CASE = {
    "collection": ROOT / "configs/pi3b_calibration_protocol_v2.json",
    "predictor": ROOT / "configs/pi3b_guard_predictor_protocol_v2.json",
    "fit": ROOT / "calibration/pi3b/protocol_v2/fit/pi3bplus_v2_fit_20260909T124920Z",
    "margin": ROOT / "calibration/pi3b/protocol_v2/margin/pi3bplus_v2_margin_20260909T125804Z",
    "frozen": ROOT / "calibration/pi3b/protocol_v2/frozen_guard_fit_margin_v1",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_fitter(
    predictor_config,
    output,
    collection_config=COLLECTION_CONFIG,
    fit_split=FIT_SPLIT,
    margin_split=MARGIN_SPLIT,
):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "device_calibration/fit_guard_predictors.py"),
            "--collection-config",
            str(collection_config),
            "--predictor-config",
            str(predictor_config),
            "--fit",
            str(fit_split),
            "--margin",
            str(margin_split),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )


class GuardPredictorConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collection = json.loads(COLLECTION_CONFIG.read_text())
        cls.predictor = json.loads(PREDICTOR_CONFIG.read_text())

    def test_current_frozen_configuration_is_accepted(self):
        settings = fitter.validate_predictor_config(self.predictor, self.collection)
        self.assertEqual(settings["cell_keys"], fitter.CELL_KEYS)
        self.assertEqual(settings["latency_quantile"], 0.99)
        self.assertEqual(settings["margin_quantile"], 0.99)
        self.assertEqual(settings["continuous_features"], fitter.CONTINUOUS_FEATURES)

    def test_unsupported_core_values_fail_closed(self):
        mutations = [
            ("latency type", lambda value: value["latency_predictor"].update(type="other")),
            ("cell keys", lambda value: value["latency_predictor"].update(cell_keys=["background"])),
            ("latency method", lambda value: value["latency_predictor"].update(quantile_method="linear")),
            ("thermal solver", lambda value: value["thermal_predictor"].update(solver="other")),
            ("thermal intercept", lambda value: value["thermal_predictor"].update(intercept=False)),
            ("thermal interactions", lambda value: value["thermal_predictor"].update(interactions=["x:y"])),
            ("margin method", lambda value: value["margin"].update(quantile_method="linear")),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                candidate = copy.deepcopy(self.predictor)
                mutate(candidate)
                with self.assertRaises(RuntimeError):
                    fitter.validate_predictor_config(candidate, self.collection)

    def test_configured_quantile_uses_requested_quantile(self):
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(fitter.configured_quantile(values, 0.5, "higher"), 3.0)
        self.assertEqual(fitter.configured_quantile(values, 0.99, "higher"), 4.0)

    def test_current_configuration_reproduces_frozen_numeric_artifacts(self):
        names = [
            "latency_predictor.csv",
            "thermal_predictor.json",
            "guard_margins.json",
            "guard_residuals_fit_margin.csv",
        ]
        cases = [
            {
                "name": "pi4b8",
                "collection": COLLECTION_CONFIG,
                "predictor": PREDICTOR_CONFIG,
                "fit": FIT_SPLIT,
                "margin": MARGIN_SPLIT,
                "frozen": FROZEN_OUTPUT,
            },
            {"name": "pi3b", **PI3_CASE},
        ]
        with tempfile.TemporaryDirectory() as directory:
            for case in cases:
                with self.subTest(device=case["name"]):
                    output = Path(directory) / case["name"]
                    completed = run_fitter(
                        case["predictor"],
                        output,
                        case["collection"],
                        case["fit"],
                        case["margin"],
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    for name in names:
                        self.assertEqual(
                            sha256(output / name),
                            sha256(case["frozen"] / name),
                        )

    def test_conflicting_margin_configuration_fails_without_output(self):
        candidate = copy.deepcopy(self.predictor)
        candidate["margin"]["quantile"] = 0.5
        candidate["margin"]["one_sided_rule"] = "max(0, Q0.5(residual))"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config = directory / "predictor.json"
            output = directory / "output"
            config.write_text(json.dumps(candidate))
            completed = run_fitter(config, output)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("margin quantile differs", completed.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
