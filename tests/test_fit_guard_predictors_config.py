import copy
import csv
import numpy as np
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

    def test_frozen_identity_and_cross_platform_numeric_reproduction(self):
        # Byte identity applies to stored artifacts. Re-fitting OLS on another
        # BLAS/platform is a numerical check, not a serialization identity check.
        # Tolerances are many orders below the logged sensor/timing precision.
        atol, rtol = 1e-10, 1e-12
        cases = [
            dict(name="pi4b8", collection=COLLECTION_CONFIG, predictor=PREDICTOR_CONFIG,
                 fit=FIT_SPLIT, margin=MARGIN_SPLIT, frozen=FROZEN_OUTPUT,
                 coverage=ROOT / "calibration/pi4b8/formal_coverage_20260910"),
            dict(name="pi3b", **PI3_CASE,
                 coverage=ROOT / "calibration/pi3b/protocol_v2/coverage/pi3bplus_v2_coverage_20260909T130714Z"),
        ]

        def compare_value(actual, expected):
            if isinstance(expected, dict):
                self.assertEqual(set(actual), set(expected))
                for key in expected:
                    compare_value(actual[key], expected[key])
            elif isinstance(expected, list):
                self.assertEqual(len(actual), len(expected))
                for a, b in zip(actual, expected):
                    compare_value(a, b)
            elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
                np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
            else:
                self.assertEqual(actual, expected)

        def bound_arrays(directory, rows):
            thermal = json.loads((directory / "thermal_predictor.json").read_text())
            margins = json.loads((directory / "guard_margins.json").read_text())
            lookup = {}
            with (directory / "latency_predictor.csv").open(newline="") as h:
                for row in csv.DictReader(h):
                    lookup[(row["action"], row["workload_bin"], row["background"])]=float(row["tau_hat_ms"])
            features = fitter.thermal_design(rows, np.asarray(thermal["continuous_mean"]), np.asarray(thermal["continuous_scale"]))
            t = features @ np.asarray(thermal["coefficients"])
            t += float(margins["epsilon_T_C_shared"]) + float(margins["fixed_thermal_buffer_c"])
            lat = np.asarray([lookup[fitter.cell(row)] + float(margins["epsilon_tau_ms_by_action"][row["action"]]) + float(margins["fixed_latency_buffer_ms"]) for row in rows])
            return lat, t

        with tempfile.TemporaryDirectory() as directory:
            for case in cases:
                with self.subTest(device=case["name"]):
                    # Original frozen bytes must match the original manifest.
                    old_summary = json.loads((case["frozen"] / "FIT_MARGIN_SUMMARY.json").read_text())
                    for name, expected_hash in old_summary["artifact_sha256"].items():
                        self.assertEqual(sha256(case["frozen"] / name), expected_hash)
                    output = Path(directory) / case["name"]
                    completed=run_fitter(case["predictor"],output,case["collection"],case["fit"],case["margin"])
                    self.assertEqual(completed.returncode,0,completed.stderr)
                    self.assertEqual(sha256(output / "latency_predictor.csv"), sha256(case["frozen"] / "latency_predictor.csv"))
                    for name in ("thermal_predictor.json", "guard_margins.json"):
                        compare_value(json.loads((output/name).read_text()), json.loads((case["frozen"]/name).read_text()))
                    with (output/"guard_residuals_fit_margin.csv").open(newline="") as a, (case["frozen"]/"guard_residuals_fit_margin.csv").open(newline="") as b:
                        aa,bb=list(csv.DictReader(a)),list(csv.DictReader(b))
                    self.assertEqual(len(aa),len(bb))
                    for ra,rb in zip(aa,bb):
                        self.assertEqual(set(ra),set(rb))
                        for k in ra:
                            try:
                                va,vb=float(ra[k]),float(rb[k])
                            except ValueError:
                                self.assertEqual(ra[k],rb[k])
                            else:
                                np.testing.assert_allclose(va,vb,atol=atol,rtol=rtol)
                    # Tolerant floats must not hide a changed coverage/admission decision.
                    with (case["coverage"]/"raw_windows.csv").open(newline="") as h:
                        rows=list(csv.DictReader(h))
                    old_lat,old_t=bound_arrays(case["frozen"],rows)
                    new_lat,new_t=bound_arrays(output,rows)
                    measured_lat=np.asarray([float(row["end_to_end_ms"]) for row in rows])
                    measured_t=np.asarray([float(row["T_end_C"]) for row in rows])
                    for left,right in [(measured_lat<=old_lat,measured_lat<=new_lat),
                                       (measured_t<=old_t,measured_t<=new_t),
                                       (old_lat<=40.,new_lat<=40.),(old_t<=82.,new_t<=82.)]:
                        np.testing.assert_array_equal(left,right)

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
