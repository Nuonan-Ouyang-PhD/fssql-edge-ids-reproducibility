#!/usr/bin/env python3
"""Pi 4B online worker used by the corrected runtime package.

The historical V4 worker and evidence remain immutable. A future formal run
must package this worker with a new hash-bound configuration and true
``windows x records`` inputs.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if (SCRIPT_DIR / "runtime").is_dir():
    sys.path.insert(0, str(SCRIPT_DIR / "runtime"))
    from action_pool_numpy import NumpyActionModel  # type: ignore  # noqa: E402
    from corrected_v5_contract import (  # type: ignore  # noqa: E402
        sha256 as package_sha256,
        validate_worker_cli_binding,
    )
    from preprocessor_numpy import PortablePreprocessor  # type: ignore  # noqa: E402
    from risk_proxy_numpy import RiskProxy  # type: ignore  # noqa: E402
else:
    sys.path.insert(0, str(REPO_ROOT))
    from online.pi4b8.corrected_v5_contract import (  # noqa: E402
        sha256 as package_sha256,
        validate_worker_cli_binding,
    )
    from runtime.action_pool_numpy import NumpyActionModel  # noqa: E402
    from runtime.preprocessor_numpy import PortablePreprocessor  # noqa: E402
    from runtime.risk_proxy_numpy import RiskProxy  # noqa: E402


ACTIONS = (
    "fisvdd_revision",
    "lucid_revision",
    "tinydl_revision",
    "oi_svdd_as_elm_revision",
)
ACTION_CONFIG_NAMES = {
    "fisvdd_revision": "FISVDD",
    "lucid_revision": "LUCID",
    "tinydl_revision": "TinyDL",
    "oi_svdd_as_elm_revision": "OI-SVDD+AS-ELM",
}
STATIC_LIGHT_ACTION_INDEX = 0
NO_PREVIOUS_ACTION = len(ACTIONS)
SUPPORTED_POLICIES = {"FSSQL-R", "Safe-Greedy", "Threshold", "Static-Light"}


def busy_worker(stop: Any, duty_cycle: float, period_s: float, cpu: int) -> None:
    """Run the frozen pinned-CPU background load used by Pi4B calibration."""
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("CPU-affinity control is unavailable on this host")
    os.sched_setaffinity(0, {cpu})
    busy_s = period_s * duty_cycle
    idle_s = period_s - busy_s
    while not stop.is_set():
        start = time.perf_counter()
        while time.perf_counter() - start < busy_s:
            pass
        if idle_s > 0:
            stop.wait(idle_s)


class BackgroundLoad:
    """Lifecycle for the exact B0/B1/B2 load encoded in the runtime config."""

    def __init__(self, workload: dict[str, Any]):
        self.workload = workload
        self.stop_event: Any = None
        self.process: mp.Process | None = None

    def start(self) -> None:
        if int(self.workload["busy_cpu_count"]) == 0:
            return
        context = mp.get_context("spawn")
        self.stop_event = context.Event()
        self.process = context.Process(
            target=busy_worker,
            args=(
                self.stop_event,
                float(self.workload["background_duty"]),
                float(self.workload["period_ms"]) / 1000.0,
                int(self.workload["pinned_cpu"]),
            ),
            daemon=True,
        )
        self.process.start()
        time.sleep(0.05)
        if not self.process.is_alive():
            self.stop()
            raise RuntimeError("background-load process failed to start")

    def stop(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.process is not None:
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2.0)

    @property
    def pid(self) -> int | None:
        return None if self.process is None else self.process.pid

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.is_alive()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def temperature_c() -> float:
    return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()) / 1000.0


def frequencies_mhz() -> list[float]:
    paths = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"))
    if not paths:
        raise RuntimeError("CPU frequency telemetry is unavailable")
    return [int(path.read_text().strip()) / 1000.0 for path in paths]


def throttled() -> tuple[str, int]:
    result = subprocess.run(
        ["vcgencmd", "get_throttled"], text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(f"vcgencmd failed: {result.stderr.strip()}")
    text = result.stdout.strip()
    return text, int(text.split("=")[-1], 16)


def load_model_thresholds(path: Path) -> dict[str, float]:
    """Read the sole runtime threshold source and fail closed on omissions."""
    document = json.loads(path.read_text())
    configured = document.get("actions")
    if not isinstance(configured, dict):
        raise RuntimeError("MODEL_HYPERPARAMETERS.json has no actions object")
    thresholds: dict[str, float] = {}
    for action in ACTIONS:
        name = ACTION_CONFIG_NAMES[action]
        try:
            value = float(configured[name]["decision_threshold_selected_on_validation"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"missing decision threshold for {name}") from error
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(f"invalid decision threshold for {name}: {value}")
        thresholds[action] = value
    return thresholds


def load_validation_action_quality(path: Path) -> dict[str, dict[str, float]]:
    """Derive the label-free reward inputs from the frozen validation CSV."""
    by_name = {name: action for action, name in ACTION_CONFIG_NAMES.items()}
    quality: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        name = row.get("action", "")
        if name not in by_name or by_name[name] in quality:
            raise RuntimeError(f"unexpected or duplicate validation action: {name!r}")
        try:
            accuracy = float(row["accuracy"])
            precision = float(row["precision"])
            recall = float(row["recall"])
            f1 = float(row["f1"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("validation metrics are incomplete") from error
        if (
            not all(math.isfinite(value) for value in (accuracy, precision, recall, f1))
            or not 0.0 <= accuracy <= 1.0
            or not 0.0 < precision <= 1.0
            or not 0.0 <= recall <= 1.0
            or not 0.0 <= f1 <= 1.0
        ):
            raise RuntimeError("validation metrics are outside their valid ranges")
        coefficient = -1.0 + recall - recall * (1.0 / precision - 1.0)
        if coefficient == 0.0:
            raise RuntimeError("validation prevalence is not identifiable")
        prevalence = (accuracy - 1.0) / coefficient
        false_positive_fraction = recall * prevalence * (1.0 / precision - 1.0)
        if not 0.0 < prevalence < 1.0:
            raise RuntimeError("derived validation prevalence is invalid")
        fpr = false_positive_fraction / (1.0 - prevalence)
        if not math.isfinite(fpr) or not 0.0 <= fpr <= 1.0 + 1e-12:
            raise RuntimeError("derived validation FPR is invalid")
        quality[by_name[name]] = {"tpr": recall, "fpr": min(fpr, 1.0), "f1": f1}
    if set(quality) != set(ACTIONS):
        raise RuntimeError("validation metrics do not contain all four actions")
    return quality


def reshape_window_records(
    values: np.ndarray,
    windows: int,
    records_per_window: int,
    name: str = "array",
) -> np.ndarray:
    """Return an explicit window/record tensor; never duplicate a row."""
    array = np.asarray(values)
    if array.ndim >= 2 and array.shape[:2] == (windows, records_per_window):
        return array
    if array.ndim >= 1 and array.shape[0] == windows * records_per_window:
        return array.reshape((windows, records_per_window) + array.shape[1:])
    raise RuntimeError(
        f"{name} must contain exactly {windows} x {records_per_window} records; "
        f"observed shape={array.shape}"
    )


def validate_batch_indices(
    values: np.ndarray,
    *,
    windows: int,
    records_per_window: int,
    base_rows: int,
    seed: int,
) -> np.ndarray:
    """Validate the frozen 600 x 16 base-row selection without copying labels."""
    indices = np.asarray(values)
    if (
        indices.shape != (windows, records_per_window)
        or not np.issubdtype(indices.dtype, np.integer)
        or np.iscomplexobj(indices)
    ):
        raise RuntimeError(
            f"batch indices must be an integer ({windows}, {records_per_window}) array"
        )
    if base_rows <= 0 or int(indices.min()) < 0 or int(indices.max()) >= base_rows:
        raise RuntimeError("batch indices reference rows outside the frozen base input")
    if any(len(set(row.tolist())) != records_per_window for row in indices):
        raise RuntimeError("a true-n16 window repeats a base record")
    order = np.random.default_rng(seed).permutation(base_rows)
    expected = order[
        np.arange(windows * records_per_window, dtype=np.int64) % base_rows
    ].reshape(windows, records_per_window)
    if not np.array_equal(indices, expected):
        raise RuntimeError("batch indices do not reproduce the frozen seed schedule")
    return indices.astype(np.int64, copy=False)


def validate_state_definition(state: dict[str, Any]) -> None:
    expected_components = [
        "risk_bin",
        "temperature_bin",
        "energy_bin",
        "backlog_bin",
        "previous_action",
    ]
    if (
        state.get("ordered_components") != expected_components
        or state.get("previous_action_values") != [*ACTIONS, "none"]
        or int(state.get("number_of_states", 0)) != 1280
        or int(state.get("number_of_actions", 0)) != 4
        or int(state.get("number_of_state_action_pairs", 0)) != 5120
    ):
        raise RuntimeError("runtime state encoding is not frozen")
    for name in (
        "risk_bin_upper_edges",
        "temperature_C_upper_edges",
        "energy_residual_upper_edges",
        "backlog_upper_edges",
    ):
        try:
            edges = np.asarray(state[name], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"invalid state-bin definition: {name}") from error
        if edges.shape != (3,) or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
            raise RuntimeError(f"invalid state-bin definition: {name}")


def state_id(
    config: dict[str, Any],
    risk: float,
    temperature: float,
    energy: float,
    backlog: float,
    previous_action: int,
) -> int:
    state = config["state"]
    validate_state_definition(state)
    raw_numeric = np.asarray([risk, temperature, energy, backlog])
    if np.iscomplexobj(raw_numeric):
        raise RuntimeError("state inputs must be real-valued")
    numeric = np.asarray(raw_numeric, dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise RuntimeError("state inputs must be finite")
    if not 0.0 <= numeric[0] <= 1.0:
        raise RuntimeError("risk state input must be in [0,1]")
    if numeric[1] < 0.0 or not 0.0 <= numeric[2] <= 1.0 or numeric[3] < 0.0:
        raise RuntimeError("temperature, energy, or backlog state input is invalid")
    if isinstance(previous_action, (bool, np.bool_)) or not isinstance(
        previous_action, (int, np.integer)
    ):
        raise RuntimeError("previous action is not a frozen action index")
    components = (
        int(np.digitize(numeric[0], state["risk_bin_upper_edges"])),
        int(np.digitize(numeric[1], state["temperature_C_upper_edges"])),
        int(np.digitize(numeric[2], state["energy_residual_upper_edges"])),
        int(np.digitize(numeric[3], state["backlog_upper_edges"])),
        int(previous_action),
    )
    cardinalities = (4, 4, 4, 4, 5)
    if any(not 0 <= value < cardinality for value, cardinality in zip(components, cardinalities)):
        raise RuntimeError(f"state component outside frozen bins: {components}")
    result = 0
    for value, cardinality in zip(components, cardinalities):
        result = result * cardinality + value
    return result


def thermal_prediction(
    thermal: dict[str, Any],
    action: str,
    temperature: float,
    latency_hat: float,
    n_records: int,
    background_duty: float,
) -> float:
    if n_records <= 0:
        raise RuntimeError("n_records must be positive")
    if action not in ACTIONS:
        raise RuntimeError(f"unknown thermal-predictor action: {action}")
    if thermal.get("continuous_feature_order") != [
        "T_start_C",
        "log2_n_records",
        "background_duty",
        "end_to_end_ms",
    ] or thermal.get("feature_order") != [
        "intercept",
        "scaled_T_start_C",
        "scaled_log2_n_records",
        "scaled_background_duty",
        "scaled_end_to_end_ms",
        "is_lucid_revision",
        "is_tinydl_revision",
        "is_oi_svdd_as_elm_revision",
    ]:
        raise RuntimeError("thermal predictor feature order mismatch")
    numeric = np.asarray(
        [temperature, math.log2(n_records), background_duty, latency_hat], dtype=np.float64
    )
    mean = np.asarray(thermal["continuous_mean"], dtype=np.float64)
    scale = np.asarray(thermal["continuous_scale"], dtype=np.float64)
    if (
        numeric.shape != mean.shape
        or mean.shape != scale.shape
        or not np.isfinite(numeric).all()
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise RuntimeError("thermal predictor shape/scaling mismatch")
    indicators = np.asarray(
        [
            float(action == "lucid_revision"),
            float(action == "tinydl_revision"),
            float(action == "oi_svdd_as_elm_revision"),
        ],
        dtype=np.float64,
    )
    design = np.concatenate(([1.0], (numeric - mean) / scale, indicators))
    coefficients = np.asarray(thermal["coefficients"], dtype=np.float64)
    if design.shape != coefficients.shape or not np.isfinite(coefficients).all():
        raise RuntimeError("thermal predictor coefficient mismatch")
    prediction = float(design @ coefficients)
    if not math.isfinite(prediction):
        raise RuntimeError("thermal predictor produced a non-finite value")
    return prediction


def build_safe_set(
    actions: Sequence[str],
    latency_predictions: Sequence[float],
    thermal_predictions: Sequence[float],
    margins: dict[str, Any],
    latency_cap_ms: float,
    thermal_cap_c: float,
) -> list[int]:
    if tuple(actions) != ACTIONS:
        raise RuntimeError("guard action order mismatch")
    if len(actions) != len(latency_predictions) or len(actions) != len(thermal_predictions):
        raise RuntimeError("guard vector length mismatch")
    latency_cap_ms = float(latency_cap_ms)
    thermal_cap_c = float(thermal_cap_c)
    latency_buffer = float(margins["fixed_latency_buffer_ms"])
    thermal_residual_margin = float(margins["epsilon_T_C_shared"])
    thermal_buffer = float(margins["fixed_thermal_buffer_c"])
    thermal_margin = thermal_residual_margin + thermal_buffer
    caps_and_margins = [
        latency_cap_ms,
        thermal_cap_c,
        latency_buffer,
        thermal_residual_margin,
        thermal_buffer,
    ]
    if not all(math.isfinite(float(value)) for value in caps_and_margins):
        raise RuntimeError("guard caps and margins must be finite")
    if (
        latency_cap_ms <= 0
        or thermal_cap_c <= 0
        or latency_buffer < 0
        or thermal_residual_margin < 0
        or thermal_buffer < 0
    ):
        raise RuntimeError("guard caps and margins are invalid")
    action_margins = margins.get("epsilon_tau_ms_by_action")
    if not isinstance(action_margins, dict) or set(action_margins) != set(ACTIONS):
        raise RuntimeError("guard action-margin mapping mismatch")
    allowed = []
    for index, action in enumerate(actions):
        latency = (
            latency_predictions[action]
            if isinstance(latency_predictions, dict)
            else latency_predictions[index]
        )
        thermal = (
            thermal_predictions[action]
            if isinstance(thermal_predictions, dict)
            else thermal_predictions[index]
        )
        latency = float(latency)
        thermal = float(thermal)
        latency_margin = float(action_margins[action])
        if (
            not all(math.isfinite(value) for value in (latency, thermal, latency_margin))
            or latency < 0
            or thermal < 0
            or latency_margin < 0
        ):
            raise RuntimeError(f"invalid guard input for {action}")
        latency_upper = latency + latency_margin + latency_buffer
        thermal_upper = thermal + thermal_margin
        if latency_upper <= latency_cap_ms and thermal_upper <= thermal_cap_c:
            allowed.append(index)
    return allowed


def masked_argmax(q_row: Sequence[float], safe_idx: Sequence[int]) -> int:
    """Argmax over admissible indices, with frozen action-order ties."""
    raw_values = np.asarray(q_row)
    if np.iscomplexobj(raw_values):
        raise RuntimeError("invalid Q row or safe-set index")
    values = np.asarray(raw_values, dtype=np.float64)
    candidates = sorted(set(int(index) for index in safe_idx))
    if not candidates:
        raise RuntimeError("masked argmax received an empty safe set")
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or any(index < 0 or index >= len(values) for index in candidates)
    ):
        raise RuntimeError("invalid Q row or safe-set index")
    return max(candidates, key=lambda index: (float(values[index]), -index))


def threshold_action(
    temperature: float,
    backlog: float,
    risk: float,
    previous_action: int | str | None,
    config: dict[str, Any],
) -> str:
    required = {
        "T_hi_C",
        "T_lo_C",
        "q_hi_records",
        "q_lo_records",
        "p_hi",
        "heavy_action",
        "light_action",
    }
    missing = sorted(required - set(config))
    if missing:
        raise RuntimeError(f"threshold controller is not frozen: missing {missing}")
    heavy = str(config["heavy_action"])
    light = str(config["light_action"])
    if heavy not in ACTIONS or light not in ACTIONS:
        raise RuntimeError("threshold controller action mismatch")
    if temperature >= float(config["T_hi_C"]) or backlog >= float(config["q_hi_records"]):
        return light
    if (
        temperature <= float(config["T_lo_C"])
        and backlog <= float(config["q_lo_records"])
        and risk >= float(config["p_hi"])
    ):
        return heavy
    if isinstance(previous_action, int):
        previous_name = None if previous_action == NO_PREVIOUS_ACTION else ACTIONS[previous_action]
    else:
        previous_name = previous_action
    return previous_name if previous_name is not None else light


def reward_components(
    config: dict[str, Any],
    risk: float,
    action_index: int,
    backlog: float,
    previous_action: int,
    quality: dict[str, dict[str, float]],
    energy_cost: Sequence[float],
) -> dict[str, float]:
    """Frozen label-free MDP utility; TP/FP are scoring outputs, not reward inputs."""
    action = ACTIONS[action_index]
    security = risk * float(quality[action]["tpr"]) - (1.0 - risk) * float(
        quality[action]["fpr"]
    )
    energy = -float(config["energy_weight"]) * float(energy_cost[action_index])
    backlog_part = -float(config["backlog_weight"]) * min(
        backlog / float(config["backlog_normalizer"]), 1.0
    )
    switched = previous_action != NO_PREVIOUS_ACTION and previous_action != action_index
    switch = -float(config["switch_weight"]) * float(switched)
    return {
        "security": security,
        "energy": energy,
        "backlog": backlog_part,
        "switch": switch,
        "total": security + energy + backlog_part + switch,
    }


def select_policy_action(
    policy: str,
    *,
    safe_idx: Sequence[int],
    risk: float,
    temperature: float,
    backlog: float,
    threshold_queue_load: float,
    previous_action: int,
    threshold_config: dict[str, Any],
    q_row: Sequence[float] | None,
    reward_config: dict[str, Any],
    quality: dict[str, dict[str, float]],
    energy_cost: Sequence[float],
) -> int:
    if policy == "Static-Light":
        return STATIC_LIGHT_ACTION_INDEX
    if policy == "Threshold":
        return ACTIONS.index(
            threshold_action(
                temperature,
                threshold_queue_load,
                risk,
                previous_action,
                threshold_config,
            )
        )
    candidates = sorted(set(int(index) for index in safe_idx))
    if any(index < 0 or index >= len(ACTIONS) for index in candidates):
        raise RuntimeError("safe set contains an invalid action index")
    candidates = candidates or [STATIC_LIGHT_ACTION_INDEX]
    if policy == "Safe-Greedy":
        return max(
            candidates,
            key=lambda index: (
                reward_components(
                    reward_config,
                    risk,
                    index,
                    backlog,
                    previous_action,
                    quality,
                    energy_cost,
                )["total"],
                -index,
            ),
        )
    if policy == "FSSQL-R":
        if q_row is None:
            raise RuntimeError("FSSQL-R requires a Q row")
        return masked_argmax(q_row, candidates)
    raise RuntimeError(f"unsupported policy: {policy}")


def score_predictions(
    probabilities: Sequence[float], labels: Sequence[int], threshold: float
) -> dict[str, Any]:
    raw_scores = np.asarray(probabilities)
    raw_truth = np.asarray(labels)
    if np.iscomplexobj(raw_scores) or np.iscomplexobj(raw_truth):
        raise RuntimeError("prediction scores and labels must be real-valued")
    scores = np.asarray(raw_scores, dtype=np.float64)
    if scores.shape != raw_truth.shape or scores.ndim != 1:
        raise RuntimeError("prediction/label shape mismatch")
    if not np.issubdtype(raw_truth.dtype, np.number):
        raise RuntimeError("scoring labels are not numeric")
    numeric_truth = np.asarray(raw_truth, dtype=np.float64)
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise RuntimeError("decision threshold must be finite and in [0,1]")
    if not np.isfinite(scores).all() or not np.isfinite(numeric_truth).all():
        raise RuntimeError("non-finite scores or labels")
    if not np.isin(numeric_truth, [0.0, 1.0]).all():
        raise RuntimeError("non-finite scores or non-binary labels")
    truth = numeric_truth.astype(np.int8)
    predictions = scores >= threshold
    return {
        "predictions": predictions.astype(np.int8),
        "tp": int(np.sum(predictions & (truth == 1))),
        "fp": int(np.sum(predictions & (truth == 0))),
        "fn": int(np.sum(~predictions & (truth == 1))),
        "tn": int(np.sum(~predictions & (truth == 0))),
    }


def read_latency_predictor(path: Path) -> dict[tuple[str, str, str], float]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"action", "workload_bin", "background", "tau_hat_ms"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError("latency predictor schema mismatch")
        predictor: dict[tuple[str, str, str], float] = {}
        for row in reader:
            key = (row["action"], row["workload_bin"], row["background"])
            if key in predictor:
                raise RuntimeError(f"duplicate latency predictor cell: {key}")
            value = float(row["tau_hat_ms"])
            if not math.isfinite(value) or value < 0:
                raise RuntimeError(f"invalid latency predictor value for {key}")
            predictor[key] = value
        return predictor


def read_raw_base_rows(path: Path, expected_rows: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        normalized_fieldnames = [field.strip().lower() for field in fieldnames]
        if len(fieldnames) != len(set(fieldnames)) or len(normalized_fieldnames) != len(
            set(normalized_fieldnames)
        ):
            raise RuntimeError("raw runtime input contains duplicate columns")
        fields = dict(zip(normalized_fieldnames, fieldnames))
        forbidden = {
            "id",
            "label",
            "attack_cat",
            "source_id",
            "source_row_index",
            "runtime_sequence",
            "source_feature_hash",
            "feature_group_hex",
            "group_id",
            "duplicate_group",
            "fold",
            "split",
            "row_id",
            "original_row_index",
        }
        leaked = sorted(
            original
            for normalized, original in fields.items()
            if normalized in forbidden or "label" in normalized
        )
        if leaked:
            raise RuntimeError(f"raw runtime input contains scoring/audit columns: {leaked}")
        rows = list(reader)
    if len(rows) != expected_rows:
        raise RuntimeError(f"raw base features require {expected_rows} rows; observed {len(rows)}")
    return rows


def read_raw_rows(path: Path, windows: int, records_per_window: int) -> list[list[dict[str, str]]]:
    """Legacy-shaped helper retained for unit probes; corrected V5 uses base rows + indices."""
    expected = windows * records_per_window
    rows = read_raw_base_rows(path, expected)
    return [
        rows[start : start + records_per_window]
        for start in range(0, expected, records_per_window)
    ]


def validate_background_load(condition: str, workload: dict[str, Any]) -> None:
    """Require the exact Pi4B calibration load, not a B1/B2 label alone."""
    expected = {
        "n16/B0": {"background": "B0", "busy_cpu_count": 0, "background_duty": 0.0},
        "n16/B1": {
            "background": "B1",
            "busy_cpu_count": 1,
            "background_duty": 0.5,
            "pinned_cpu": 3,
            "period_ms": 100.0,
        },
        "n16/B2": {
            "background": "B2",
            "busy_cpu_count": 1,
            "background_duty": 1.0,
            "pinned_cpu": 3,
            "period_ms": 100.0,
        },
    }
    if condition not in expected:
        raise RuntimeError(f"unsupported corrected workload: {condition}")
    for field, value in expected[condition].items():
        if workload.get(field) != value:
            raise RuntimeError(
                f"background-load mismatch for {condition}: {field}={workload.get(field)!r}; "
                f"expected {value!r}"
            )
    stabilization = workload.get("background_stabilization_seconds")
    if not isinstance(stabilization, (int, float)) or not math.isfinite(stabilization):
        raise RuntimeError("background stabilization interval is not frozen")
    if float(stabilization) != 0.5:
        raise RuntimeError("background stabilization must equal the frozen 0.5 seconds")


def validate_runtime_config(config: dict[str, Any], condition: str, policy: str) -> dict[str, Any]:
    if config.get("status") != "FROZEN_BEFORE_CORRECTED_PI4B8_EXECUTION":
        raise RuntimeError("corrected runtime configuration is not frozen")
    if config.get("official_test_access") != "forbidden":
        raise RuntimeError("corrected online execution must not access official test")
    if tuple(config.get("action_order", [])) != ACTIONS:
        raise RuntimeError("action order mismatch")
    if config.get("action_roles", {}).get("light") != ACTIONS[STATIC_LIGHT_ACTION_INDEX]:
        raise RuntimeError("Static-Light action mismatch")
    if config.get("constraints") != {"latency_cap_ms": 40.0, "thermal_cap_c": 82.0}:
        raise RuntimeError("corrected runtime must retain the 40-ms/82-C constraints")
    if float(config.get("decision_window_ms", 0.0)) != 100.0:
        raise RuntimeError("corrected runtime must retain the 100-ms decision window")
    if float(config.get("minimum_frequency_mhz", 0.0)) != 1350.0:
        raise RuntimeError("corrected runtime must retain the 1350-MHz frequency gate")
    if config.get("energy_boundary") != "mac_start_ack_to_end_ack":
        raise RuntimeError("POWER-Z energy boundary is not frozen")
    if config.get("background_load_lifetime") != "before_start_ack_through_end_ack":
        raise RuntimeError("background-load lifetime is not frozen")
    timing = config.get("timing_boundaries", {})
    if timing != {
        "controller": "after_preprocessing_through_action_binding",
        "e2e": "preprocessing_start_through_post_action_telemetry",
        "switching": "resident_model_action_binding_only",
    }:
        raise RuntimeError("controller/E2E/switching timing boundaries are not frozen")
    if policy not in SUPPORTED_POLICIES or policy not in config.get("policies", []):
        raise RuntimeError("policy is outside the corrected protocol")
    workload = config.get("workloads", {}).get(condition)
    if not isinstance(workload, dict):
        raise RuntimeError(f"condition is not frozen: {condition}")
    if int(workload.get("n_records", 0)) != 16:
        raise RuntimeError("corrected Pi4B evaluation requires true n=16 batches")
    validate_background_load(condition, workload)
    if config.get("risk_aggregation") != "mean_of_16_frozen_proxy_probabilities":
        raise RuntimeError("risk aggregation is not frozen")
    if config.get("backlog_source") != "frozen_runtime_input_records":
        raise RuntimeError("backlog source is not frozen")
    if (
        float(config.get("energy_state_initialization", math.nan)) != 1.0
        or config.get("energy_state_reset") != "once_at_run_start"
    ):
        raise RuntimeError("runtime energy-state initialization is not frozen")
    if config.get("threshold_queue_value") != "n_records_plus_queued_backlog":
        raise RuntimeError("Threshold queue-load semantics are not frozen")
    state = config.get("state", {})
    validate_state_definition(state)
    quality = config.get("validation_action_quality")
    if not isinstance(quality, dict) or set(quality) != set(ACTIONS):
        raise RuntimeError("validation-only action quality is not frozen")
    for action in ACTIONS:
        metrics = quality[action]
        if not isinstance(metrics, dict) or not {"tpr", "fpr"}.issubset(metrics):
            raise RuntimeError(f"validation action quality is incomplete for {action}")
        values = (float(metrics["tpr"]), float(metrics["fpr"]))
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise RuntimeError(f"validation action quality is invalid for {action}")
    reward = config.get("reward")
    required_reward = {"energy_weight", "backlog_weight", "switch_weight", "backlog_normalizer"}
    if not isinstance(reward, dict) or not required_reward.issubset(reward):
        raise RuntimeError("reward definition is not frozen")
    if (
        reward.get("security")
        != "risk*validation_TPR - (1-risk)*validation_FPR"
        or reward.get("ground_truth_label_use") != "forbidden"
    ):
        raise RuntimeError("reward security/label semantics are not frozen")
    if any(not math.isfinite(float(reward[name])) for name in required_reward):
        raise RuntimeError("reward definition contains non-finite values")
    if (
        any(float(reward[name]) < 0 for name in ("energy_weight", "backlog_weight", "switch_weight"))
        or float(reward["backlog_normalizer"]) <= 0
    ):
        raise RuntimeError("reward weights or backlog normalizer are invalid")
    costs = config.get("normalized_energy_cost_by_condition", {}).get(condition)
    if not isinstance(costs, dict) or set(costs) != set(ACTIONS):
        raise RuntimeError("condition-specific normalized energy costs are not frozen")
    cost_values = [float(costs[action]) for action in ACTIONS]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in cost_values):
        raise RuntimeError("normalized energy costs must be finite values in [0,1]")
    decrements = config.get("energy_state_decrement_by_condition", {}).get(condition)
    if not isinstance(decrements, dict) or set(decrements) != set(ACTIONS):
        raise RuntimeError("condition-specific energy-state decrements are not frozen")
    decrement_values = [float(decrements[action]) for action in ACTIONS]
    if any(not math.isfinite(value) or value < 0.0 for value in decrement_values):
        raise RuntimeError("energy-state decrements must be finite and nonnegative")
    threshold_config = config.get("threshold_controller", {})
    if threshold_config.get("uses_shield") is not False:
        raise RuntimeError("Threshold must retain its frozen unshielded baseline role")
    numeric_thresholds = {
        name: float(threshold_config.get(name, float("nan")))
        for name in ("T_hi_C", "T_lo_C", "q_hi_records", "q_lo_records", "p_hi")
    }
    if not all(math.isfinite(value) for value in numeric_thresholds.values()):
        raise RuntimeError("Threshold controller contains non-finite values")
    if not 0.0 <= numeric_thresholds["p_hi"] <= 1.0:
        raise RuntimeError("Threshold p_hi must be in [0,1]")
    if not 0.0 <= numeric_thresholds["T_lo_C"] < numeric_thresholds["T_hi_C"] < 82.0:
        raise RuntimeError("Threshold temperature hysteresis is invalid")
    if not 0.0 <= numeric_thresholds["q_lo_records"] < numeric_thresholds["q_hi_records"]:
        raise RuntimeError("Threshold backlog hysteresis is invalid")
    roles = config.get("action_roles", {})
    if (
        threshold_config.get("light_action") != roles.get("light")
        or threshold_config.get("heavy_action") != roles.get("heavy")
        or roles.get("light") == roles.get("heavy")
    ):
        raise RuntimeError("Threshold light/heavy roles are not frozen consistently")
    threshold_action(0.0, 0.0, 0.0, NO_PREVIOUS_ACTION, threshold_config)
    quality_source = config.get("validation_action_quality_source", {})
    if quality_source != {
        "path": "models/VALIDATION_METRICS.csv",
        "sha256": "7b2db09489ebc2c2e02c7604b0a7749a3a218d8d0b6abf547123f963affa1755",
        "derivation": (
            "TPR=recall; validation prevalence is solved from "
            "accuracy/precision/recall; FPR=FP/(1-prevalence)"
        ),
    }:
        raise RuntimeError("validation-quality source lineage is not frozen")
    return workload


def validate_fssql_q_lineage(
    runtime_config: dict[str, Any],
    q_config: dict[str, Any],
    q_config_path: Path,
    q_manifest: dict[str, Any],
    q_table_path: Path,
    seed: int,
    evaluation_condition: str,
) -> None:
    """Reject the legacy single-record Q tables for corrected true-n=16 runs."""
    training = q_config.get("training", {})
    dynamics = q_config.get("development_dynamics", {})
    evaluation = q_config.get("evaluation", {})
    if tuple(q_config.get("actions", [])) != ACTIONS:
        raise RuntimeError("FSSQL-R Q action order differs from runtime")
    if int(training.get("records_per_window", 0)) != 16:
        raise RuntimeError("FSSQL-R Q lineage is not trained for true n=16 windows")
    if (
        q_config.get("status") != "FROZEN_BEFORE_PI4B8_TRUE_N16_Q_TRAINING"
        or int(training.get("base_records", 0)) != 6000
        or training.get("condition") != "n16/B0"
        or training.get("window_order") != "seed_specific_frozen_batch_indices"
        or training.get("batch_index_rule")
        != "order=default_rng(seed).permutation(6000);index[w,j]=order[(16*w+j)%6000]"
        or training.get("ground_truth_label_access") != "forbidden"
    ):
        raise RuntimeError("FSSQL-R true-n16 training protocol lineage is incomplete")
    expected_batch_input = f"scheduler_batch_indices_seed_{seed}"
    if training.get("batch_index_inputs", {}).get(str(seed)) != expected_batch_input:
        raise RuntimeError("FSSQL-R Q seed-specific batch-index binding is missing")
    state = q_config.get("state", {})
    try:
        validate_state_definition(state)
    except RuntimeError as error:
        raise RuntimeError("FSSQL-R Q state encoding differs from runtime") from error
    if state != runtime_config.get("state"):
        raise RuntimeError("FSSQL-R Q state bins differ from runtime")
    if training.get("risk_aggregation") != runtime_config.get("risk_aggregation"):
        raise RuntimeError("FSSQL-R Q risk aggregation differs from runtime")
    if dynamics.get("backlog_source") != runtime_config.get("backlog_source"):
        raise RuntimeError("FSSQL-R Q backlog semantics differ from runtime")
    if dynamics.get("backlog_scope") != "frozen_zero_backlog_no_adaptation_claim":
        raise RuntimeError("FSSQL-R Q was not frozen as a zero-backlog control")
    if q_config.get("reward") != runtime_config.get("reward"):
        raise RuntimeError("FSSQL-R Q reward differs from runtime")
    if dynamics.get("normalized_energy_cost_by_condition") != runtime_config.get(
        "normalized_energy_cost_by_condition"
    ):
        raise RuntimeError("FSSQL-R Q energy-cost semantics differ from runtime")
    if dynamics.get("energy_state_decrement_by_condition") != runtime_config.get(
        "energy_state_decrement_by_condition"
    ):
        raise RuntimeError("FSSQL-R Q energy-state dynamics differ from runtime")
    if training.get("official_test_access") != "forbidden":
        raise RuntimeError("FSSQL-R Q lineage permits official-test access")
    if (
        float(evaluation.get("epsilon", float("nan"))) != 0.0
        or evaluation.get("q_updates") is not False
        or evaluation.get("replay_updates") is not False
    ):
        raise RuntimeError("FSSQL-R evaluation must use epsilon=0 with no updates")
    if evaluation_condition not in evaluation.get("conditions", []):
        raise RuntimeError("FSSQL-R evaluation condition is outside the frozen Q scope")
    if evaluation.get("training_to_evaluation_seed_map", {}).get(str(seed)) != seed:
        raise RuntimeError("FSSQL-R Q seed mapping is not identity")
    if q_manifest.get("status") != "PI4B8_TRUE_N16_Q_TRAINING_COMPLETE":
        raise RuntimeError("FSSQL-R Q manifest is not a corrected true-n16 artifact")
    if (
        q_manifest.get("training_condition") != "n16/B0"
        or int(q_manifest.get("records_per_window", 0)) != 16
        or q_manifest.get("risk_aggregation")
        != "mean_of_16_frozen_proxy_probabilities"
    ):
        raise RuntimeError("FSSQL-R Q manifest training semantics are incomplete")
    if q_manifest.get("worker_semantics_sha256") != sha256(Path(__file__)):
        raise RuntimeError("FSSQL-R Q worker-semantics hash mismatch")
    if (
        q_manifest.get("official_test_access") != "not used"
        or q_manifest.get("ground_truth_label_access") != "not used"
    ):
        raise RuntimeError("FSSQL-R Q manifest does not preserve label isolation")
    if q_manifest.get("config_sha256") != sha256(q_config_path):
        raise RuntimeError("FSSQL-R Q configuration hash mismatch")
    expected_inputs = {
        name: record.get("sha256")
        for name, record in q_config.get("inputs", {}).items()
        if isinstance(record, dict)
    }
    if q_manifest.get("input_sha256") != expected_inputs:
        raise RuntimeError("FSSQL-R Q input dependency hashes differ from its protocol")
    matching = [
        row
        for row in q_manifest.get("runs", [])
        if row.get("policy") == "FSSQL-R" and int(row.get("seed", -1)) == seed
    ]
    if len(matching) != 1:
        raise RuntimeError(f"FSSQL-R Q manifest has no unique table for seed {seed}")
    if matching[0].get("table_sha256") != sha256(q_table_path):
        raise RuntimeError(f"FSSQL-R Q-table hash mismatch for seed {seed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--q-config", type=Path)
    parser.add_argument("--q-manifest", type=Path)
    parser.add_argument("--model-hyperparameters", type=Path, required=True)
    parser.add_argument("--raw-features", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--risk-features", type=Path, required=True)
    parser.add_argument("--backlog", type=Path, required=True)
    parser.add_argument("--batch-indices", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--q-table", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_document = validate_worker_cli_binding(
        args.asset_root,
        args.package_manifest,
        policy=args.policy,
        condition=args.condition,
        seed=args.seed,
        supplied={
            "runtime_config": args.runtime_config,
            "model_hyperparameters": args.model_hyperparameters,
            "raw_features": args.raw_features,
            "features": args.features,
            "risk_features": args.risk_features,
            "backlog": args.backlog,
            "batch_indices": args.batch_indices,
            "labels": args.labels,
            "q_config": args.q_config,
            "q_manifest": args.q_manifest,
            "q_table": args.q_table,
        },
    )
    package_manifest_digest = package_sha256(args.package_manifest)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)

    runtime_config = json.loads(args.runtime_config.read_text())
    workload = validate_runtime_config(runtime_config, args.condition, args.policy)
    q_config = json.loads(args.q_config.read_text()) if args.q_config else None
    q_manifest = json.loads(args.q_manifest.read_text()) if args.q_manifest else None
    n_records = int(workload["n_records"])
    background = str(workload["background"])
    background_duty = float(workload["background_duty"])

    base_records = int(package_document["base_records"])
    raw_base = read_raw_base_rows(args.raw_features, expected_rows=base_records)
    frozen_feature_base = np.load(args.features, allow_pickle=False)
    risk_feature_base = np.load(args.risk_features, allow_pickle=False)
    if (
        frozen_feature_base.ndim != 2
        or risk_feature_base.ndim != 2
        or frozen_feature_base.shape[0] != base_records
        or risk_feature_base.shape[0] != base_records
    ):
        raise RuntimeError(
            f"corrected base feature arrays must each contain {base_records} rows"
        )
    if np.iscomplexobj(frozen_feature_base) or np.iscomplexobj(risk_feature_base):
        raise RuntimeError("runtime feature arrays must be real-valued")
    try:
        frozen_features_numeric = np.asarray(frozen_feature_base, dtype=np.float64)
        risk_features_numeric = np.asarray(risk_feature_base, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RuntimeError("runtime feature arrays must be numeric") from error
    if not np.isfinite(frozen_features_numeric).all() or not np.isfinite(
        risk_features_numeric
    ).all():
        raise RuntimeError("runtime feature arrays must be finite")
    batch_indices = validate_batch_indices(
        np.load(args.batch_indices, allow_pickle=False),
        windows=args.windows,
        records_per_window=n_records,
        base_rows=len(raw_base),
        seed=args.seed,
    )
    raw_batches = [[raw_base[int(index)] for index in row] for row in batch_indices]
    frozen_features = frozen_feature_base[batch_indices]
    risk_features = risk_feature_base[batch_indices]
    backlog_values = np.load(args.backlog, allow_pickle=False)
    if (
        backlog_values.shape != (args.windows,)
        or not np.issubdtype(backlog_values.dtype, np.number)
        or np.iscomplexobj(backlog_values)
        or not np.isfinite(backlog_values).all()
        or np.any(backlog_values < 0)
    ):
        raise RuntimeError("backlog input must contain one finite nonnegative value per window")
    if args.labels.resolve() in {
        args.features.resolve(),
        args.risk_features.resolve(),
        args.backlog.resolve(),
        args.batch_indices.resolve(),
        args.raw_features.resolve(),
    }:
        raise RuntimeError("scoring labels must be a separate sidecar")
    if any(
        os.path.samefile(args.labels, path)
        for path in (
            args.features,
            args.risk_features,
            args.backlog,
            args.batch_indices,
            args.raw_features,
        )
    ):
        raise RuntimeError("scoring labels must not alias a runtime decision input")

    preprocessor = PortablePreprocessor(args.asset_root / "preprocessing/preprocessor_portable.npz")
    risk_proxy = RiskProxy(args.asset_root / "risk_proxy/risk_proxy.npz")
    thresholds = load_model_thresholds(args.model_hyperparameters)
    models = {
        action: NumpyActionModel(
            args.asset_root / "models" / f"{action}.npz",
            decision_threshold=thresholds[action],
        )
        for action in ACTIONS
    }
    latency = read_latency_predictor(args.asset_root / "guard/latency_predictor.csv")
    thermal = json.loads((args.asset_root / "guard/thermal_predictor.json").read_text())
    margins = json.loads((args.asset_root / "guard/guard_margins.json").read_text())
    workload_bin = f"n{n_records}"
    try:
        _ = [latency[action, workload_bin, background] for action in ACTIONS]
    except KeyError as error:
        raise RuntimeError(f"guard has no exact {workload_bin}/{background} cell") from error

    quality_path = args.asset_root / "config/VALIDATION_METRICS.csv"
    quality_source = runtime_config["validation_action_quality_source"]
    if sha256(quality_path) != quality_source["sha256"]:
        raise RuntimeError("validation-quality source hash mismatch")
    quality = load_validation_action_quality(quality_path)
    configured_quality = runtime_config["validation_action_quality"]
    if any(
        not math.isclose(
            float(quality[action][metric]),
            float(configured_quality[action][metric]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for action in ACTIONS
        for metric in ("tpr", "fpr", "f1")
    ):
        raise RuntimeError("validation-quality values differ from their frozen source")
    reward_config = runtime_config["reward"]
    energy_cost = np.asarray(
        [
            float(runtime_config["normalized_energy_cost_by_condition"][args.condition][action])
            for action in ACTIONS
        ],
        dtype=np.float64,
    )
    energy_decrement = np.asarray(
        [
            float(runtime_config["energy_state_decrement_by_condition"][args.condition][action])
            for action in ACTIONS
        ],
        dtype=np.float64,
    )

    q_values = None
    if args.policy == "FSSQL-R":
        if args.q_table is None or args.q_config is None or args.q_manifest is None:
            raise RuntimeError("FSSQL-R requires --q-table, --q-config and --q-manifest")
        assert q_config is not None and q_manifest is not None
        validate_fssql_q_lineage(
            runtime_config,
            q_config,
            args.q_config,
            q_manifest,
            args.q_table,
            args.seed,
            args.condition,
        )
        q_input_hashes = q_manifest["input_sha256"]
        q_runtime_dependencies = {
            "runtime_config": args.runtime_config,
            "risk_proxy": args.asset_root / "risk_proxy/risk_proxy.npz",
            "validation_metrics": args.asset_root / "config/VALIDATION_METRICS.csv",
            "latency_predictor": args.asset_root / "guard/latency_predictor.csv",
            "thermal_predictor": args.asset_root / "guard/thermal_predictor.json",
            "guard_margins": args.asset_root / "guard/guard_margins.json",
        }
        mismatched_dependencies = [
            name
            for name, path in q_runtime_dependencies.items()
            if q_input_hashes.get(name) != sha256(path)
        ]
        if mismatched_dependencies:
            raise RuntimeError(
                "FSSQL-R Q/runtime dependency hash mismatch: "
                f"{mismatched_dependencies}"
            )
        with np.load(args.q_table, allow_pickle=False) as q_artifact:
            required = {
                "q_values",
                "actions",
                "training_seed",
                "shielded",
                "records_per_window",
                "training_condition",
                "config_sha256",
                "runtime_config_sha256",
                "risk_features_sha256",
                "batch_indices_sha256",
                "backlog_sha256",
            }
            if not required.issubset(q_artifact.files):
                raise RuntimeError("FSSQL-R Q table is missing embedded lineage metadata")
            if tuple(q_artifact["actions"].tolist()) != ACTIONS:
                raise RuntimeError("FSSQL-R Q-table action order mismatch")
            if int(q_artifact["training_seed"]) != args.seed:
                raise RuntimeError("FSSQL-R Q-table seed mismatch")
            if not bool(q_artifact["shielded"]):
                raise RuntimeError("FSSQL-R received an unshielded Q table")
            if int(q_artifact["records_per_window"]) != 16:
                raise RuntimeError("FSSQL-R Q table is not a true-n16 artifact")
            if str(q_artifact["training_condition"]) != "n16/B0":
                raise RuntimeError("FSSQL-R Q table training condition mismatch")
            input_hashes = q_manifest["input_sha256"]
            embedded_hashes = {
                "config_sha256": (str(q_artifact["config_sha256"]), sha256(args.q_config)),
                "runtime_config_sha256": (
                    str(q_artifact["runtime_config_sha256"]),
                    input_hashes["runtime_config"],
                ),
                "risk_features_sha256": (
                    str(q_artifact["risk_features_sha256"]),
                    input_hashes["scheduler_risk_features_base"],
                ),
                "batch_indices_sha256": (
                    str(q_artifact["batch_indices_sha256"]),
                    input_hashes[f"scheduler_batch_indices_seed_{args.seed}"],
                ),
                "backlog_sha256": (
                    str(q_artifact["backlog_sha256"]),
                    input_hashes["scheduler_backlog"],
                ),
            }
            mismatched = [
                name for name, (observed, expected) in embedded_hashes.items()
                if observed != expected
            ]
            if mismatched:
                raise RuntimeError(f"FSSQL-R embedded lineage hash mismatch: {mismatched}")
            q_values = q_artifact["q_values"].copy()
        if np.any(backlog_values != 0):
            raise RuntimeError(
                "FSSQL-R true-n16 Q is a zero-backlog control and cannot claim "
                "backlog adaptation"
            )
        if q_values.shape != (int(runtime_config["state"]["number_of_states"]), len(ACTIONS)):
            raise RuntimeError(f"Q-table shape mismatch: {q_values.shape}")
        if (
            not np.issubdtype(q_values.dtype, np.number)
            or np.iscomplexobj(q_values)
            or not np.isfinite(q_values).all()
        ):
            raise RuntimeError("Q table must contain only finite numeric values")

    background_load = BackgroundLoad(workload)
    background_load.start()
    atexit.register(background_load.stop)
    time.sleep(float(workload["background_stabilization_seconds"]))
    if int(workload["busy_cpu_count"]) and not background_load.alive:
        raise RuntimeError("background-load process stopped before RUN_START_ACK")

    run_start_utc = time.time()
    run_start_monotonic = time.monotonic()
    print(
        json.dumps(
            {
                "event": "RUN_START_ACK",
                "run_id": args.run_id,
                "pi_utc": run_start_utc,
                "pi_monotonic": run_start_monotonic,
                "pid": os.getpid(),
                "runtime_config_sha256": sha256(args.runtime_config),
                "package_manifest_sha256": package_manifest_digest,
                "source_manifest_sha256": package_document["source_manifest_sha256"],
                "threshold_source_sha256": sha256(args.model_hyperparameters),
                "q_table_sha256": "" if args.q_table is None else sha256(args.q_table),
                "records_per_window": n_records,
                "background": background,
                "background_load_pid": background_load.pid,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    probabilities_by_window: list[np.ndarray] = []
    start = run_start_monotonic
    previous_action = NO_PREVIOUS_ACTION
    energy = float(runtime_config["energy_state_initialization"])
    for window_id in range(args.windows):
        scheduled = start + window_id * float(runtime_config["decision_window_ms"]) / 1000.0
        remaining = scheduled - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        actual = time.monotonic()
        e2e_start_ns = time.perf_counter_ns()
        if int(workload["busy_cpu_count"]) and not background_load.alive:
            raise RuntimeError(f"background-load process stopped at window {window_id}")

        step = time.perf_counter_ns()
        transformed = preprocessor.transform(raw_batches[window_id])
        preprocessing_ms = (time.perf_counter_ns() - step) / 1e6
        if (
            not np.issubdtype(transformed.dtype, np.number)
            or np.iscomplexobj(transformed)
            or not np.isfinite(transformed).all()
        ):
            raise RuntimeError(f"preprocessor produced non-finite values at window {window_id}")
        if not np.array_equal(transformed, frozen_features[window_id]):
            raise RuntimeError(f"preprocessing parity failure at window {window_id}")

        controller_start_ns = time.perf_counter_ns()
        step = time.perf_counter_ns()
        proxy_values = risk_proxy.predict(risk_features[window_id])
        risk_proxy_ms = (time.perf_counter_ns() - step) / 1e6
        if (
            proxy_values.shape != (n_records,)
            or np.iscomplexobj(proxy_values)
            or not np.isfinite(proxy_values).all()
            or np.any(proxy_values < 0.0)
            or np.any(proxy_values > 1.0)
        ):
            raise RuntimeError("risk proxy output mismatch")
        risk = float(np.mean(proxy_values))

        step = time.perf_counter_ns()
        current_temperature = temperature_c()
        frequencies = frequencies_mhz()
        throttle_text, throttle_value = throttled()
        telemetry_ms = (time.perf_counter_ns() - step) / 1e6
        if (
            not math.isfinite(current_temperature)
            or current_temperature < 0.0
            or not frequencies
            or not all(math.isfinite(value) and value > 0.0 for value in frequencies)
        ):
            raise RuntimeError("pre-action hardware telemetry is invalid")
        if throttle_value & 0xF:
            raise RuntimeError(f"current throttle/undervoltage fault: {throttle_text}")
        if min(frequencies) < float(runtime_config["minimum_frequency_mhz"]):
            raise RuntimeError("CPU frequency invariant failed")

        backlog = float(backlog_values[window_id])
        threshold_queue_load = float(n_records) + backlog
        step = time.perf_counter_ns()
        state = state_id(
            runtime_config, risk, current_temperature, energy, backlog, previous_action
        )
        state_ms = (time.perf_counter_ns() - step) / 1e6

        guard_start = time.perf_counter_ns()
        step = time.perf_counter_ns()
        latency_predictions = np.asarray(
            [latency[action, workload_bin, background] for action in ACTIONS],
            dtype=np.float64,
        )
        latency_guard_ms = (time.perf_counter_ns() - step) / 1e6
        step = time.perf_counter_ns()
        thermal_predictions = np.asarray(
            [
                thermal_prediction(
                    thermal,
                    action,
                    current_temperature,
                    float(latency_predictions[index]),
                    n_records,
                    background_duty,
                )
                for index, action in enumerate(ACTIONS)
            ]
        )
        thermal_guard_ms = (time.perf_counter_ns() - step) / 1e6
        step = time.perf_counter_ns()
        safe_idx = build_safe_set(
            ACTIONS,
            latency_predictions,
            thermal_predictions,
            margins,
            float(runtime_config["constraints"]["latency_cap_ms"]),
            float(runtime_config["constraints"]["thermal_cap_c"]),
        )
        admissible_set_ms = (time.perf_counter_ns() - step) / 1e6
        guard_total_ms = (time.perf_counter_ns() - guard_start) / 1e6

        q_lookup_ms = 0.0
        q_row = None
        if args.policy == "FSSQL-R":
            step = time.perf_counter_ns()
            q_row = q_values[state].copy()
            q_lookup_ms = (time.perf_counter_ns() - step) / 1e6
        step = time.perf_counter_ns()
        action_index = select_policy_action(
            args.policy,
            safe_idx=safe_idx,
            risk=risk,
            temperature=current_temperature,
            backlog=backlog,
            threshold_queue_load=threshold_queue_load,
            previous_action=previous_action,
            threshold_config=runtime_config["threshold_controller"],
            q_row=q_row,
            reward_config=reward_config,
            quality=quality,
            energy_cost=energy_cost,
        )
        policy_selection_ms = (time.perf_counter_ns() - step) / 1e6

        switched = previous_action != NO_PREVIOUS_ACTION and previous_action != action_index
        step = time.perf_counter_ns()
        selected_action = ACTIONS[action_index]
        selected_model = models[selected_action]
        model_binding_ms = (time.perf_counter_ns() - step) / 1e6
        switching_overhead_ms = model_binding_ms if switched else 0.0
        controller_ms = (time.perf_counter_ns() - controller_start_ns) / 1e6

        step = time.perf_counter_ns()
        probabilities = selected_model.predict_proba(transformed)
        detector_inference_ms = (time.perf_counter_ns() - step) / 1e6
        if (
            probabilities.shape != (n_records,)
            or np.iscomplexobj(probabilities)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
        ):
            raise RuntimeError("detector output does not contain 16 finite scores")

        step = time.perf_counter_ns()
        end_temperature = temperature_c()
        end_frequencies = frequencies_mhz()
        end_throttle_text, end_throttle_value = throttled()
        post_action_telemetry_ms = (time.perf_counter_ns() - step) / 1e6
        if (
            not math.isfinite(end_temperature)
            or end_temperature < 0.0
            or not end_frequencies
            or not all(math.isfinite(value) and value > 0.0 for value in end_frequencies)
        ):
            raise RuntimeError("post-action hardware telemetry is invalid")
        if end_throttle_value & 0xF:
            raise RuntimeError(
                f"post-action throttle/undervoltage fault: {end_throttle_text}"
            )
        if min(end_frequencies) < float(runtime_config["minimum_frequency_mhz"]):
            raise RuntimeError("post-action CPU frequency invariant failed")
        e2e_ms = (time.perf_counter_ns() - e2e_start_ns) / 1e6

        reward = reward_components(
            reward_config,
            risk,
            action_index,
            backlog,
            previous_action,
            quality,
            energy_cost,
        )
        latency_upper = np.asarray(
            [
                latency_predictions[index]
                + float(margins["epsilon_tau_ms_by_action"][action])
                + float(margins["fixed_latency_buffer_ms"])
                for index, action in enumerate(ACTIONS)
            ]
        )
        thermal_upper = thermal_predictions + float(margins["epsilon_T_C_shared"]) + float(
            margins["fixed_thermal_buffer_c"]
        )
        rows.append(
            {
                "run_id": args.run_id,
                "window_id": window_id,
                "scheduled_start_monotonic": scheduled,
                "actual_start_monotonic": actual,
                "deadline_jitter_ms": (actual - scheduled) * 1000.0,
                "policy": args.policy,
                "condition": args.condition,
                "seed": args.seed,
                "records_processed": n_records,
                "selected_action": selected_action,
                "previous_action": "" if previous_action == NO_PREVIOUS_ACTION else ACTIONS[previous_action],
                "switched": int(switched),
                "safe_set_mask": "".join("1" if index in safe_idx else "0" for index in range(len(ACTIONS))),
                "safe_set_size": len(safe_idx),
                "fallback": int(not safe_idx and args.policy in {"FSSQL-R", "Safe-Greedy"}),
                "predicted_latency_by_action_ms": json.dumps(
                    dict(zip(ACTIONS, latency_predictions.tolist()))
                ),
                "latency_headroom_by_action_ms": json.dumps(
                    {
                        action: float(runtime_config["constraints"]["latency_cap_ms"] - latency_upper[index])
                        for index, action in enumerate(ACTIONS)
                    }
                ),
                "predicted_thermal_by_action_C": json.dumps(dict(zip(ACTIONS, thermal_predictions.tolist()))),
                "thermal_headroom_by_action_C": json.dumps(
                    {
                        action: float(runtime_config["constraints"]["thermal_cap_c"] - thermal_upper[index])
                        for index, action in enumerate(ACTIONS)
                    }
                ),
                "preprocessing_ms": preprocessing_ms,
                "risk_proxy_ms": risk_proxy_ms,
                "telemetry_ms": telemetry_ms,
                "state_construction_ms": state_ms,
                "latency_guard_ms": latency_guard_ms,
                "thermal_guard_ms": thermal_guard_ms,
                "admissible_set_ms": admissible_set_ms,
                "guard_total_ms": guard_total_ms,
                "q_lookup_ms": q_lookup_ms,
                "policy_selection_ms": policy_selection_ms,
                "switching_overhead_ms": switching_overhead_ms,
                "detector_inference_ms": detector_inference_ms,
                "post_action_telemetry_ms": post_action_telemetry_ms,
                "controller_ms": controller_ms,
                "e2e_ms": e2e_ms,
                "scoring_ms": "",
                "temperature_start_C": current_temperature,
                "temperature_end_C": end_temperature,
                "temperature_C": max(current_temperature, end_temperature),
                "cpu_frequency_start_mhz_min": min(frequencies),
                "cpu_frequency_end_mhz_min": min(end_frequencies),
                "cpu_frequency_mhz_min": min(frequencies + end_frequencies),
                "throttled_start": throttle_text,
                "throttled_end": end_throttle_text,
                "throttled": end_throttle_text,
                "undervoltage": int(bool((throttle_value | end_throttle_value) & 1)),
                "risk": risk,
                "backlog_records": backlog,
                "threshold_queue_load_records": threshold_queue_load,
                "background_load_pid": "" if background_load.pid is None else background_load.pid,
                "background_load_alive": int(
                    int(workload["busy_cpu_count"]) == 0 or background_load.alive
                ),
                "background_pinned_cpu": workload.get("pinned_cpu", ""),
                "background_duty": background_duty,
                "background_period_ms": workload.get("period_ms", ""),
                "prediction_scores": json.dumps(probabilities.tolist()),
                "prediction_bits": "",
                "labels_scoring_only": "",
                "tp": "",
                "fp": "",
                "fn": "",
                "tn": "",
                "reward": reward["total"],
                "reward_security": reward["security"],
                "reward_energy": reward["energy"],
                "reward_backlog": reward["backlog"],
                "reward_switch": reward["switch"],
                "q_update_ms": 0.0,
                "replay_update_ms": 0.0,
            }
        )
        probabilities_by_window.append(np.asarray(probabilities, dtype=np.float64))
        previous_action = action_index
        energy = max(0.0, energy - float(energy_decrement[action_index]))

    decision_loop_end_pi_monotonic = time.monotonic()
    # Label sidecar access is deliberately after every action and prediction.
    scoring_start_pi_monotonic = time.monotonic()
    label_base = np.load(args.labels, allow_pickle=False)
    if (
        label_base.shape != (base_records,)
        or not np.issubdtype(label_base.dtype, np.number)
        or np.iscomplexobj(label_base)
        or not np.isfinite(label_base).all()
        or not np.isin(label_base, [0, 1]).all()
    ):
        raise RuntimeError(
            f"scoring label base must contain {base_records} binary labels"
        )
    labels = label_base[batch_indices]
    for row, probabilities, truth in zip(rows, probabilities_by_window, labels):
        step = time.perf_counter_ns()
        result = score_predictions(probabilities, truth, thresholds[row["selected_action"]])
        truth_array = np.asarray(truth, dtype=np.int8)
        row["prediction_bits"] = json.dumps(result["predictions"].tolist())
        row["labels_scoring_only"] = json.dumps(truth_array.tolist())
        for name in ("tp", "fp", "fn", "tn"):
            row[name] = result[name]
        row["scoring_ms"] = (time.perf_counter_ns() - step) / 1e6
    scoring_end_pi_monotonic = time.monotonic()

    output_path = args.output / "windows.csv"
    with output_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    digest = sha256(output_path)
    evidence_write_end_pi_monotonic = time.monotonic()
    if int(workload["busy_cpu_count"]) and not background_load.alive:
        raise RuntimeError("background-load process stopped before the final window")
    run_end_utc = time.time()
    run_end_monotonic = time.monotonic()
    print(
        json.dumps(
            {
                "event": "RUN_END_ACK",
                "run_id": args.run_id,
                "pi_utc": run_end_utc,
                "pi_monotonic": run_end_monotonic,
                "decision_loop_end_pi_monotonic": decision_loop_end_pi_monotonic,
                "scoring_start_pi_monotonic": scoring_start_pi_monotonic,
                "scoring_end_pi_monotonic": scoring_end_pi_monotonic,
                "evidence_write_end_pi_monotonic": evidence_write_end_pi_monotonic,
                "windows_completed": len(rows),
                "records_completed": len(rows) * n_records,
                "raw_artifact_path": str(output_path),
                "raw_artifact_sha256": digest,
                "package_manifest_sha256": package_manifest_digest,
                "validity_status": "PASS",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    background_load.stop()


if __name__ == "__main__":
    main()
