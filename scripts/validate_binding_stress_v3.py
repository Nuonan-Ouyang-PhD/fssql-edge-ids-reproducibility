#!/usr/bin/env python3
"""Independently validate the accepted Protocol V3 binding-stress sweep."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = (
    ROOT
    / "online/pi3b/binding_stress_v3/"
    "accepted_attempt_binding_stress_v3_attempt2_20260909T224600Z"
)
CONFIG_PATH = ROOT / "configs/pi3b_binding_stress_protocol_v3.json"
VALIDATION_PATH = CAMPAIGN / "INDEPENDENT_VALIDATION.json"
FROZEN_COMMIT = "2386ba7fa90b2641140b3ad7ff8a79b8aeb9d0a6"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(left: float, right: float, tolerance: float = 2e-9) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def throttle_value(text: str) -> int:
    match = re.search(r"0x[0-9a-fA-F]+", text)
    require(match is not None, f"unparseable throttle value: {text}")
    return int(match.group(0), 16)


def guarded_temperature(
    thermal: dict[str, object],
    margins: dict[str, object],
    action: str,
    temperature: float,
    n_records: int,
    background_duty: float,
    latency_hat: float,
) -> float:
    numeric = [temperature, math.log2(n_records), background_duty, latency_hat]
    scaled = [
        (value - mean) / scale
        for value, mean, scale in zip(
            numeric, thermal["continuous_mean"], thermal["continuous_scale"]
        )
    ]
    indicators = [
        float(action == "lucid_revision"),
        float(action == "tinydl_revision"),
        float(action == "oi_svdd_as_elm_revision"),
    ]
    design = [1.0, *scaled, *indicators]
    prediction = sum(
        value * coefficient
        for value, coefficient in zip(design, thermal["coefficients"])
    )
    return (
        prediction
        + float(margins["epsilon_T_C_shared"])
        + float(margins["fixed_thermal_buffer_c"])
    )


config = json.loads(CONFIG_PATH.read_text())
summary_path = CAMPAIGN / "SWEEP_SUMMARY.json"
summary = json.loads(summary_path.read_text())
progress_rows = list(
    csv.DictReader((CAMPAIGN / "SWEEP_PROGRESS.csv").open(newline="", encoding="utf-8"))
)
thermal = json.loads(
    (ROOT / config["inputs"]["thermal_predictor"]["path"]).read_text()
)
margins = json.loads(
    (ROOT / config["inputs"]["guard_margins"]["path"]).read_text()
)
with (ROOT / config["inputs"]["latency_predictor"]["path"]).open(
    newline="", encoding="utf-8"
) as stream:
    latency = {
        (row["action"], row["workload_bin"], row["background"]): float(
            row["tau_hat_ms"]
        )
        for row in csv.DictReader(stream)
    }

require(config["status"] == "FROZEN_BEFORE_BINDING_STRESS_SWEEP", "config not frozen")
require(digest(CONFIG_PATH) == summary["config_sha256"], "config hash mismatch")
require(summary["git_commit"] == FROZEN_COMMIT, "source commit mismatch")
require(summary["status"] == "BINDING_STRESS_POINT_SELECTED", "unexpected sweep status")
require(not (CAMPAIGN / "SWEEP_STOP.json").exists(), "stop marker exists")
require(len(progress_rows) == summary["candidates_evaluated"] == 4, "candidate count mismatch")

actions = list(config["actions"])
background_duty = {
    row["id"]: float(row["duty_cycle"])
    for row in config["background_conditions"]
}
total_expected = len(config["diagnostic_seeds"]) * int(config["windows_per_seed"])
candidate_results = []
artifact_hashes_reproduced = 0
total_windows = 0
total_idle_samples = 0
total_fault_samples = 0
global_max_temperature = -math.inf
global_min_frequency = math.inf

for offset, progress in enumerate(progress_rows):
    planned = config["candidate_grid"][offset]
    index = int(planned["candidate_index"])
    name = f"candidate_{index:02d}_n{planned['n_records']}_{planned['background']}"
    candidate_dir = CAMPAIGN / name
    require(progress == {
        "candidate_index": str(index),
        "n_records": str(planned["n_records"]),
        "background": planned["background"],
        "status": "BINDING_STRESS_CANDIDATE_VALID",
        "qualified": progress["qualified"],
        "output": name,
    }, f"progress/order mismatch: {name}")
    require(
        {path.name for path in candidate_dir.iterdir()} == {
            "CANDIDATE_MANIFEST.json",
            "diagnostic_windows.csv",
            "fault_telemetry_1hz.csv",
            "idle_baseline.csv",
            "seed_progress.csv",
        },
        f"unexpected candidate artifacts: {name}",
    )

    manifest_path = candidate_dir / "CANDIDATE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    require(
        digest(manifest_path) == summary["candidate_manifest_sha256"][name],
        f"candidate manifest hash mismatch: {name}",
    )
    require(manifest["status"] == "BINDING_STRESS_CANDIDATE_VALID", f"invalid candidate: {name}")
    require(not manifest["failure_reason"] and manifest["formal_diagnostic"] is True, f"candidate status fields: {name}")
    require(
        manifest["candidate_index"] == index
        and manifest["n_records"] == planned["n_records"]
        and manifest["background"] == planned["background"],
        f"candidate identity mismatch: {name}",
    )
    require(
        manifest["windows_expected"] == manifest["windows_observed"] == total_expected,
        f"manifest window count mismatch: {name}",
    )
    for artifact, expected_hash in manifest["artifact_sha256"].items():
        require(digest(candidate_dir / artifact) == expected_hash, f"artifact hash mismatch: {name}/{artifact}")
        artifact_hashes_reproduced += 1

    idle_rows = list(
        csv.DictReader((candidate_dir / "idle_baseline.csv").open(newline="", encoding="utf-8"))
    )
    require(len(idle_rows) == config["idle_baseline"]["samples"] == 30, f"idle count mismatch: {name}")
    require(all(throttle_value(row["throttled"]) & 0xF == 0 for row in idle_rows), f"idle hardware fault: {name}")
    total_idle_samples += len(idle_rows)

    fault_rows = list(
        csv.DictReader((candidate_dir / "fault_telemetry_1hz.csv").open(newline="", encoding="utf-8"))
    )
    require(fault_rows, f"empty fault telemetry: {name}")
    require(all(throttle_value(row["throttled"]) & 0xF == 0 for row in fault_rows), f"fault monitor bit: {name}")
    require(
        all(float(row["temperature_c"]) < config["constraints"]["emergency_stop_c"] for row in fault_rows),
        f"fault monitor temperature stop: {name}",
    )
    total_fault_samples += len(fault_rows)

    seed_rows = list(
        csv.DictReader((candidate_dir / "seed_progress.csv").open(newline="", encoding="utf-8"))
    )
    require([int(row["seed"]) for row in seed_rows] == config["diagnostic_seeds"], f"seed order mismatch: {name}")
    require(
        all(
            int(row["windows_expected"]) == int(row["windows_observed"]) == config["windows_per_seed"]
            for row in seed_rows
        ),
        f"seed completeness mismatch: {name}",
    )

    rows = list(
        csv.DictReader((candidate_dir / "diagnostic_windows.csv").open(newline="", encoding="utf-8"))
    )
    require(len(rows) == total_expected, f"window count mismatch: {name}")
    require(
        float(rows[0]["T_start_C"])
        <= statistics.median(float(row["temperature_C"]) for row in idle_rows)
        + config["idle_baseline"]["start_limit_c_above_baseline"],
        f"cooldown start rule mismatch: {name}",
    )

    safe_counts: Counter[int] = Counter()
    admissible_counts: Counter[str] = Counter()
    fallback_count = 0
    candidate_max_temperature = -math.inf
    candidate_min_frequency = math.inf
    for row_number, row in enumerate(rows):
        seed_offset, window_id = divmod(row_number, int(config["windows_per_seed"]))
        require(int(row["candidate_index"]) == index, f"window candidate mismatch: {name}/{row_number}")
        require(int(row["seed"]) == config["diagnostic_seeds"][seed_offset], f"window seed mismatch: {name}/{row_number}")
        require(int(row["window_id"]) == window_id, f"window order mismatch: {name}/{row_number}")
        require(int(row["n_records"]) == planned["n_records"] and row["background"] == planned["background"], f"window workload mismatch: {name}/{row_number}")
        require(row["probe_action"] == actions[window_id % len(actions)], f"probe schedule mismatch: {name}/{row_number}")
        require(row["valid"] == "1", f"invalid window: {name}/{row_number}")
        require(throttle_value(row["throttled_start"]) & 0xF == 0 and throttle_value(row["throttled_end"]) & 0xF == 0, f"window hardware fault: {name}/{row_number}")

        frequency = float(row["cpu_frequency_min_mhz"])
        t_start = float(row["T_start_C"])
        t_end = float(row["T_end_C"])
        require(frequency >= config["minimum_cpu_frequency_mhz"], f"frequency violation: {name}/{row_number}")
        require(max(t_start, t_end) < config["constraints"]["emergency_stop_c"], f"temperature stop: {name}/{row_number}")
        candidate_min_frequency = min(candidate_min_frequency, frequency)
        candidate_max_temperature = max(candidate_max_temperature, t_start, t_end)

        logged_prediction = json.loads(row["predicted_latency_by_action_ms"])
        logged_latency_upper = json.loads(row["guarded_latency_by_action_ms"])
        logged_latency_headroom = json.loads(row["latency_headroom_by_action_ms"])
        logged_temperature_upper = json.loads(row["guarded_temperature_by_action_C"])
        logged_thermal_headroom = json.loads(row["thermal_headroom_by_action_C"])
        recomputed_safe = []
        for action in actions:
            predicted = latency[(action, f"n{planned['n_records']}", planned["background"])]
            latency_upper = predicted + margins["epsilon_tau_ms_by_action"][action] + margins["fixed_latency_buffer_ms"]
            latency_headroom = config["constraints"]["latency_cap_ms"] - latency_upper
            temperature_upper = guarded_temperature(
                thermal,
                margins,
                action,
                t_start,
                planned["n_records"],
                background_duty[planned["background"]],
                predicted,
            )
            thermal_headroom = config["constraints"]["thermal_cap_c"] - temperature_upper
            require(close(logged_prediction[action], predicted), f"latency prediction mismatch: {name}/{row_number}/{action}")
            require(close(logged_latency_upper[action], latency_upper), f"guarded latency mismatch: {name}/{row_number}/{action}")
            require(close(logged_latency_headroom[action], latency_headroom), f"latency headroom mismatch: {name}/{row_number}/{action}")
            require(close(logged_temperature_upper[action], temperature_upper), f"guarded temperature mismatch: {name}/{row_number}/{action}")
            require(close(logged_thermal_headroom[action], thermal_headroom), f"thermal headroom mismatch: {name}/{row_number}/{action}")
            if latency_headroom >= 0.0 and thermal_headroom >= 0.0:
                recomputed_safe.append(action)

        require(row["safe_set_mask"] == "|".join(recomputed_safe), f"safe-set mask mismatch: {name}/{row_number}")
        require(int(row["safe_set_size"]) == len(recomputed_safe), f"safe-set size mismatch: {name}/{row_number}")
        require(int(row["fallback_required"]) == int(not recomputed_safe), f"fallback mismatch: {name}/{row_number}")
        safe_counts[len(recomputed_safe)] += 1
        admissible_counts.update(recomputed_safe)
        fallback_count += int(not recomputed_safe)

    binding_fraction = sum(safe_counts[size] for size in range(len(actions))) / total_expected
    at_least_two_fraction = sum(count for size, count in safe_counts.items() if size >= 2) / total_expected
    fallback_rate = fallback_count / total_expected
    distinct_actions = [action for action in actions if admissible_counts[action]]
    qualified = (
        binding_fraction >= config["qualification"]["minimum_binding_fraction"]
        and at_least_two_fraction >= config["qualification"]["minimum_at_least_two_safe_fraction"]
        and len(distinct_actions) >= config["qualification"]["minimum_distinct_admissible_actions"]
        and fallback_rate <= config["qualification"]["maximum_fallback_rate"]
    )
    require(manifest["safe_set_size_counts"] == {str(size): safe_counts[size] for size in range(len(actions) + 1)}, f"safe count mismatch: {name}")
    require(close(manifest["binding_fraction"], binding_fraction), f"binding fraction mismatch: {name}")
    require(close(manifest["at_least_two_safe_fraction"], at_least_two_fraction), f"two-safe fraction mismatch: {name}")
    require(close(manifest["fallback_rate"], fallback_rate), f"fallback rate mismatch: {name}")
    require(manifest["distinct_admissible_actions"] == distinct_actions, f"admissible action mismatch: {name}")
    require(manifest["qualified"] is qualified and int(progress["qualified"]) == int(qualified), f"qualification mismatch: {name}")
    require(close(manifest["maximum_temperature_C"], candidate_max_temperature, 0.001), f"maximum temperature mismatch: {name}")
    require(close(manifest["minimum_cpu_frequency_mhz"], candidate_min_frequency, 0.001), f"minimum frequency mismatch: {name}")

    candidate_results.append({
        "candidate_index": index,
        "n_records": planned["n_records"],
        "background": planned["background"],
        "windows": len(rows),
        "safe_set_size_counts": {str(size): safe_counts[size] for size in range(len(actions) + 1)},
        "binding_fraction": binding_fraction,
        "at_least_two_safe_fraction": at_least_two_fraction,
        "fallback_rate": fallback_rate,
        "distinct_admissible_actions": distinct_actions,
        "maximum_temperature_C": candidate_max_temperature,
        "minimum_cpu_frequency_mhz": candidate_min_frequency,
        "qualified": qualified,
    })
    total_windows += len(rows)
    global_max_temperature = max(global_max_temperature, candidate_max_temperature)
    global_min_frequency = min(global_min_frequency, candidate_min_frequency)

qualified_candidates = [row for row in candidate_results if row["qualified"]]
require(len(qualified_candidates) == 1, "expected exactly one evaluated qualifier")
selected = qualified_candidates[0]
require(selected == candidate_results[-1], "sweep did not stop at first qualifier")
require(
    summary["selected_candidate"]
    == {key: selected[key] for key in ("candidate_index", "n_records", "background")},
    "selected candidate mismatch",
)
require(total_windows == 7200, "total diagnostic window count is not 7200")

raw_artifact_hashes = {
    str(path.relative_to(CAMPAIGN)): digest(path)
    for path in sorted(CAMPAIGN.rglob("*"))
    if path.is_file() and path != VALIDATION_PATH
}
validation = {
    "status": "BINDING_STRESS_V3_INDEPENDENT_VALIDATION_PASS",
    "validated_campaign": CAMPAIGN.name,
    "source_commit": FROZEN_COMMIT,
    "config_sha256": digest(CONFIG_PATH),
    "completed_utc": summary["completed_utc"],
    "candidate_order": "exact prefix of frozen grid; stopped at first qualifier",
    "candidates_evaluated": len(candidate_results),
    "windows": total_windows,
    "diagnostic_seeds": config["diagnostic_seeds"],
    "windows_per_candidate": total_expected,
    "idle_samples": total_idle_samples,
    "fault_monitor_samples": total_fault_samples,
    "current_fault_low_nibble_windows": 0,
    "artifact_hashes_reproduced_from_candidate_manifests": artifact_hashes_reproduced,
    "forbidden_selection_inputs_used": [],
    "candidate_results": candidate_results,
    "selected_candidate": summary["selected_candidate"],
    "global_maximum_temperature_C": global_max_temperature,
    "global_minimum_cpu_frequency_mhz": global_min_frequency,
    "raw_artifact_sha256": raw_artifact_hashes,
}
VALIDATION_PATH.write_text(json.dumps(validation, indent=2) + "\n")
print(json.dumps(validation, indent=2))
