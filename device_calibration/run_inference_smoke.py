#!/usr/bin/env python3
"""Device-local development-data inference smoke and timing run."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from action_pool_numpy import NumpyActionModel
from preprocessor_numpy import PortablePreprocessor
from risk_proxy_numpy import RiskProxy


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "device_smoke_output"
ACTIONS = [
    "fisvdd_revision",
    "lucid_revision",
    "tinydl_revision",
    "oi_svdd_as_elm_revision",
]
RISK_FEATURES = [
    "rate", "sttl", "dttl", "sload", "dload", "ct_state_ttl",
    "ct_dst_src_ltm", "is_sm_ips_ports",
]


def command(args: list[str]) -> str:
    return subprocess.run(args, text=True, capture_output=True, check=False).stdout.strip()


def temperature() -> float | None:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def summarize(values: list[int]) -> dict[str, float]:
    samples = np.asarray(values, dtype=np.float64) / 1e6
    return {
        "n": int(len(samples)),
        "mean_ms": float(samples.mean()),
        "p50_ms": float(np.quantile(samples, 0.50)),
        "p95_ms": float(np.quantile(samples, 0.95)),
        "p99_ms": float(np.quantile(samples, 0.99)),
        "max_ms": float(samples.max()),
    }


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


initial_throttled = command(["vcgencmd", "get_throttled"])
initial_throttled_value = int(initial_throttled.split("=")[-1], 16)
if initial_throttled_value & 0xF:
    raise RuntimeError(f"current undervoltage/throttle flag at start: {initial_throttled}")
if OUTPUT.exists():
    raise FileExistsError(f"refusing to overwrite device smoke output: {OUTPUT}")
OUTPUT.mkdir()

with (ROOT / "scheduler_train_6000_raw_features.csv").open(newline="", encoding="utf-8") as stream:
    raw_rows = list(csv.DictReader(stream))
features = np.load(ROOT / "scheduler_train_6000_features.npy", allow_pickle=False)
risk_features = np.load(ROOT / "scheduler_train_6000_risk_features.npy", allow_pickle=False)
preprocessor = PortablePreprocessor(ROOT / "preprocessor_portable.npz")
risk_proxy = RiskProxy(ROOT / "risk_proxy.npz")
models = {
    action: NumpyActionModel(ROOT / f"{action}.npz")
    for action in ACTIONS
}

for action, model in models.items():
    for index in range(100):
        model.predict_proba(features[index:index + 1])
for index in range(100):
    preprocessor.transform([raw_rows[index]])
    risk_proxy.predict(risk_features[index:index + 1])

rows = []
durations: dict[str, list[int]] = {
    "preprocessing": [],
    "risk_proxy": [],
    **{action: [] for action in ACTIONS},
}
start_temperature = temperature()
for sequence in range(len(features)):
    start = time.perf_counter_ns()
    transformed = preprocessor.transform([raw_rows[sequence]])
    duration = time.perf_counter_ns() - start
    durations["preprocessing"].append(duration)
    rows.append([sequence, "preprocessing", duration, ""])
    if not np.array_equal(transformed[0], features[sequence]):
        raise RuntimeError(f"portable preprocessing mismatch at sequence {sequence}")

    start = time.perf_counter_ns()
    probability = float(risk_proxy.predict(risk_features[sequence:sequence + 1])[0])
    duration = time.perf_counter_ns() - start
    durations["risk_proxy"].append(duration)
    rows.append([sequence, "risk_proxy", duration, probability])

    for action, model in models.items():
        start = time.perf_counter_ns()
        probability = float(model.predict_proba(features[sequence:sequence + 1])[0])
        duration = time.perf_counter_ns() - start
        durations[action].append(duration)
        rows.append([sequence, action, duration, probability])

end_temperature = temperature()
final_throttled = command(["vcgencmd", "get_throttled"])
final_throttled_value = int(final_throttled.split("=")[-1], 16)
csv_path = OUTPUT / "component_latency.csv"
with csv_path.open("x", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["sequence", "component", "duration_ns", "probability"])
    writer.writerows(rows)

summary = {
    "status": (
        "PRELIMINARY_DEVICE_SMOKE_PASS"
        if not final_throttled_value & 0xF
        else "PRELIMINARY_DEVICE_SMOKE_INVALID_CURRENT_THROTTLE"
    ),
    "evidence_boundary": "scheduler_train-only pipeline smoke; not formal guard calibration or evaluation",
    "hostname": platform.node(),
    "platform": platform.platform(),
    "architecture": platform.machine(),
    "python": platform.python_version(),
    "numpy": np.__version__,
    "cpu_count": os.cpu_count(),
    "governors": command(["sh", "-c", "paste -sd, /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"]),
    "throttled_before_after": [initial_throttled, final_throttled],
    "temperature_c_before_after": [start_temperature, end_temperature],
    "rows": len(features),
    "components": {name: summarize(values) for name, values in durations.items()},
    "raw_log": csv_path.name,
    "raw_log_sha256": sha256(csv_path),
}
with (OUTPUT / "SUMMARY.json").open("x", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2)
    stream.write("\n")
print(json.dumps(summary, indent=2))
