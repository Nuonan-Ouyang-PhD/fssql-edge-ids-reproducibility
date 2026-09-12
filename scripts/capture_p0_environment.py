#!/usr/bin/env python3
"""Capture local and Raspberry Pi environment snapshots without changing devices."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


TARGETS = {
    "pi3bplus": "pi@pi3bplus.local",
    "pi4b4g": "pi@pi4b.local",
    "pi4b8g": "pi@pi4b8g.local",
    "pi5": "pi5@pi5.local",
}

REMOTE_COMMAND = r"""
set -eu
printf 'hostname='; hostname
printf 'model='; tr -d '\000' </proc/device-tree/model; printf '\n'
printf 'serial='; awk -F ': ' '/^Serial/{print $2}' /proc/cpuinfo
printf 'architecture='; uname -m
printf 'kernel='; uname -r
printf 'os_pretty_name='; sed -n 's/^PRETTY_NAME=//p' /etc/os-release
printf 'python='; python3 --version 2>&1
printf 'cpu_count='; getconf _NPROCESSORS_ONLN
printf 'memory_kib='; awk '/MemTotal/{print $2}' /proc/meminfo
printf 'governors='; paste -sd, /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true
printf 'frequencies_khz='; paste -sd, /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null || true
printf 'temperature='; vcgencmd measure_temp 2>/dev/null || cat /sys/class/thermal/thermal_zone0/temp
printf 'throttled='; vcgencmd get_throttled 2>/dev/null || true
printf 'clock_arm='; vcgencmd measure_clock arm 2>/dev/null || true
printf 'loadavg='; cat /proc/loadavg
printf 'addresses='; hostname -I
printf 'time_utc='; date -u +%Y-%m-%dT%H:%M:%SZ
printf 'ntp='; timedatectl show -p NTPSynchronized --value 2>/dev/null || true
printf 'cooling_devices='; for p in /sys/class/thermal/cooling_device*/type; do [ -r "$p" ] && tr '\n' ',' <"$p"; done; printf '\n'
""".strip()


def run(command: list[str], timeout: int = 30) -> dict:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
        return {
            "command": command,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "command": command,
            "exit_code": None,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
            "error": "timeout",
        }


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)

snapshot = {
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "local": {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "sw_vers": run(["sw_vers"]),
        "uname": run(["uname", "-a"]),
        "network": run(["ifconfig", "en0"]),
        "usb": run(["system_profiler", "SPUSBDataType"]),
    },
    "devices": {},
}

for device_id, target in TARGETS.items():
    snapshot["devices"][device_id] = run(
        [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=8",
            "-o", "StrictHostKeyChecking=accept-new",
            target,
            REMOTE_COMMAND,
        ],
        timeout=40,
    )

with output.open("x", encoding="utf-8") as stream:
    json.dump(snapshot, stream, indent=2, ensure_ascii=False)
    stream.write("\n")

print(output)

