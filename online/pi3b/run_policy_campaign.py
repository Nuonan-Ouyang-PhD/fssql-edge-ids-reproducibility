#!/usr/bin/env python3
"""Execute the frozen matrix sequentially and stop on the first invalid run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite campaign output: {args.output}")
    args.output.mkdir(parents=True)
    config = json.loads(args.config.read_text())
    if sha256(args.matrix) != config["execution_order"]["expected_matrix_sha256"]:
        raise RuntimeError("campaign matrix hash mismatch")
    rows = list(csv.DictReader(args.matrix.open()))
    if len(rows) != 160 or [int(row["global_order"]) for row in rows] != list(range(1, 161)):
        raise RuntimeError("campaign matrix is incomplete or out of order")
    progress_path = args.output / "CAMPAIGN_PROGRESS.csv"
    fields = ["global_order", "run_id", "status", "start_utc", "end_utc", "output", "failure_reason"]
    with progress_path.open("x", newline="", encoding="utf-8", buffering=1) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            run_id = row["planned_run_id"]
            run_output = args.output / f"{int(row['global_order']):03d}_{run_id}"
            start = utc_now()
            command = [
                sys.executable, str(ROOT / "online/pi3b/collect_policy_run.py"),
                "--config", str(args.config), "--input", str(args.input),
                "--policy", row["policy"], "--regime", row["regime"], "--seed", row["seed"],
                "--run-id", run_id, "--output", str(run_output), "--git-commit", args.git_commit,
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            manifest_path = run_output / "RUN_MANIFEST.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            status = manifest.get("status", "CAMPAIGN_CHILD_NO_MANIFEST")
            failure = manifest.get("failure_reason", result.stderr.strip() or result.stdout.strip())
            writer.writerow({
                "global_order": row["global_order"], "run_id": run_id, "status": status,
                "start_utc": start, "end_utc": utc_now(), "output": run_output.name,
                "failure_reason": failure,
            })
            stream.flush()
            if result.returncode or status != "FORMAL_POLICY_RUN_PASS":
                (args.output / "CAMPAIGN_STOP.json").write_text(json.dumps({
                    "status": "CAMPAIGN_STOPPED_ON_INVALID_ATTEMPT", "global_order": int(row["global_order"]),
                    "run_id": run_id, "failure_reason": failure, "automatic_retry": False,
                }, indent=2) + "\n")
                raise SystemExit(result.returncode or 2)
    (args.output / "CAMPAIGN_COMPLETE.json").write_text(json.dumps({
        "status": "FORMAL_PI3B_POLICY_MATRIX_COMPLETE", "runs": len(rows), "completed_utc": utc_now(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
