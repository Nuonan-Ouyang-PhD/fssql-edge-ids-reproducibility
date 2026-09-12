#!/usr/bin/env python3
"""Pi4B8 online-phase gate; formal execution remains blocked until inputs are frozen."""
from __future__ import annotations
import argparse, hashlib, json, socket
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=ROOT/'configs/pi4b8_online_experiment_v1.json'); p.add_argument('--formal',action='store_true'); a=p.parse_args(); c=json.loads(a.config.read_text())
    if socket.gethostname()!=c['device_id']: raise SystemExit(f"wrong host: expected {c['device_id']}, got {socket.gethostname()}")
    if c['guard']['latency_cap_ms']!=40.0 or c['guard']['thermal_cap_C']!=82.0 or c['guard']['decision_window_ms']!=100.0: raise SystemExit('guard cap/window mismatch')
    if a.formal: raise SystemExit('formal execution blocked: Pi4B8 input and policy runner freeze is incomplete')
    print(json.dumps({'status':'PI4B8_ONLINE_GATE_PASS_DEVELOPMENT_ONLY','device_id':c['device_id'],'config_sha256':sha256(a.config)},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
