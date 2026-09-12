#!/usr/bin/env python3
"""Fail-closed Pi4B8 formal lineage gate (separate from development gate)."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=ROOT/'configs/pi4b8_online_experiment_v1.json'); p.add_argument('--commissioning',action='store_true'); p.add_argument('--formal',action='store_true'); a=p.parse_args(); c=json.loads(a.config.read_text())
 required=(ROOT/'evidence/pi4b8_commissioning/PI4B8_COMMISSIONING_VALIDATION.json',ROOT/'evidence/pi4b8_commissioning/PI4B8_COMMISSIONING_DECISION.md',ROOT/'PI4B8_FINAL_EXECUTION_MANIFEST.json')
 if a.formal and (not a.commissioning or any(not x.exists() for x in required)): raise SystemExit('formal execution blocked: successful commissioning and final execution manifests are required')
 if c['official_test']!='sealed': raise SystemExit('official test must remain sealed')
 if c['guard']['latency_cap_ms']!=40.0 or c['guard']['thermal_cap_C']!=82.0 or c['guard']['decision_window_ms']!=100.0: raise SystemExit('guard mismatch')
 for key in ('latency_predictor','thermal_predictor','margins'):
  pth=ROOT/c['guard'][key]
  if not pth.exists() or sha(pth)!=c['guard']['hashes'][key+'_sha256']: raise SystemExit(f'guard hash mismatch: {key}')
 pkg=ROOT/'.formal_inputs/pi4b8_scheduler_train_v1'; m=json.loads((pkg/'PACKAGE_MANIFEST.json').read_text())
 if m['official_test']!='sealed_not_read' or m['windows_per_run']!=600: raise SystemExit('input package is not sealed development package')
 for n,r in m['files'].items():
  if sha(pkg/n)!=r['sha256']: raise SystemExit(f'input hash mismatch: {n}')
 q=ROOT/'q_training/pi4b8_q_v1/Q_TABLE_MANIFEST.json'; qm=json.loads(q.read_text())
 if qm['seeds'] != [5101,5102,5103,5104,5105]: raise SystemExit('Q seed mapping mismatch')
 for n,h in qm['artifact_sha256'].items():
  if sha(q.parent/n)!=h: raise SystemExit(f'Q hash mismatch: {n}')
 for f in pkg.glob('*_features.npy'):
  x=np.load(f,allow_pickle=False)
  if x.shape[0]!=600 or x.shape[1] not in (191,8): raise SystemExit(f'bad runtime input: {f.name}')
 for f in pkg.glob('*_raw_features.csv'):
  h=f.read_text().splitlines()[0].split(',')
  if any(x in h for x in ('label','attack_cat','source_id','source_row_index')): raise SystemExit(f'label-bearing runtime input: {f.name}')
 print(json.dumps({'status':'PI4B8_FORMAL_RUNNER_LINEAGE_GATE_PASS','commissioning_required':True,'formal_execution':'blocked_until_commissioning_and_final_manifest'},indent=2))
if __name__=='__main__': main()
