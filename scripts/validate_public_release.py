#!/usr/bin/env python3
"""Validate current R6 analysis/manuscript release; validate submission-ready evidence."""
from __future__ import annotations
import argparse,hashlib,json,sys,subprocess
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_pi4b8_r6 import validate_cells
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--with-evidence',action='store_true');args=p.parse_args();errors=[]
 manifest=ROOT/'PUBLIC_RELEASE_MANIFEST_SHA256.txt'
 if not manifest.exists(): errors.append('Missing public release manifest')
 else:
  seen=set()
  for line in manifest.read_text().splitlines():
   if not line.strip():continue
   digest,relative=line.split('  ',1);target=(ROOT/relative).resolve()
   if not target.is_relative_to(ROOT.resolve()) or relative in seen: errors.append('Invalid/duplicate path: '+relative);continue
   seen.add(relative)
   if not target.is_file() or sha(target)!=digest:errors.append('Checksum mismatch: '+relative)
  try:
   tracked={x for x in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if x}
   tracked.discard('PUBLIC_RELEASE_MANIFEST_SHA256.txt')
   listed=set(seen)
   omitted=sorted(tracked-listed)
   extra=sorted(listed-tracked)
   if omitted: errors.append('Unlisted release files: '+', '.join(omitted))
   if extra: errors.append('Manifest paths not tracked: '+', '.join(extra))
  except (OSError,subprocess.CalledProcessError):
   errors.append('Cannot enumerate tracked release files for completeness check')
 d=ROOT/'results/pi4b8_r6_revision'
 for n in ['RUN_LEVEL_METRICS.csv','PAIRED_ALL_POLICY_CONTRASTS.csv','Q_STATE_SUPPORT.csv','ACCEPTED_RUN_MANIFEST.json','ANALYSIS_VALIDATION.json']:
  if not (d/n).is_file():errors.append('Missing current analysis: '+n)
 if errors:
  print(json.dumps({'status':'FAIL','errors':errors},indent=2));raise SystemExit(1)
 cells=json.loads((d/'ACCEPTED_RUN_MANIFEST.json').read_text())['cells'];validate_cells(cells)
 runs=pd.read_csv(d/'RUN_LEVEL_METRICS.csv');paired=pd.read_csv(d/'PAIRED_ALL_POLICY_CONTRASTS.csv')
 if len(runs)!=60 or runs.windows.sum()!=36000 or runs.records_processed.sum()!=576000:errors.append('Incorrect R6 totals')
 if len(paired[paired.metric=='fpr'])!=18:errors.append('Missing FPR contrasts')
 if (paired.n_pairs!=5).any():errors.append('Wrong paired statistical unit')
 for n in ['main.tex','supplement.tex','response_to_reviewers.tex','cover_letter.tex']:
  if not (ROOT/'paper'/n).is_file():errors.append('Missing manuscript source: '+n)
 for n in ['main','supp','response','cover-letter']:
  if not (ROOT/'paper'/f'TSUSC-2026-07-0200-{n}.pdf').is_file():errors.append('Missing current manuscript PDF: '+n)
 if args.with_evidence:
  for c in cells:
   base=ROOT/c['evidence_directory']
   for name,key in [('pi_windows.csv','pi_raw_sha256'),('POWERZ_raw.csv','powerz_raw_sha256'),('RUN_MANIFEST.json','run_manifest_sha256')]:
    target=base/name
    if not target.is_file() or sha(target)!=c[key]:errors.append('R6 evidence hash mismatch: '+str(target.relative_to(ROOT)))
  runtime=ROOT/'evidence/pi4b8_corrected/pi4b8_formal_runtime_v5_r6'
  if not runtime.is_dir():errors.append('Missing frozen executed runtime')
 print(json.dumps({'status':'FAIL' if errors else 'R6_RELEASE_CHECK_PASS','accepted_cells':len(runs),'accepted_windows':int(runs.windows.sum()),'record_evaluations':int(runs.records_processed.sum()),'fpr_contrasts':18,'raw_hashes_checked':bool(args.with_evidence),'errors':errors},indent=2))
 if errors: raise SystemExit(1)
if __name__=='__main__':main()
