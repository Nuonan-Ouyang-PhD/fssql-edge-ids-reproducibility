#!/usr/bin/env python3
from pathlib import Path
import sys, json, csv

ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path('major_revision_rebuild')
required=[
 'provenance/PROVENANCE_MANIFEST.csv',
 'data/DATASET_SHA256.json','data/SPLIT_MANIFEST.csv','data/LEAKAGE_AUDIT.json',
 'models/MODEL_REIMPLEMENTATION_DECISION.md','models/MODEL_HYPERPARAMETERS.json','models/MODEL_SHA256.json','models/RISK_PROXY_SPEC.json',
 'calibration/pi3b/guard_coverage_summary.json','calibration/pi3b/guard_residuals.csv',
 'q_training/checkpoints.csv','q_training/Q_TABLE_MANIFEST.json',
 'analysis/per_seed_metrics.csv','analysis/paired_tests.csv','analysis/table13_safe_set_distribution.csv','analysis/margin_sensitivity.csv',
 'overhead/controller_microbenchmark.csv','overhead/switching_matrix.csv',
 'calibration/pi4b8/guard_coverage_summary.json',
 'power/POWER_SUMMARY.csv',
 'audit/REVIEWER_RESPONSE_EVIDENCE_MAP.csv','audit/FINAL_MAJOR_REVISION_REBUILD_AUDIT.md'
]
missing=[p for p in required if not (ROOT/p).exists()]
# Online run counts are checked from manifest rather than raw directories.
manifest=ROOT/'online'/'RUN_MANIFEST.csv'
errs=[]
if not manifest.exists(): missing.append('online/RUN_MANIFEST.csv')
else:
    rows=list(csv.DictReader(manifest.open(encoding='utf-8-sig')))
    valid=[r for r in rows if str(r.get('valid','')).lower() in ('1','true','yes','valid')]
    methods=['Static-Heavy','Static-Light','Round-Robin','Greedy-Risk','Safe-Greedy','Unshielded Q','FSSQL-R','Threshold']
    for regime in ('natural','near_cap'):
        for m in methods:
            n=sum(1 for r in valid if r.get('device_id')=='pi3bplus' and r.get('regime')==regime and r.get('policy')==m)
            if n<10: errs.append(f'pi3bplus {regime} {m}: {n}/10 valid runs')
    for regime in ('natural','near_cap'):
        for m in ['Static-Light','Threshold','Safe-Greedy','FSSQL-R']:
            n=sum(1 for r in valid if r.get('device_id')=='pi4b8g' and r.get('regime')==regime and r.get('policy')==m)
            if n<5: errs.append(f'pi4b8g {regime} {m}: {n}/5 valid runs')

if missing or errs:
    print('FAIL')
    if missing:
        print('Missing required artifacts:')
        for x in missing: print(' -',x)
    if errs:
        print('Run-count errors:')
        for x in errs: print(' -',x)
    sys.exit(2)
print('PASS')
