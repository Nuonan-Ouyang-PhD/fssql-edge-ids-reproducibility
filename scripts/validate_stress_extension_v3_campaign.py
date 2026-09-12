#!/usr/bin/env python3
import csv, hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / 'evidence/formal_stress_extension_v3_20260910T004200Z'
CFG = ROOT / 'configs/pi3b_stress_extension_v3.json'
MATRIX = ROOT / 'configs/pi3b_stress_extension_v3_matrix.csv'

def sha(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
cfg=json.loads(CFG.read_text()); matrix=list(csv.DictReader(MATRIX.open()))
progress=list(csv.DictReader((C/'CAMPAIGN_PROGRESS.csv').open()))
assert len(matrix)==len(progress)==60
assert json.loads((C/'CAMPAIGN_COMPLETE.json').read_text())['runs']==60
assert not (C/'CAMPAIGN_STOP.json').exists()
agg=defaultdict(lambda: {'runs':0,'windows':0,'records':0,'tp':0,'fp':0,'tn':0,'fn':0,'fallbacks':0,'latency_violations':0,'thermal_violations':0,'any_violations':0,'max_end_to_end_ms':0.0,'max_temperature_C':-math.inf,'min_cpu_frequency_mhz':math.inf,'actions':Counter(),'safe_set_sizes':Counter()})
total=0; faults=0; indices_ok=True; hashes_ok=True; statuses_ok=True
for i,(m,p) in enumerate(zip(matrix,progress),1):
    d=C/f'{i:03d}_{m["planned_run_id"]}'; man=json.loads((d/'RUN_MANIFEST.json').read_text())
    statuses_ok &= p['status']=='FORMAL_POLICY_RUN_PASS' and man['status']=='FORMAL_POLICY_RUN_PASS' and man['valid'] is True and man['formal'] is True
    hashes_ok &= man['git_commit']=='3a345622dc00480a519b3c915801817b59a32529' and man['config_sha256']==sha(CFG) and man['windows_observed']==600
    for n,h in man['artifact_sha256'].items(): hashes_ok &= sha(d/n)==h
    rows=list(csv.DictReader((d/'windows.csv').open())); total += len(rows)
    key=(m['policy'],m['regime']); a=agg[key]; a['runs']+=1; a['windows']+=len(rows)
    seen=Counter()
    for r in rows:
        a['records'] += int(r['n_records']); a['tp']+=int(r['tp']); a['fp']+=int(r['fp']); a['tn']+=int(r['tn']); a['fn']+=int(r['fn']); a['fallbacks']+=int(r['fallback']); a['latency_violations']+=int(r['latency_violation']); a['thermal_violations']+=int(r['thermal_violation']); a['any_violations']+=int(r['any_violation']); a['max_end_to_end_ms']=max(a['max_end_to_end_ms'],float(r['end_to_end_ms'])); a['max_temperature_C']=max(a['max_temperature_C'],float(r['T_end_C'])); a['min_cpu_frequency_mhz']=min(a['min_cpu_frequency_mhz'],float(r['cpu_frequency_min_mhz'])); a['actions'][r['selected_action']]+=1; a['safe_set_sizes'][r['safe_set_size']]+=1
        seen.update(json.loads(r['batch_array_indices']));
    indices_ok &= len(seen)==600 and set(seen.values())=={16}
    faults += sum(1 for r in csv.DictReader((d/'fault_telemetry_1hz.csv').open()) if int(r['throttled'],16)&15)
out=[]
for (policy,regime),a in sorted(agg.items()):
    out.append({'policy':policy,'regime':regime,**{k:(dict(v) if isinstance(v,Counter) else v) for k,v in a.items()}})
ind={'status':'PASS' if total==36000 and statuses_ok and hashes_ok and indices_ok and faults==0 else 'FAIL','campaign':'formal_stress_extension_v3_20260910T004200Z','runs':len(progress),'windows':total,'records':sum(x['records'] for x in out),'hardware_fault_samples':faults,'statuses_ok':statuses_ok,'hashes_ok':hashes_ok,'trace_index_multiplicity_ok':indices_ok,'aggregate_sha256':None}
(C/'STRESS_EXTENSION_V3_POLICY_REGIME_AGGREGATE.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
ind['aggregate_sha256']=sha(C/'STRESS_EXTENSION_V3_POLICY_REGIME_AGGREGATE.json')
(C/'INDEPENDENT_VALIDATION.json').write_text(json.dumps(ind,indent=2,sort_keys=True)+'\n')
print(json.dumps(ind,indent=2))
