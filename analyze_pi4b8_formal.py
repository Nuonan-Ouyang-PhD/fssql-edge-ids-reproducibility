#!/usr/bin/env python3
import csv,json,hashlib,math
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel,wilcoxon
ROOT=Path(__file__).resolve().parent; BASE=ROOT/'evidence/pi4b8_formal_matrix'; OUT=ROOT/'analysis/pi4b8_formal_v1'; OUT.mkdir(parents=True,exist_ok=True)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p): return list(csv.DictReader(open(p,newline='')))
comp=json.load(open(ROOT/'PI4B8_FORMAL_EVIDENCE_COMPLETENESS_AFTER_RECOVERY.json')); selected=[]
for x in comp['runs']:
 rid=x['run_id']; aid=rid+'_attempt2' if x['execution_order'] in (40,51) else rid; d=BASE/aid; files=list(d.glob('*/windows.csv')); assert files,aid
 selected.append({'cell_run_id':rid,'accepted_attempt_id':aid,'execution_order':x['execution_order'],'policy':x['policy'],'condition':x['condition'],'seed':x['seed'],'pi_raw':str(files[0]),'powerz':str(d/'POWERZ_raw.csv'),'handshake':str(d/'HANDSHAKE.json')})
manifest={'status':'FROZEN_ACCEPTED_SELECTOR','accepted_cell_count':60,'selection_rule':'original attempt for 58 valid cells; _attempt2 for execution orders 40 and 51; never mix attempts','runs':selected}; mp=OUT/'ACCEPTED_RUN_MANIFEST.json'; mp.write_text(json.dumps(manifest,indent=2)+'\n'); (OUT/'ACCEPTED_RUN_MANIFEST_SHA256.txt').write_text(sha(mp)+'\n')
metrics=[]
for a in selected:
 rr=rows(a['pi_raw']); vals=lambda k:np.array([float(r[k]) for r in rr]); c={k:sum(int(r[k]) for r in rr) for k in ('tp','fp','fn','tn')}; tp,fp,fn,tn=[c[k] for k in ('tp','fp','fn','tn')]; pr=tp/(tp+fp) if tp+fp else 0; re=tp/(tp+fn) if tp+fn else 0; f1=2*pr*re/(pr+re) if pr+re else 0; actions={}
 for r in rr: actions[r['selected_action']]=actions.get(r['selected_action'],0)+1
 e2=vals('e2e_ms'); ctl=vals('controller_ms'); det=vals('detector_inference_ms'); temp=vals('temperature_C'); sw=sum(r['previous_action'] not in ('','None') and r['previous_action']!=r['selected_action'] for r in rr)
 energy=power=peak=float('nan'); pz=Path(a['powerz']); hs=Path(a['handshake'])
 if pz.exists() and hs.exists():
  mm=rows(pz); mt=np.array([float(r['sample_monotonic']) for r in mm]); pw=np.array([float(r['power_W']) for r in mm]); h=json.load(open(hs)); s,e=h['start_ack_receive_mac_monotonic'],h['end_ack_receive_mac_monotonic']; tt=np.r_[s,mt[(mt>s)&(mt<e)],e]; pp=np.interp(tt,mt,pw); energy=float(np.trapz(pp,tt)); power=energy/(e-s); peak=float(pp.max())
 metrics.append({**a,**c,'precision':pr,'recall':re,'f1':f1,'fpr':fp/(fp+tn) if fp+tn else 0,'cumulative_reward':float(vals('reward').sum()),'mean_reward':float(vals('reward').mean()),'p50_e2e':float(np.percentile(e2,50)),'p95_e2e':float(np.percentile(e2,95)),'p99_e2e':float(np.percentile(e2,99)),'max_e2e':float(e2.max()),'mean_controller':float(ctl.mean()),'p95_controller':float(np.percentile(ctl,95)),'p99_controller':float(np.percentile(ctl,99)),'mean_detector':float(det.mean()),'switch_count':int(sw),'switch_rate':sw/len(rr),'switch_cost':float(vals('reward_switch').sum()),'latency_violations':int((e2>40).sum()),'thermal_violations':int((temp>82).sum()),'safe_set_mean':float(vals('safe_set_size').mean()),'mean_temperature':float(temp.mean()),'max_temperature':float(temp.max()),'mean_power_W':power,'peak_power_W':peak,'total_energy_J':energy,'total_energy_Wh':energy/3600 if math.isfinite(energy) else float('nan'),'J_per_window':energy/len(rr) if math.isfinite(energy) else float('nan'),'J_per_1000_records':energy*1000/len(rr) if math.isfinite(energy) else float('nan'),'J_per_correct_attack':energy/tp if tp and math.isfinite(energy) else float('nan'),'action_composition':actions})
 fields=[k for k in metrics[0] if k not in ('pi_raw','powerz','handshake','action_composition')]+['action_composition']; outcsv=OUT/'RUN_LEVEL_METRICS.csv'
with outcsv.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); [w.writerow({k:(json.dumps(m[k],sort_keys=True) if k=='action_composition' else m[k]) for k in fields}) for m in metrics]
def boot(x):
 rng=np.random.default_rng(20260911); z=[np.mean(rng.choice(x,len(x),replace=True)) for _ in range(10000)]; return [float(np.percentile(z,2.5)),float(np.percentile(z,97.5))]
keys=['f1','precision','recall','fpr','cumulative_reward','mean_reward','p50_e2e','p95_e2e','p99_e2e','max_e2e','mean_controller','p95_controller','p99_controller','mean_detector','switch_count','switch_rate','switch_cost','latency_violations','thermal_violations','safe_set_mean','mean_temperature','max_temperature','mean_power_W','peak_power_W','total_energy_J','total_energy_Wh','J_per_window','J_per_1000_records','J_per_correct_attack']; pairs=[]
for cond in ('n16/B0','n16/B1','n16/B2'):
 for pa,pb in (('FSSQL-R','Safe-Greedy'),('FSSQL-R','Threshold'),('FSSQL-R','Static-Light'),('Safe-Greedy','Threshold')):
  for k in keys:
   aa=np.array([m[k] for m in metrics if m['policy']==pa and m['condition']==cond]); bb=np.array([m[k] for m in metrics if m['policy']==pb and m['condition']==cond]); diff=aa-bb
   pairs.append({'metric':k,'condition':cond,'policy_a':pa,'policy_b':pb,'a_values':aa.tolist(),'b_values':bb.tolist(),'paired_difference_values':diff.tolist(),'mean_difference':float(np.mean(diff)),'sd_difference':float(np.std(diff,ddof=1)),'median_difference':float(np.median(diff)),'bootstrap_ci95_mean_difference':boot(diff),'paired_t_p':float(ttest_rel(aa,bb).pvalue),'wilcoxon_p_exact':float(wilcoxon(diff).pvalue) if np.any(diff) else 1.0,'paired_cohen_dz':float(np.mean(diff)/np.std(diff,ddof=1)) if np.std(diff,ddof=1)>0 else float('nan')})
(OUT/'PAIRED_SEED_RESULTS.json').write_text(json.dumps(pairs,indent=2,allow_nan=True)+'\n'); (OUT/'ANALYSIS_README.md').write_text('Statistical unit is matched run/seed (n=5), never pooled windows. Energy is absolute KM003C DC input-side integration over Mac ACK boundaries. Official test remains sealed.\n'); print('PI4B8_ANALYSIS_PIPELINE_PASS',len(metrics),'runs',len(pairs),'paired results')
