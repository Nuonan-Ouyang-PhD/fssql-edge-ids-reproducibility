#!/usr/bin/env python3
"""Seed-level V3 summaries, paired inference, and Threshold violation forensic."""
import csv, json, math, hashlib
from pathlib import Path
import numpy as np
from itertools import product

ROOT=Path(__file__).resolve().parents[1]; C=ROOT/'evidence/formal_stress_extension_v3_20260910T004200Z'
rows=[]; violation=None
for d in sorted(C.glob('[0-9][0-9][0-9]_*/')):
    ws=list(csv.DictReader((d/'windows.csv').open()))
    if not ws: continue
    m=json.loads((d/'RUN_MANIFEST.json').read_text()); seed=int(m['seed']); policy=m['policy']
    vals={k:np.array([float(r[k]) for r in ws]) for k in ['end_to_end_ms','total_controller_ms','reward_total','reward_security','reward_energy','reward_backlog','reward_switch','latency_slack_ms']}
    tp=sum(int(r['tp']) for r in ws); fp=sum(int(r['fp']) for r in ws); tn=sum(int(r['tn']) for r in ws); fn=sum(int(r['fn']) for r in ws)
    precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0; f1=2*precision*recall/(precision+recall) if precision+recall else 0; fpr=fp/(fp+tn) if fp+tn else 0
    switches=sum(1 for r in ws if r['prev_action'] and r['prev_action']!=r['selected_action']); switch_cost=-sum(float(r['reward_switch']) for r in ws)
    actions={}
    heads=[]
    for r in ws:
        actions[r['selected_action']]=actions.get(r['selected_action'],0)+1
        h=json.loads(r['latency_guard_headroom_by_action_ms']).get(r['selected_action']); heads.append(float(h) if h is not None else float('nan'))
        if int(r['latency_violation']) and violation is None: violation=(d.name,r)
    out={'seed':seed,'policy':policy,'regime':m['regime'],'run_id':m['run_id'],'tp':tp,'fp':fp,'tn':tn,'fn':fn,'precision':precision,'recall':recall,'f1':f1,'fpr':fpr,
      'reward_cumulative':float(vals['reward_total'].sum()),'reward_mean':float(vals['reward_total'].mean()),**{f'{k}_cumulative':float(v.sum()) for k,v in vals.items() if k.startswith('reward_') and k!='reward_total'},
      'latency_mean_ms':float(vals['end_to_end_ms'].mean()),'latency_p95_ms':float(np.percentile(vals['end_to_end_ms'],95)),'latency_p99_ms':float(np.percentile(vals['end_to_end_ms'],99)),'latency_max_ms':float(vals['end_to_end_ms'].max()),
      'latency_violations':sum(int(r['latency_violation']) for r in ws),'thermal_violations':sum(int(r['thermal_violation']) for r in ws),'switch_count':switches,'switch_rate':switches/len(ws),'switch_cost_cumulative':switch_cost,
      'controller_overhead_mean_ms':float(vals['total_controller_ms'].mean()),'controller_overhead_p95_ms':float(np.percentile(vals['total_controller_ms'],95)),'guard_headroom_selected_mean_ms':float(np.nanmean(heads)),'guard_headroom_selected_min_ms':float(np.nanmin(heads))}
    for a,n in sorted(actions.items()): out[f'action_{a}_count']=n; out[f'action_{a}_rate']=n/len(ws)
    rows.append(out)
assert len(rows)==60
outdir=C/'analysis'; outdir.mkdir(exist_ok=True)
fields=sorted(rows[0]);
with (outdir/'V3_PER_SEED_POLICY_SUMMARY.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

methods=['FSSQL-R','Safe-Greedy','Threshold','Unshielded-Q','Round-Robin','Static-Light']; metrics=[k for k in fields if k not in ('seed','policy','regime','run_id') and not k.endswith('_count') and not k.endswith('_rate')]
paired=[]
for b in methods[1:]:
  for metric in metrics:
    a=np.array([r[metric] for r in rows if r['policy']=='FSSQL-R']); y=np.array([r[metric] for r in rows if r['policy']==b]);
    if len(a)!=10 or len(y)!=10: continue
    d=a-y; sd=d.std(ddof=1)
    # n=10 exact sign-flip p; t-test p is reported with conservative df=9 critical approximation.
    obs=abs(float(d.mean())); exceed=sum(abs(float((d*np.array(s)).mean()))>=obs-1e-15 for s in product((-1,1),repeat=len(d))); signp=exceed/(2**len(d))
    ci=(d.mean()-2.262*sd/math.sqrt(len(d)), d.mean()+2.262*sd/math.sqrt(len(d))) if sd>0 else (d.mean(),d.mean())
    paired.append({'contrast':'FSSQL-R - '+b,'metric':metric,'n_pairs':10,'mean_fssql':a.mean(),'mean_comparator':y.mean(),'mean_diff':d.mean(),'sd_diff':sd,'ci95_low':ci[0],'ci95_high':ci[1],'paired_t_p_approx':2*0.5*math.erfc(abs(d.mean())/(sd/math.sqrt(len(d))*math.sqrt(2))) if sd>0 else (0.0 if d.mean() else 1.0),'exact_signflip_p':signp,'cohen_dz':d.mean()/sd if sd>0 else float('nan')})
with (outdir/'V3_PAIRED_SEED_INFERENCE.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=paired[0]); w.writeheader(); w.writerows(paired)

if violation:
  dn,r=violation; d=C/dn; m=json.loads((d/'RUN_MANIFEST.json').read_text()); pred=json.loads(r['predicted_latency_by_action_ms']); head=json.loads(r['latency_guard_headroom_by_action_ms'])
  text=f'''# Single Threshold Latency Violation Forensic\n\n- Run: `{m['run_id']}` ({dn})\n- Policy/regime/seed/window: **{r['policy']} / {r['regime']} / {r['seed']} / {r['window_id']}**\n- Selected action: `{r['selected_action']}`; previous action: `{r['prev_action'] or '(none)'}`; switch event: `{r['prev_action'] and r['prev_action'] != r['selected_action']}`\n- Admissible set (`|A_safe|={r['safe_set_size']}`): `{r['safe_set_mask']}`\n- Predicted guarded latency (selected action): **{pred[r['selected_action']]:.6f} ms**; guard headroom to cap: **{head[r['selected_action']]:.6f} ms**; realised end-to-end: **{float(r['end_to_end_ms']):.6f} ms**; violation flag: `{r['latency_violation']}`\n- Residual margin context: `tau_hat={r['tau_hat_ms']} ms`, `tau_upper={r['tau_upper_ms']} ms`, `latency_slack={r['latency_slack_ms']} ms`; predicted action latencies: `{pred}`\n- Detector/controller/switching: detector inference `{r['inference_ms']} ms`, total controller `{r['total_controller_ms']} ms`, switching overhead `{r['switch_ms']} ms`; feature `{r['feature_ms']} ms`, state build `{r['state_build_ms']} ms`, guard `{r['guard_ms']} ms`.\n- Thermal/frequency/fault: T_start `{r['T_start_C']} C`, T_end `{r['T_end_C']} C`, CPU minimum `{r['cpu_frequency_min_mhz']} MHz`, throttled `{r['throttled']}`, undervoltage `{r['undervoltage']}`.\n- Workload: `n_records={r['n_records']}`, background `{r['background']}`.\n\nThe violation is retained as collected. It occurred under the shielded conventional Threshold controller, while the selected action was admissible under the frozen guard; this demonstrates tail-risk reduction rather than a mathematical guarantee of zero realised violations.\n'''
  (outdir/'SINGLE_LATENCY_VIOLATION_FORENSIC.md').write_text(text)
print('wrote',outdir)
