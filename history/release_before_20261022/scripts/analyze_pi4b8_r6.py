#!/usr/bin/env python3
import csv,json,math,statistics
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EV=ROOT/'evidence/pi4b8_corrected/v5_r6_execution'; OUT=ROOT/'results/pi4b8_r6'; OUT.mkdir(parents=True,exist_ok=True)
plan=json.load(open(ROOT/'evidence/pi4b8_corrected/v5_r6_authorization/PI4B8_CORRECTED_V5_FORMAL_PLAN.json'))
final=json.load(open(EV/'PI4B8_FORMAL_EVIDENCE_FINAL_VALIDATION.json'))
rows=[]
for c in final['cells']:
    p=Path(c['evidence_directory']); wr=list(csv.DictReader(open(p/'pi_windows.csv')))
    def f(k): return np.array([float(x[k]) for x in wr])
    tp,fp,fn,tn=[sum(int(x[k]) for x in wr) for k in ('tp','fp','fn','tn')]; n=tp+fp+fn+tn
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0; prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0; fpr=fp/(fp+tn) if fp+tn else 0
    actions={}
    for x in wr: actions[x['selected_action']]=actions.get(x['selected_action'],0)+1
    head=[]
    for x in wr:
      try: head.append(float(json.loads(x['latency_headroom_by_action_ms'])[x['selected_action']]))
      except Exception: pass
    power=json.load(open(p/'RUN_VALIDATION.json'))['powerz']
    out={'run_id':c['accepted_run_id'],'cell_run_id':c['cell_run_id'],'execution_index':c['execution_index'],'policy':c['policy'],'condition':c['condition'],'seed':c['seed'],'replacement':c['replacement'],'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':prec,'recall':rec,'f1':f1,'fpr':fpr,'reward_cumulative':float(f('reward').sum()),'reward_mean':float(f('reward').mean())}
    for k in ('reward_security','reward_energy','reward_backlog','reward_switch','e2e_ms','controller_ms','switching_overhead_ms','temperature_C','cpu_frequency_mhz_min','safe_set_size','fallback','preprocessing_ms','telemetry_ms','state_construction_ms','latency_guard_ms','thermal_guard_ms','admissible_set_ms','policy_selection_ms','q_lookup_ms','detector_inference_ms','scoring_ms'):
      a=f(k); out[k+'_mean']=float(a.mean());
      if k=='e2e_ms':
       out['e2e_p50_ms']=float(np.percentile(a,50)); out['e2e_p95_ms']=float(np.percentile(a,95)); out['e2e_p99_ms']=float(np.percentile(a,99)); out['e2e_max_ms']=float(a.max()); out['latency_violations']=int((a>40).sum())
    out['switch_count']=int(sum(int(x['switched']) for x in wr)); out['switch_rate']=out['switch_count']/len(wr); out['selected_headroom_ms_mean']=float(np.mean(head)) if head else float('nan'); out['action_composition']=json.dumps(actions,sort_keys=True); out.update({'energy_J':power['energy_J'],'energy_Wh':power['energy_J']/3600,'mean_power_W':power['mean_power_W'],'peak_power_W':power['peak_observed_with_interpolated_boundaries_W'],'energy_per_window_J':power['energy_J']/len(wr),'energy_per_1000_records_J':power['energy_J']/(sum(int(x['records_processed']) for x in wr)/1000),'energy_per_correct_attack_J':power['energy_J']/tp if tp else float('nan')})
    rows.append(out)
fields=list(rows[0]);
with open(OUT/'PI4B8_R6_RUN_LEVEL_METRICS.csv','w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)
def paired(a,b):
 d=np.array(a)-np.array(b); n=len(d); obs=abs(float(d.mean())); exceed=0; total=2**n
 for mask in range(total):
  s=np.array([1 if mask>>i&1 else -1 for i in range(n)]); exceed += abs(float((d*s).mean())) >= obs-1e-15
 sd=float(d.std(ddof=1)); boots=[]; rng=np.random.default_rng(20261022)
 for _ in range(2000): boots.append(float(rng.choice(d,n).mean()))
 return {'n_pairs':n,'mean_diff':float(d.mean()),'sd_diff':sd,'ci95_low':float(np.quantile(boots,.025)),'ci95_high':float(np.quantile(boots,.975)),'exact_signflip_p':exceed/total,'cohen_dz':float(d.mean()/sd) if sd else float('nan')}
metrics=[k for k in fields if k not in ('run_id','cell_run_id','execution_index','policy','condition','seed','replacement','action_composition') and not k.startswith('tp') and not k.startswith('fp') and not k.startswith('fn') and not k.startswith('tn')]
prow=[]
for cond in sorted(set(x['condition'] for x in rows)):
 sub=[x for x in rows if x['condition']==cond]; by={p:{x['seed']:x for x in sub if x['policy']==p} for p in set(x['policy'] for x in sub)}
 for m in metrics:
  if not all(x in by.get('FSSQL-R',{}) and x in by.get('Safe-Greedy',{}) for x in []): pass
  seeds=sorted(set(by.get('FSSQL-R',{}))&set(by.get('Safe-Greedy',{}))); a=[float(by['FSSQL-R'][s][m]) for s in seeds]; b=[float(by['Safe-Greedy'][s][m]) for s in seeds]
  if not all(math.isfinite(x) for x in a+b): continue
  prow.append({'condition':cond,'metric':m,'method_a':'FSSQL-R','method_b':'Safe-Greedy',**paired(a,b)})
with open(OUT/'PI4B8_R6_PAIRED_FSSQL_R_VS_SAFE_GREEDY.csv','w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=list(prow[0])); w.writeheader(); w.writerows(prow)
# All policy contrasts use the same matched seeds and condition-level pairing.
allpairs=[]
policies=['FSSQL-R','Safe-Greedy','Threshold','Static-Light']
for cond in sorted(set(x['condition'] for x in rows)):
 sub=[x for x in rows if x['condition']==cond]; by={p:{x['seed']:x for x in sub if x['policy']==p} for p in policies}
 for ia,a_name in enumerate(policies):
  for b_name in policies[ia+1:]:
   seeds=sorted(set(by.get(a_name,{}))&set(by.get(b_name,{})))
   for m in metrics:
    a=[float(by[a_name][s][m]) for s in seeds]; b=[float(by[b_name][s][m]) for s in seeds]
    if not a or not all(math.isfinite(x) for x in a+b): continue
    allpairs.append({'condition':cond,'metric':m,'method_a':a_name,'method_b':b_name,**paired(a,b)})
with open(OUT/'PI4B8_R6_PAIRED_ALL_POLICY_CONTRASTS.csv','w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=list(allpairs[0])); w.writeheader(); w.writerows(allpairs)
print('WROTE',len(rows),'runs',len(prow),'paired rows')
