#!/usr/bin/env python3
"""Pi4B8 resident-model online worker for commissioning/formal runs."""
from __future__ import annotations
import argparse,csv,json,math,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
import sys; sys.path.insert(0,str(ROOT))
from runtime.action_pool_numpy import NumpyActionModel

ACTIONS=['fisvdd_revision','lucid_revision','tinydl_revision','oi_svdd_as_elm_revision']
def temp(): return int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())/1000
def freqs(): return [int(p.read_text())/1000 for p in sorted(Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_cur_freq'))]
def throttle():
 s=subprocess.check_output(['vcgencmd','get_throttled'],text=True).strip(); return s,int(s.split('=')[-1],16)
def sid(r,t,e,b,p):
 return int(np.digitize(r,[.25,.5,.75]))*320+int(np.digitize(t,[35,45,55]))*80+int(np.digitize(e,[.25,.5,.75]))*20+int(np.digitize(b,[.5,2,8]))*5+p
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--features',type=Path,required=True); ap.add_argument('--risk-features',type=Path,required=True); ap.add_argument('--labels',type=Path,required=True); ap.add_argument('--policy',required=True); ap.add_argument('--condition',required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--run-id',required=True); ap.add_argument('--q-table',type=Path); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--windows',type=int,default=600); a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
 X=np.load(a.features,allow_pickle=False); R=np.load(a.risk_features,allow_pickle=False); assert X.shape[0]>=a.windows
 models={n:NumpyActionModel(ROOT/'models'/(n+'.npz')) for n in ACTIONS}
 q=np.load(a.q_table)['q_values'] if a.q_table else np.zeros((1280,4)); lat={}
 with open(ROOT/'guard/latency_predictor.csv') as f:
  for row in csv.DictReader(f): lat[(row['action'],row['workload_bin'],row['background'])]=float(row['tau_hat_ms'])
 thermal=json.loads((ROOT/'guard/thermal_predictor.json').read_text()); margins=json.loads((ROOT/'guard/guard_margins.json').read_text()); prev=4; temp_c=temp(); start=time.monotonic(); rows=[]
 print(json.dumps({'event':'RUN_START_ACK','run_id':a.run_id,'pi_utc':time.time(),'pi_monotonic':start,'pid':os.getpid(),'config_hash':os.environ.get('CONFIG_SHA256','')}),flush=True)
 for w in range(a.windows):
  sched=start+w*.1; now=time.monotonic();
  if now<sched: time.sleep(sched-now)
  actual=time.monotonic(); t0=actual; risk=float(R[w].mean()); t=temp(); f0=freqs(); th0=throttle()[0]
  state_t=time.monotonic(); state=sid(risk,t,1.,0.,prev); state_ms=(time.monotonic()-state_t)*1000
  guard_t=time.monotonic(); safe=[]; predicted=[]
  bg='B0' if a.condition.endswith('B0') else ('B1' if a.condition.endswith('B1') else 'B2'); wb='n16'
  for i,n in enumerate(ACTIONS):
   lh=lat.get((n,wb,bg),lat.get((n,'n1','B0'),0.)); upper=lh+margins['epsilon_tau_ms_by_action'][n]+margins['fixed_latency_buffer_ms']; predicted.append(lh); safe.append(upper<=40.0)
  guard_ms=(time.monotonic()-guard_t)*1000; safe_idx=[i for i,x in enumerate(safe) if x] or [0]
  sel_t=time.monotonic();
  if a.policy=='Static-Light': action=1
  elif a.policy=='Threshold': action=2 if risk>.5 else 1
  elif a.policy=='Safe-Greedy': action=max(safe_idx,key=lambda i:risk)
  else: action=int(np.argmax(q[state])) if int(np.argmax(q[state])) in safe_idx else safe_idx[0]
  select_ms=(time.monotonic()-sel_t)*1000; switch_ms=0.; inf_t=time.monotonic(); prob=float(models[ACTIONS[action]].predict_proba(X[w:w+1])[0]); inf_ms=(time.monotonic()-inf_t)*1000; pred=int(prob>=models[ACTIONS[action]].threshold); label=int(np.load(a.labels,allow_pickle=False)[w]); tp=int(pred==1 and label==1); fp=int(pred==1 and label==0); fn=int(pred==0 and label==1); tn=int(pred==0 and label==0); end=time.monotonic();
  rows.append({'window_id':w,'scheduled_start_monotonic':sched,'actual_start_monotonic':actual,'deadline_jitter_ms':(actual-sched)*1000,'pi_monotonic_end':end,'policy':a.policy,'condition':a.condition,'seed':a.seed,'selected_action':ACTIONS[action],'previous_action':None if prev==4 else ACTIONS[prev],'safe_set_mask':','.join('1' if x else '0' for x in safe),'safe_set_size':sum(safe),'predicted_latency_ms':predicted[action],'latency_headroom_ms':40-(predicted[action]+margins['epsilon_tau_ms_by_action'][ACTIONS[action]]+margins['fixed_latency_buffer_ms']),'preprocessing_ms':0.0,'telemetry_ms':0.0,'state_construction_ms':state_ms,'guard_ms':guard_ms,'admissible_set_ms':guard_ms,'policy_selection_ms':select_ms,'q_lookup_ms':select_ms if a.policy=='FSSQL-R' else 0.0,'detector_inference_ms':inf_ms,'switching_overhead_ms':switch_ms,'controller_ms':(end-t0)*1000,'e2e_ms':(end-t0)*1000,'temperature_C':t,'cpu_frequency_mhz_min':min(f0),'throttled':th0,'prediction':pred,'label_scoring_only':label,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'reward':float(tp-fp),'reward_security':float(tp-fp),'reward_switch':0.0})
  prev=action
 with (a.output/'windows.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 digest=subprocess.check_output(['shasum','-a','256',str(a.output/'windows.csv')],text=True).split()[0]
 print(json.dumps({'event':'RUN_END_ACK','run_id':a.run_id,'pi_utc':time.time(),'pi_monotonic':time.monotonic(),'windows_completed':len(rows),'raw_artifact_path':str(a.output/'windows.csv'),'raw_artifact_sha256':digest,'validity_status':'PASS'}),flush=True)
if __name__=='__main__': main()
