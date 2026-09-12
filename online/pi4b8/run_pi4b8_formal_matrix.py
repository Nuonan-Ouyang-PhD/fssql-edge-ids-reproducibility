#!/usr/bin/env python3
"""Execute the frozen Pi4B8 formal matrix without adaptive decisions."""
import csv, json, hashlib, subprocess, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'evidence/pi4b8_formal_matrix'; OUT.mkdir(parents=True,exist_ok=True)
import argparse
ap=argparse.ArgumentParser(); ap.add_argument('--start',type=int,default=41); args=ap.parse_args()
policies=['FSSQL-R','Safe-Greedy','Threshold','Static-Light']; conds=[('n16/B0','natural'),('n16/B1','robustness_B1'),('n16/B2','robustness_B2')]; seeds=[5101,5102,5103,5104,5105]
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
order=[]
for seed in seeds:
 rng=np.random.default_rng(20260910+seed); block=[(p,c) for c,_ in conds for p in policies]; rng.shuffle(block); order += [(seed,p,c,cl) for p,c in block for _,cl in [next(x for x in conds if x[0]==c)]]
for i,(seed,policy,condition,conddir) in enumerate(order,1):
 if i < args.start: continue
 run_id=f'pi4b8_formal_{i:02d}_{policy.lower().replace("-", "_")}_{condition.replace("/", "_")}_s{seed}'
 out=OUT/run_id; out.mkdir(exist_ok=False)
 meter=ROOT/'online/pi4b8/powerz_hid_capture.py'; py=str(ROOT/'.venv/bin/python'); meter_file=out/'POWERZ_raw.csv'; meter_start=time.monotonic()
 mp=subprocess.Popen([py,str(meter),'--output',str(meter_file),'--seconds','80','--sps','10'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True); time.sleep(5)
 qarg=f' --q-table q_tables/fssql_r_seed_{seed}.npz' if policy=='FSSQL-R' else ''
 remote=f'cd /home/pi/tsusc_major_revision/pi4b8_formal_runtime_v4 && python3 pi4b8_online_worker.py --features inputs/{conddir}/{seed}/features.npy --risk-features inputs/{conddir}/{seed}/risk_features.npy --labels scoring/{conddir}/{seed}/labels.npy --policy {policy} --condition {condition} --seed {seed} --run-id {run_id} --output {run_id} --windows 600{qarg}'
 req=time.monotonic(); proc=subprocess.Popen(['ssh','pi@pi4b8g.local',remote],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1); lines=[]; first=proc.stdout.readline(); ack=time.monotonic(); lines.append(first); end_line=''; end_ack=None
 while True:
  line=proc.stdout.readline()
  if line: lines.append(line)
  if '"event": "RUN_END_ACK"' in line: end_line=line; end_ack=time.monotonic(); break
  if proc.poll() is not None and not line: break
 proc.wait(); (out/'worker.stdout').write_text(''.join(lines)); (out/'worker.stderr').write_text(proc.stderr.read())
 subprocess.run(['scp','-r',f'pi@pi4b8g.local:/home/pi/tsusc_major_revision/pi4b8_formal_runtime_v4/{run_id}',str(out)],check=False)
 time.sleep(5); meter_end=time.monotonic()
 try: mp.wait(timeout=15)
 except subprocess.TimeoutExpired: mp.terminate(); mp.wait()
 if end_ack is None or not meter_file.exists(): status='INVALID'
 else: status='VALID_CANDIDATE'
 (out/'HANDSHAKE.json').write_text(json.dumps({'run_id':run_id,'seed':seed,'policy':policy,'condition':condition,'meter_capture_start_mac_monotonic':meter_start,'start_request_mac_monotonic':req,'start_ack_receive_mac_monotonic':ack,'end_ack_receive_mac_monotonic':end_ack,'meter_capture_end_mac_monotonic':meter_end,'pi_start_ack':json.loads(first) if first else None,'pi_end_ack':json.loads(end_line) if end_line else None,'status':status},indent=2)+'\n')
 print(f'{i}/60 {run_id} {status}',flush=True)
print('60-RUN MATRIX EXECUTION COMPLETE')
