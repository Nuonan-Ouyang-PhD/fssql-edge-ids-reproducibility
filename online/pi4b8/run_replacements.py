import json, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; E=ROOT/'evidence/pi4b8_formal_matrix'; py=str(ROOT/'.venv/bin/python')
cells=[('pi4b8_formal_40_safe_greedy_n16_B2_s5104','Safe-Greedy','n16/B2','robustness_B2',5104),('pi4b8_formal_51_fssql_r_n16_B2_s5105','FSSQL-R','n16/B2','robustness_B2',5105)]
for orig,policy,condition,cd,seed in cells:
 rid=orig+'_attempt2'; out=E/rid; out.mkdir(parents=True,exist_ok=False); meter_file=out/'POWERZ_raw.csv'; meter_start=time.monotonic(); mp=subprocess.Popen([py,str(ROOT/'online/pi4b8/powerz_hid_capture.py'),'--output',str(meter_file),'--seconds','80','--sps','10'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True); time.sleep(5)
 q=f' --q-table q_tables/fssql_r_seed_{seed}.npz' if policy=='FSSQL-R' else ''
 remote=f'cd /home/pi/tsusc_major_revision/pi4b8_formal_runtime_v4 && python3 pi4b8_online_worker.py --features inputs/{cd}/{seed}/features.npy --risk-features inputs/{cd}/{seed}/risk_features.npy --labels scoring/{cd}/{seed}/labels.npy --policy {policy} --condition {condition} --seed {seed} --run-id {rid} --output {rid} --windows 600{q}'
 req=time.monotonic(); p=subprocess.Popen(['ssh','pi@pi4b8g.local',remote],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1); first=p.stdout.readline(); ack=time.monotonic(); lines=[first]; end=None; endack=None
 while True:
  line=p.stdout.readline()
  if line: lines.append(line)
  if '"event": "RUN_END_ACK"' in line: end=line; endack=time.monotonic(); break
  if p.poll() is not None and not line: break
 p.wait(); (out/'worker.stdout').write_text(''.join(lines)); (out/'worker.stderr').write_text(p.stderr.read()); time.sleep(5); meter_end=time.monotonic()
 try: mp.wait(timeout=15)
 except subprocess.TimeoutExpired: mp.terminate(); mp.wait()
 (out/'HANDSHAKE.json').write_text(json.dumps({'run_id':rid,'original_run_id':orig,'policy':policy,'condition':condition,'seed':seed,'meter_capture_start_mac_monotonic':meter_start,'start_request_mac_monotonic':req,'start_ack_receive_mac_monotonic':ack,'end_ack_receive_mac_monotonic':endack,'meter_capture_end_mac_monotonic':meter_end,'pi_start_ack':json.loads(first),'pi_end_ack':json.loads(end) if end else None},indent=2)+'\n')
 if p.returncode or not meter_file.exists(): raise SystemExit(f'replacement failed: {rid}')
 subprocess.run(['scp','-r',f'pi@pi4b8g.local:/home/pi/tsusc_major_revision/pi4b8_formal_runtime_v4/{rid}',str(out)],check=True)
 print(rid,'COMPLETE',flush=True)
