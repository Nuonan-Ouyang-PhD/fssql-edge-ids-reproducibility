#!/usr/bin/env python3
"""Mac-side commissioning orchestrator for the Pi4B8 worker and KM003C."""
from __future__ import annotations
import argparse,csv,datetime,hashlib,json,subprocess,time,sys,select
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--run-id',default='pi4b8_commissioning_v4'); p.add_argument('--seed',type=int,default=5101); p.add_argument('--windows',type=int,default=600); a=p.parse_args(); out=ROOT/'evidence/pi4b8_commissioning'/a.run_id; out.mkdir(parents=True,exist_ok=False)
 py=str(ROOT/'.venv/bin/python'); gate=subprocess.run([py,str(ROOT/'online/pi4b8/pi4b8_formal_runner.py'),'--commissioning'],capture_output=True,text=True); (out/'gate.txt').write_text(gate.stdout+gate.stderr)
 if gate.returncode: raise SystemExit('lineage gate failed')
 meter_out=out/'POWERZ_raw.csv'; meter=ROOT/'online/pi4b8/powerz_hid_capture.py'; meter_start=time.monotonic(); mp=subprocess.Popen([py,str(meter),'--output',str(meter_out),'--seconds',str(a.windows//10+20),'--sps','10'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True); time.sleep(5)
 remote=f"cd /home/pi/tsusc_major_revision/pi4b8_formal_runtime_v4 && python3 pi4b8_online_worker.py --features inputs/natural/5101/features.npy --risk-features inputs/natural/5101/risk_features.npy --labels scoring/natural/5101/labels.npy --policy Static-Light --condition n16/B0 --seed {a.seed} --run-id {a.run_id} --q-table q_tables/fssql_r_seed_5101.npz --output {a.run_id} --windows {a.windows}"
 request_ts=time.monotonic(); run=subprocess.Popen(['ssh','pi@pi4b8g.local',remote],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
 start_line=run.stdout.readline(); start_ack=time.monotonic(); end_line=''; end_ack=None; lines=[start_line]
 if not start_line: raise SystemExit('COMMISSIONING_FAIL: missing RUN_START_ACK')
 while True:
  line=run.stdout.readline()
  if line: lines.append(line)
  if line.startswith('{"event": "RUN_END_ACK"'):
   end_line=line; end_ack=time.monotonic(); break
  if run.poll() is not None and not line: break
 run.wait(); (out/'worker.stdout').write_text(''.join(lines)); (out/'worker.stderr').write_text(run.stderr.read())
 meter_end_request=None
 time.sleep(5); meter_end_request=time.monotonic()
 try: mp.wait(timeout=15)
 except subprocess.TimeoutExpired: mp.terminate(); mp.wait()
 end_meter=meter_end_request
 subprocess.run(['scp','-r',f'pi@pi4b8g.local:/home/pi/tsusc_major_revision/pi4b8_formal_runtime_v4/{a.run_id}',str(out)],check=False)
 try:
  st=json.loads(start_line); en=json.loads(end_line)
  handshake={'run_id':a.run_id,'meter_capture_start_mac_monotonic':meter_start,'start_request_mac_monotonic':request_ts,'start_ack_receive_mac_monotonic':start_ack,'run_start_utc':st.get('pi_utc'),'run_start_monotonic':st.get('pi_monotonic'),'end_ack_receive_mac_monotonic':end_ack,'run_end_utc':en.get('pi_utc'),'run_end_monotonic':en.get('pi_monotonic'),'meter_capture_end_mac_monotonic':end_meter,'start_request_to_ack_ms':(start_ack-request_ts)*1000,'end_ack_to_meter_stop_ms':(end_meter-end_ack)*1000}
  (out/'HANDSHAKE.json').write_text(json.dumps(handshake,indent=2)+'\n')
 except Exception as e: raise SystemExit(f'COMMISSIONING_FAIL: malformed ACK: {e}')
 if run.returncode or not meter_out.exists(): raise SystemExit('COMMISSIONING_FAIL: worker or KM003C capture failed')
 print('COMMISSIONING_EXECUTION_COMPLETE')
if __name__=='__main__': main()
