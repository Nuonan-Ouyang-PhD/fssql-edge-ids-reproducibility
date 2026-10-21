#!/usr/bin/env python3
"""Resume untouched formal cells after an explicitly authorized replacement."""
import argparse, json
from pathlib import Path
from pi4b8_corrected_v5_execution_orchestrator import execute_cell, load_json, sha256, validate_materialized_package, runtime_dependency_contract, revalidate_complete_cell

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--plan',type=Path,required=True); ap.add_argument('--start',type=int,required=True); ap.add_argument('--end',type=int,required=True); ap.add_argument('--package-root',type=Path,required=True); ap.add_argument('--package-manifest',type=Path,required=True); ap.add_argument('--evidence-root',type=Path,required=True); ap.add_argument('--ssh-target',required=True); ap.add_argument('--mac-python',required=True); args=ap.parse_args()
    plan=load_json(args.plan); package=validate_materialized_package(args.package_root,args.package_manifest); _,dep,_=runtime_dependency_contract(args.package_root,package)
    reg=args.evidence_root/'FORMAL_REPLACEMENT_CONTINUATION_REGISTRY.jsonl'; args.evidence_root.mkdir(parents=True,exist_ok=True)
    with reg.open('a',encoding='utf-8') as out:
      for i in range(args.start,args.end+1):
        run=plan['runs'][i-1]; path=args.evidence_root/run['run_id']
        if path.exists(): raise RuntimeError(f'non-empty target {path}')
        out.write(json.dumps({'event':'EXECUTION_STARTED','execution_index':i,'run_id':run['run_id'],'policy':run['policy'],'condition':run['condition'],'seed':run['seed']},sort_keys=True)+'\n'); out.flush()
        try:
          execute_cell(args.package_root,args.package_manifest,run,evidence_root=args.evidence_root,ssh_target=args.ssh_target,mac_python=args.mac_python)
          checked=revalidate_complete_cell(path,run,package_root=args.package_root,manifest_path=args.package_manifest,dependency=dep)
          out.write(json.dumps({'event':'VALID_COMPLETE','execution_index':i,'run_id':run['run_id'],'run_manifest_sha256':checked['run_manifest_sha256']},sort_keys=True)+'\n'); out.flush(); print(f'{i}/60 VALID_COMPLETE',flush=True)
        except Exception as e:
          out.write(json.dumps({'event':'EXECUTION_INVALID_TECHNICAL','execution_index':i,'run_id':run['run_id'],'reason':str(e)},sort_keys=True)+'\n'); out.flush(); raise
if __name__=='__main__': main()
