#!/usr/bin/env python3
"""Execute one pre-authorized technical replacement cell.

This wrapper does not alter the frozen formal plan or any scientific asset. It
uses the existing contemporaneous execute_cell path with a new attempt ID.
"""
import argparse, json
from pathlib import Path
from pi4b8_corrected_v5_execution_orchestrator import (
    execute_cell, load_json, sha256, validate_materialized_package,
    runtime_dependency_contract, revalidate_complete_cell, _write_json_new,
)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True); ap.add_argument('--index', type=int, required=True)
    ap.add_argument('--attempt-id', required=True); ap.add_argument('--package-root', type=Path, required=True)
    ap.add_argument('--package-manifest', type=Path, required=True); ap.add_argument('--evidence-root', type=Path, required=True)
    ap.add_argument('--ssh-target', required=True); ap.add_argument('--mac-python', required=True)
    args=ap.parse_args(); plan=load_json(args.plan); run=plan['runs'][args.index-1].copy()
    original=run['run_id']; run['run_id']=args.attempt_id
    run['remote_output']=run['remote_output'].rsplit('/',1)[0]+'/'+args.attempt_id
    run['worker_argv']=[args.attempt_id if x==original else (run['remote_output'] if x==plan['runs'][args.index-1]['remote_output'] else x) for x in run['worker_argv']]
    # Replace the old run-id and remote output arguments deterministically.
    old=plan['runs'][args.index-1]
    run['worker_argv']=[args.attempt_id if x==original else run['remote_output'] if x==old['remote_output'] else x for x in run['worker_argv']]
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    package=validate_materialized_package(args.package_root,args.package_manifest)
    _, dep, _=runtime_dependency_contract(args.package_root,package)
    prereg={"status":"AUTHORIZED_TECHNICAL_REPLACEMENT","original_run_id":original,"replacement_run_id":args.attempt_id,"execution_index":args.index,"policy":run['policy'],"condition":run['condition'],"seed":run['seed'],"reason":"original attempt invalidated solely for first-window temperature gate","package_manifest_sha256":sha256(args.package_manifest),"statement":"authorized before inspecting scientific outcomes"}
    _write_json_new(args.evidence_root/'TECHNICAL_REPLACEMENT_PREREGISTRATION.json',prereg)
    result=execute_cell(args.package_root,args.package_manifest,run,evidence_root=args.evidence_root,ssh_target=args.ssh_target,mac_python=args.mac_python)
    checked=revalidate_complete_cell(args.evidence_root/args.attempt_id,run,package_root=args.package_root,manifest_path=args.package_manifest,dependency=dep)
    _write_json_new(args.evidence_root/'TECHNICAL_REPLACEMENT_VALIDATION.json',checked)
    print('TECHNICAL_REPLACEMENT_VALID_COMPLETE')
    print(json.dumps(checked,sort_keys=True))
if __name__=='__main__': main()
