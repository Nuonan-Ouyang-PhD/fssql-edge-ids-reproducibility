#!/usr/bin/env python3
"""Recompute the corrected Pi 4B R6 campaign from the frozen accepted runs.

No training, new hardware measurement or official-test access is performed.
The original results/pi4b8_r6 directory is not overwritten.
"""
from __future__ import annotations
import argparse, csv, hashlib, itertools, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from online.pi4b8.powerz_evidence_validator import validate_powerz_evidence
POLICIES = ('FSSQL-R', 'Safe-Greedy', 'Threshold', 'Static-Light')
CONDITIONS = ('n16/B0', 'n16/B1', 'n16/B2')
SEEDS = tuple(range(5101,5106))
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261022
IDENTITY_AND_COUNT_FIELDS = {'run_id','cell_run_id','execution_index','policy','condition','seed','replacement','action_composition','tp','fp','fn','tn','windows','records_processed'}

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def json_write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path outside repository: {relative}')
    if not path.is_file() and not path.is_dir():
        raise FileNotFoundError(path)
    return path

def validate_cells(cells: list[dict]) -> None:
    keys=[(c['policy'],c['condition'],int(c['seed'])) for c in cells]
    expected=set(itertools.product(POLICIES,CONDITIONS,SEEDS))
    if len(keys)!=60 or len(set(keys))!=60 or set(keys)!=expected:
        raise ValueError('Expected exactly four policies x three conditions x five seeds')
    if len({c['accepted_run_id'] for c in cells})!=60:
        raise ValueError('Duplicate accepted run ID')

def metric_columns(frame: pd.DataFrame) -> list[str]:
    """Exclude exact identity/count names, never prefixes (which exclude FPR)."""
    result=[c for c in frame.columns if c not in IDENTITY_AND_COUNT_FIELDS and pd.api.types.is_numeric_dtype(frame[c])]
    if 'fpr' not in result:
        raise ValueError('FPR is a required paired endpoint')
    return result

def paired_statistics(a, b) -> dict:
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    if a.shape!=b.shape or a.ndim!=1 or len(a)!=5 or not np.isfinite(np.r_[a,b]).all():
        raise ValueError('Paired comparison requires five finite matched values')
    d=a-b; n=len(d); mean=float(d.mean()); sd=float(d.std(ddof=1))
    signs=np.asarray(list(itertools.product((-1,1),repeat=n)))
    null_means=(signs*d).mean(axis=1)
    p=float(np.mean(np.abs(null_means)>=abs(mean)-1e-15))
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    boot=rng.choice(d, size=(BOOTSTRAP_REPLICATES,n), replace=True).mean(axis=1)
    lo,hi=np.quantile(boot,[.025,.975])
    return {'n_pairs':n,'mean_a':float(a.mean()),'mean_b':float(b.mean()),
        'mean_diff':mean,'median_diff':float(np.median(d)),'sd_diff':sd,
        'ci95_low':float(lo),'ci95_high':float(hi),'exact_signflip_p':p,
        'cohen_dz':mean/sd if sd>0 else None}

def derive_run(root: Path, c: dict) -> tuple[dict,dict,pd.DataFrame]:
    p=resolve(root,c['evidence_directory']); wfile=p/'pi_windows.csv'; meter=p/'POWERZ_raw.csv'
    for file,field in [(wfile,'pi_raw_sha256'),(meter,'powerz_raw_sha256'),(p/'RUN_MANIFEST.json','run_manifest_sha256')]:
        if sha(file)!=c[field]: raise ValueError(f'Frozen hash mismatch: {file}')
    w=pd.read_csv(wfile)
    if len(w)!=600 or w['window_id'].tolist()!=list(range(600)):
        raise ValueError(f'Window mismatch: {c["accepted_run_id"]}')
    for field in ('run_id','policy','condition','seed'):
        exp=c['accepted_run_id'] if field=='run_id' else c[field]
        if not (w[field].astype(str)==str(exp)).all(): raise ValueError(f'Identity mismatch: {field}')
    if not (w['records_processed']==16).all(): raise ValueError('Not true n16')
    if not (w[['tp','fp','fn','tn']].sum(axis=1)==16).all(): raise ValueError('Scoring count mismatch')
    if not np.allclose(w['reward'],w[['reward_security','reward_energy','reward_backlog','reward_switch']].sum(axis=1),rtol=0,atol=1e-12):
        raise ValueError('Reward decomposition mismatch')
    for raw in w['batch_indices']:
        indices=json.loads(raw)
        if len(indices)!=16 or len(set(indices))!=16: raise ValueError('Invalid batch indices')
    h=json.loads((p/'HANDSHAKE.json').read_text())
    power=validate_powerz_evidence(meter,h,run_id=c['accepted_run_id'])
    archived=json.loads((p/'RUN_VALIDATION.json').read_text())['powerz']
    if not math.isclose(power['energy_J'],archived['energy_J'],rel_tol=1e-12,abs_tol=1e-9):
        raise ValueError('Energy recomputation mismatch')
    tp,fp,fn,tn=(int(w[k].sum()) for k in ('tp','fp','fn','tn'))
    denom=2*tp+fp+fn
    out={k:c[k] for k in ('cell_run_id','execution_index','policy','condition','seed','replacement')}
    out.update(run_id=c['accepted_run_id'],windows=len(w),records_processed=int(w.records_processed.sum()),tp=tp,fp=fp,fn=fn,tn=tn,
        precision=tp/(tp+fp) if tp+fp else np.nan,recall=tp/(tp+fn) if tp+fn else np.nan,
        f1=2*tp/denom if denom else np.nan,fpr=fp/(fp+tn) if fp+tn else np.nan,
        reward_cumulative=float(w.reward.sum()),reward_mean=float(w.reward.mean()))
    numeric=['reward_security','reward_energy','reward_backlog','reward_switch','e2e_ms','controller_ms',
        'switching_overhead_ms','temperature_C','cpu_frequency_mhz_min','safe_set_size','fallback','preprocessing_ms',
        'risk_proxy_ms','telemetry_ms','state_construction_ms','latency_guard_ms','thermal_guard_ms','admissible_set_ms',
        'policy_selection_ms','q_lookup_ms','detector_inference_ms','post_action_telemetry_ms','scoring_ms']
    for k in numeric:
        vals=w[k].to_numpy(float)
        if not np.isfinite(vals).all(): raise ValueError(f'Nonfinite {k}')
        out[k+'_mean']=float(vals.mean())
    for prefix,k in [('e2e','e2e_ms'),('controller','controller_ms')]:
        for pct in (50,95,99): out[f'{prefix}_p{pct}_ms']=float(np.percentile(w[k],pct))
        out[f'{prefix}_max_ms']=float(w[k].max())
    out.update(latency_violations=int((w.e2e_ms>40).sum()),thermal_violations=int((w.temperature_end_C>82).sum()),
        switch_count=int(w.switched.sum()),switch_rate=float(w.switched.mean()),temperature_max_C=float(w.temperature_C.max()),
        minimum_frequency_mhz=float(w.cpu_frequency_mhz_min.min()),
        selected_headroom_ms_mean=float(np.mean([json.loads(row.latency_headroom_by_action_ms)[row.selected_action] for row in w.itertuples()])),
        action_composition=json.dumps(w.selected_action.value_counts().sort_index().to_dict(),sort_keys=True),
        energy_J=power['energy_J'],energy_Wh=power['energy_J']/3600,
        mean_power_W=power['mean_power_W'],peak_power_W=power['peak_observed_with_interpolated_boundaries_W'],
        energy_per_window_J=power['energy_J']/600,energy_per_1000_records_J=power['energy_J']/9.6,
        energy_per_correct_attack_J=power['energy_J']/tp if tp else np.nan,
        ack_duration_s=power['duration_s'],pre_roll_s=power['pre_roll_s'],post_roll_s=power['post_roll_s'])
    return out,power,w

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args(); root=args.root.resolve()
    out=args.output or root/'results/pi4b8_r6_revision'
    if out.exists(): raise FileExistsError(f'Refusing to overwrite: {out}')
    final_path=root/'evidence/pi4b8_corrected/v5_r6_execution/PI4B8_FORMAL_EVIDENCE_FINAL_VALIDATION.json'
    final=json.loads(final_path.read_text()); cells=final['cells']; validate_cells(cells)
    rows=[]; powers=[]; support=[]
    for c in sorted(cells,key=lambda x:x['execution_index']):
        row,power,w=derive_run(root,c); rows.append(row); powers.append(power)
        if c['policy']=='FSSQL-R':
            qp=root/f'evidence/pi4b8_corrected/pi4b8_formal_runtime_v5_r6/q_tables/fssql_r_seed_{c["seed"]}.npz'
            with np.load(qp,allow_pickle=False) as q:
                qv=q['q_values']; visits=q['visits']; states=w.state_id.to_numpy(int)
                unvisited=visits[states].sum(axis=1)==0; zero=(qv[states]==0).all(axis=1)
                action0=w.selected_action_index.to_numpy(int)==0
                # Audit actual masked choices, not merely the number of actions.
                for state,mask,selected in zip(states,w.safe_action_indices,w.selected_action_index):
                    safe=sorted(json.loads(mask)); candidates=safe or [0]
                    expected=max(candidates,key=lambda a:(qv[state,a],-a))
                    if int(selected)!=expected: raise ValueError('Masked Q selection mismatch')
                support.append({'run_id':c['accepted_run_id'],'condition':c['condition'],'seed':c['seed'],
                    'windows':len(w),'unvisited_state_windows':int(unvisited.sum()),'zero_q_row_windows':int(zero.sum()),
                    'unvisited_zero_q_action0_windows':int((unvisited&zero&action0).sum()),
                    'fisvdd_windows':int(action0.sum()),'q_sha256':sha(qp)})
    df=pd.DataFrame(rows); columns=metric_columns(df)
    pairs=[]
    for condition in CONDITIONS:
        for a,b in itertools.combinations(POLICIES,2):
            aa=df[(df.condition==condition)&(df.policy==a)].set_index('seed').loc[list(SEEDS)]
            bb=df[(df.condition==condition)&(df.policy==b)].set_index('seed').loc[list(SEEDS)]
            for metric in columns:
                if not np.isfinite(np.r_[aa[metric],bb[metric]].astype(float)).all(): continue
                pairs.append({'condition':condition,'metric':metric,'method_a':a,'method_b':b,**paired_statistics(aa[metric],bb[metric])})
    paired=pd.DataFrame(pairs)
    if len(paired[paired.metric=='fpr'])!=18: raise ValueError('Incomplete FPR contrasts')
    summary=[]
    for (condition,policy),g in df.groupby(['condition','policy']):
        for metric in columns:
            v=g[metric].to_numpy(float)
            summary.append(dict(condition=condition,policy=policy,metric=metric,n=len(v),mean=float(v.mean()),sd=float(v.std(ddof=1)),median=float(np.median(v)),min=float(v.min()),max=float(v.max())))
    out.mkdir(parents=True)
    df.to_csv(out/'RUN_LEVEL_METRICS.csv',index=False)
    paired.to_csv(out/'PAIRED_ALL_POLICY_CONTRASTS.csv',index=False)
    paired[(paired.method_a=='FSSQL-R')&(paired.method_b=='Safe-Greedy')].to_csv(out/'PAIRED_FSSQL_VS_SAFE_GREEDY.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'POLICY_SUMMARY.csv',index=False)
    pd.DataFrame(support).to_csv(out/'Q_STATE_SUPPORT.csv',index=False)
    json_write(out/'POWER_RECOMPUTATION.json',powers)
    json_write(out/'ACCEPTED_RUN_MANIFEST.json',{'source_manifest':str(final_path.relative_to(root)),'source_sha256':sha(final_path),'cells':cells})
    json_write(out/'ANALYSIS_VALIDATION.json',{
        'status':'PASS','accepted_runs':len(df),'windows':int(df.windows.sum()),'record_evaluations':int(df.records_processed.sum()),
        'replacements':[c['accepted_run_id'] for c in cells if c['replacement']],
        'fpr_contrasts':18,'all_60_raw_meter_and_run_manifest_hashes_checked':True,
        'energy_recomputed_from_raw_meter':True,'masked_actions_recomputed':True,
        'analysis_script_sha256':sha(Path(__file__)),'bootstrap_replicates':BOOTSTRAP_REPLICATES,'bootstrap_seed':BOOTSTRAP_SEED,
        'tests':'two-sided exact paired sign-flip; unadjusted, descriptive multiple endpoints',
        'official_test_accessed':False,'new_hardware_or_training_runs':False})
    (out/'README.md').write_text('''# Corrected Pi 4B R6 analysis\n\nThese tables are derived only from the 60 accepted R6 cells. Original results in `results/pi4b8_r6/` remain unchanged.\nThe update explicitly includes FPR in all 18 condition-specific policy contrasts. The statistical unit is the matched run/seed (five pairs per condition), not records or windows. Percentile paired bootstrap intervals use 2,000 resamples and seed 20261022; exact two-sided sign-flip tests enumerate all 32 sign assignments. P-values are unadjusted descriptive results, not confirmatory findings across all endpoints. Cohen's dz is undefined when the paired differences have zero standard deviation and is left blank.\nQ state-support analysis is a post hoc diagnostic using archived visits and Q values; it did not select or tune a policy. Measured energy uses actual meter timestamps within the ACK interval. No official test, model training or hardware execution is performed by this analysis.\n''')
    json_write(out/'SHA256.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(f'PASS: {len(df)} runs, {int(df.windows.sum())} windows; FPR included; energy independently integrated.')

if __name__=='__main__': main()
