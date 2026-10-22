#!/usr/bin/env python3
"""Regenerate R6 paper tables; run analysis first. Does not access official test."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'paper'; D=ROOT/'results/pi4b8_r6_revision'
df=pd.read_csv(D/'RUN_LEVEL_METRICS.csv'); pairs=pd.read_csv(D/'PAIRED_FSSQL_VS_SAFE_GREEDY.csv'); support=pd.read_csv(D/'Q_STATE_SUPPORT.csv')
(P/'tables').mkdir(exist_ok=True)
policies=['FSSQL-R','Safe-Greedy','Threshold','Static-Light']; conditions=['n16/B0','n16/B1','n16/B2']
def ms(g,k,dec): return f'${g[k].mean():.{dec}f}\\pm{g[k].std(ddof=1):.{dec}f}$'
lines=[r'\begin{table*}[t]',r'\caption{Corrected Pi 4B online evaluation: five matched seeds per condition, 600 windows of 16 records per run. FPR is median [minimum, maximum]; other entries are mean $\pm$ SD. All four policies have zero observed latency and thermal violations. Energy includes the acknowledged run interval, not detector execution alone.}',r'\label{tab:pi4}',r'\centering\scriptsize',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{llcccccc}',r'\toprule',r'Load & Policy & F1 & FPR: median [range] & Mean reward & P99 E2E (ms) & Controller (ms) & Energy (J) \\',r'\midrule']
for i,cond in enumerate(conditions):
 for pol in policies:
  g=df[(df.condition==cond)&(df.policy==pol)]; f=g.fpr
  lines.append(f'{cond[-2:]} & {pol} & {ms(g,"f1",4)} & {f.median():.4f} [{f.min():.4f},{f.max():.4f}] & {ms(g,"reward_mean",4)} & {ms(g,"e2e_p99_ms",3)} & {ms(g,"controller_ms_mean",3)} & {ms(g,"energy_J",2)} '+r'\\')
 if i<2:lines.append(r'\midrule')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']; (P/'tables/pi4_online.tex').write_text('\n'.join(lines)+'\n')
# Compact paired table for main text: FPR deliberately explicit.
lines=[r'\begin{table}[t]',r'\caption{FSSQL-R minus Safe-Greedy on Pi 4B. Paired mean difference and percentile bootstrap 95\% interval; exact two-sided sign-flip $p$. Five pairs per condition.}',r'\label{tab:paired}',r'\centering\scriptsize',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{llrrr}',r'\toprule',r'Load & Metric & Difference & 95\% interval & $p$ \\',r'\midrule']
for cond in conditions:
 for metric,name in [('f1','F1'),('fpr','FPR'),('energy_J','Energy (J)')]:
  x=pairs[(pairs.condition==cond)&(pairs.metric==metric)].iloc[0]; dec=3 if metric=='energy_J' else 4
  lines.append(f'{cond[-2:]} & {name} & {x.mean_diff:+.{dec}f} & [{x.ci95_low:+.{dec}f},{x.ci95_high:+.{dec}f}] & {x.exact_signflip_p:.4f} '+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(P/'tables/pi4_paired.tex').write_text('\n'.join(lines)+'\n')

rows=[]
for cond in conditions:
 for seed in range(5101,5106):
  g=df[(df.condition==cond)&(df.seed==seed)].set_index('policy'); z=support[(support.condition==cond)&(support.seed==seed)].iloc[0]
  rows.append(f"{cond[-2:]} & {seed} & {g.loc['FSSQL-R','f1']:.4f} & {g.loc['FSSQL-R','fpr']:.4f} & {g.loc['Safe-Greedy','f1']:.4f} & {g.loc['Safe-Greedy','fpr']:.4f} & {g.loc['Threshold','fpr']:.4f} & {int(z.unvisited_state_windows)}/600 " + r"\\")
start=r"\begin{table*}[t]\caption{Pi 4B per-seed outcomes and post hoc FSSQL training-support diagnostic. Static-Light FPR is 1.0 in every run.}\centering\footnotesize"+"\n"+r"\begin{tabular}{llrrrrrr}\toprule"+"\n"+r"Load & Seed & FSSQL F1 & FSSQL FPR & Greedy F1 & Greedy FPR & Threshold FPR & Unvisited states (windows) \\\midrule"
(P/'tables/pi4_seed_details.tex').write_text(start+'\n'+'\n'.join(rows)+'\n'+r"\bottomrule\end{tabular}\end{table*}"+'\n')
rows=[]
for cond in conditions:
 for metric,label,dec in [('f1','F1',5),('fpr','FPR',5),('reward_mean','Mean reward',5),('e2e_p99_ms','P99 E2E (ms)',3),('energy_J','Energy (J)',3)]:
  x=pairs[(pairs.condition==cond)&(pairs.metric==metric)].iloc[0]
  rows.append(f'{cond[-2:]} & {label} & {x.mean_diff:+.{dec}f} & [{x.ci95_low:+.{dec}f}, {x.ci95_high:+.{dec}f}] & {x.exact_signflip_p:.4f} & {x.cohen_dz:.3f} '+r"\\")
start=r"\begin{table*}[t]\caption{Corrected Pi 4B primary paired comparison (five matched runs per row).}\centering\footnotesize"+'\n'+r"\begin{tabular}{llrrrr}\toprule"+'\n'+r"Load & Metric & Mean difference & Bootstrap 95\% interval & Exact $p$ & $d_z$ \\\midrule"
(P/'tables/pi4_paired_details.tex').write_text(start+'\n'+'\n'.join(rows)+'\n'+r"\bottomrule\end{tabular}\end{table*}"+'\n')

# Retained Pi 3B+ comparison: expose all seed dispersion in the supplement.
p3=pd.read_csv(P/'data/PI3B_PER_SEED_POLICY_SUMMARY.csv')
lines=[r'\begin{table*}[t]',r'\caption{Pi 3B+ stationary binding comparison: mean $\pm$ SD across ten matched seeds; violations are total counts. Per-seed and P99 values accompany the source.}',r'\centering\scriptsize',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lccccccr}',r'\toprule',r'Policy & F1 & FPR & Mean reward & Mean E2E (ms) & Controller (ms) & Switch cost & Viol. \\',r'\midrule']
for pol in ['FSSQL-R','Safe-Greedy','Threshold','Unshielded-Q','Round-Robin','Static-Light']:
 g=p3[p3.policy==pol]
 if len(g)!=10: raise ValueError('Pi3B matched-seed count mismatch')
 lines.append(f'{pol} & {ms(g,"f1",4)} & {ms(g,"fpr",4)} & {ms(g,"reward_mean",4)} & {ms(g,"latency_mean_ms",3)} & {ms(g,"controller_overhead_mean_ms",3)} & {ms(g,"switch_cost_cumulative",3)} & {int(g.latency_violations.sum())} '+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
(P/'tables/pi3_summary.tex').write_text('\n'.join(lines)+'\n')

print('Five manuscript tables generated from the retained seed-level results.')
