#!/usr/bin/env python3
"""Generate manuscript figures from the corrected R6 analysis and checkpoints."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'paper/figures'; out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(ROOT/'results/pi4b8_r6_revision/RUN_LEVEL_METRICS.csv')
    policies=['FSSQL-R','Safe-Greedy','Threshold','Static-Light']
    conditions=['n16/B0','n16/B1','n16/B2']
    for metric,name,ylabel,kind in [('fpr','pi4b_fpr','False-positive rate','range'),('energy_J','pi4b_energy','DC input energy per run (J)','sd')]:
        fig,ax=plt.subplots(figsize=(6.4,3.65))
        for j,p in enumerate(policies):
            g=d[d.policy==p].groupby('condition')[metric]
            mean=g.mean().reindex(conditions).to_numpy()
            err=np.vstack((mean-g.min().reindex(conditions),g.max().reindex(conditions)-mean)) if kind=='range' else g.std().reindex(conditions).to_numpy()
            x=np.arange(3)+(j-1.5)*.045
            ax.errorbar(x,mean,yerr=err,marker=['o','s','^','D'][j],capsize=3,linewidth=1.25,label=p)
        ax.set_xticks(range(3),['B0','B1','B2']);ax.set_xlabel('Configured background condition');ax.set_ylabel(ylabel)
        if metric=='fpr': ax.set_ylim(-.035,1.06)
        ax.grid(axis='y',alpha=.25)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.22),ncol=2,fontsize=9,frameon=False)
        fig.tight_layout();fig.savefig(out/f'{name}.pdf',bbox_inches='tight');fig.savefig(out/f'{name}.png',dpi=220,bbox_inches='tight');plt.close(fig)
    q=pd.read_csv(ROOT/'q_training/pi4b8_true_n16_v3/checkpoints.csv')
    fig,ax=plt.subplots(figsize=(6.4,3.6))
    for seed,g in q.groupby('seed'):
        ax.plot(g.window,g.rolling_reward,linewidth=.9,label=str(seed))
    ax.set_xlabel('Training windows');ax.set_ylabel('Rolling reward');ax.grid(alpha=.25)
    ax.legend(title='Training seed',ncol=5,fontsize=8,loc='lower right',frameon=False)
    fig.tight_layout();fig.savefig(out/'pi4b_learning.pdf',bbox_inches='tight');fig.savefig(out/'pi4b_learning.png',dpi=200,bbox_inches='tight');plt.close(fig)
    # Coverage residuals already archived; these are not new calibration fits.
    for suffix,col,xlabel in [('latency','tau_residual_ms','Latency residual (ms)'),('thermal','T_residual_C','Temperature residual (degrees C)')]:
        fig,ax=plt.subplots(figsize=(6.4,3.45))
        for name,relative in [('Pi 3B+','calibration/pi3b/protocol_v2/coverage_evaluation_v1/guard_residuals_coverage.csv'),('Pi 4B 8GB','calibration/pi4b8/coverage_evaluation_v1/guard_residuals_coverage.csv')]:
            x=np.sort(pd.read_csv(ROOT/relative)[col].to_numpy());y=np.arange(1,len(x)+1)/len(x)
            ax.step(x,y,where='post',label=name,linewidth=1.3)
        ax.set_xlabel(xlabel);ax.set_ylabel('Empirical cumulative probability');ax.grid(alpha=.25);ax.legend(frameon=False)
        fig.tight_layout();fig.savefig(out/f'coverage_{suffix}.pdf',bbox_inches='tight');plt.close(fig)
    print('Updated figures: online FPR, physical energy, learning checkpoints, and held-out residual ECDFs.')

if __name__=='__main__': main()
