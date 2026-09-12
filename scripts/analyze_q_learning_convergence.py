#!/usr/bin/env python3
import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; src=ROOT/'q_training/checkpoints.csv'; out=ROOT/'q_training/Q_CONVERGENCE_STABILITY_ANALYSIS.md'
rows=list(csv.DictReader(src.open())); policies=sorted(set(r['policy'] for r in rows)); lines=['# Q-learning 6000-window convergence and stability analysis','',f'Input: `q_training/checkpoints.csv` ({len(rows)} checkpoint rows; 10 seeds × 2 policies × 30 checkpoints).','', 'The analysis is descriptive and frozen: no retraining, tuning, or official-test access.']
for p in policies:
  sub=[r for r in rows if r['policy']==p]; finals=[r for r in sub if r['window']=='6000']; late=[r for r in sub if int(r['window'])>=5000]; early=[r for r in sub if int(r['window'])<=1000]
  def mean(rs,k): return sum(float(r[k]) for r in rs)/len(rs)
  def sd(rs,k):
    x=[float(r[k]) for r in rs]; m=sum(x)/len(x); return (sum((v-m)**2 for v in x)/(len(x)-1))**.5
  lines += ['',f'## {p}','', f'- Final rolling reward: mean {mean(finals,"rolling_reward"):.6f}, SD {sd(finals,"rolling_reward"):.6f}; early mean {mean(early,"rolling_reward"):.6f}; late mean {mean(late,"rolling_reward"):.6f}.', f'- Final rolling utility: mean {mean(finals,"rolling_utility"):.6f}, SD {sd(finals,"rolling_utility"):.6f}; late mean {mean(late,"rolling_utility"):.6f}.', f'- Final absolute TD error: mean {mean(finals,"td_abs_mean"):.6f}; late mean {mean(late,"td_abs_mean"):.6f}.', f'- Final Q-update magnitude: mean {mean(finals,"q_update_abs_mean"):.6f}; late mean {mean(late,"q_update_abs_mean"):.6f}.', f'- Final visited states/actions: mean {mean(finals,"visited_states"):.2f}/{mean(finals,"visited_state_actions"):.2f}; final greedy-policy change rate mean {mean(finals,"greedy_policy_change_rate"):.8f}.']
lines += ['', '## Interpretation', '', 'Both policies complete the preregistered 6000-window training horizon with deterministic replay validation already recorded in `q_training/INDEPENDENT_VALIDATION.json`. Late-window metrics are reported as stability diagnostics. Sparse visited-state support and non-zero residual TD error remain visible; therefore this evidence supports late-budget behavioral stability on visited states, not mathematical convergence over the full state space.']
out.write_text('\n'.join(lines)+'\n'); print(out)
