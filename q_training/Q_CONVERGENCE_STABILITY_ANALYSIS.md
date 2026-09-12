# Q-learning 6000-window convergence and stability analysis

Input: `q_training/checkpoints.csv` (1200 checkpoint rows; 10 seeds × 2 policies × 30 checkpoints).

The analysis is descriptive and frozen: no retraining, tuning, or official-test access.

## FSSQL-R

- Final rolling reward: mean 0.526027, SD 0.054483; early mean 0.468582; late mean 0.538307.
- Final rolling utility: mean 0.610795, SD 0.047842; late mean 0.619906.
- Final absolute TD error: mean 0.408324; late mean 0.392142.
- Final Q-update magnitude: mean 0.007742; late mean 0.008250.
- Final visited states/actions: mean 34.90/118.90; final greedy-policy change rate mean 0.00015625.

## Unshielded-Q

- Final rolling reward: mean 0.518533, SD 0.049802; early mean 0.469135; late mean 0.540959.
- Final rolling utility: mean 0.603689, SD 0.050267; late mean 0.618975.
- Final absolute TD error: mean 0.405581; late mean 0.388966.
- Final Q-update magnitude: mean 0.008650; late mean 0.008190.
- Final visited states/actions: mean 35.20/120.90; final greedy-policy change rate mean 0.00031250.

## Interpretation

Both policies complete the preregistered 6000-window training horizon with deterministic replay validation already recorded in `q_training/INDEPENDENT_VALIDATION.json`. Late-window metrics are reported as stability diagnostics. Sparse visited-state support and non-zero residual TD error remain visible; therefore this evidence supports late-budget behavioral stability on visited states, not mathematical convergence over the full state space.

## Reviewer-requested 600-window checkpoint

| Policy | reward @600 | utility @600 | states @600 | state-actions @600 | reward @6000 | utility @6000 | states @6000 | state-actions @6000 | greedy-change @600 → @6000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FSSQL-R | 0.483742 | 0.575863 | 29.7 | 96.3 | 0.526027 | 0.610795 | 34.9 | 118.9 | 0.006875 → 0.000156 |
| Unshielded-Q | 0.488472 | 0.581728 | 29.8 | 95.7 | 0.518533 | 0.603689 | 35.2 | 120.9 | 0.007031 → 0.000313 |

The retained artifacts contain final Q tables and checkpoint behavioral diagnostics, but no serialized Q-table snapshot at window 600. Therefore element-wise Q-table drift and exact statewise greedy-policy agreement from 600 to 6000 are not identifiable without rerunning training; no such values are fabricated. The checkpoint change-rate evidence does show a marked reduction in policy changes by the final checkpoint.

Frozen schedule: 1,280 discretized states and 5,120 state-action pairs; alpha `min(0.5,(1+visit_count)^-0.6)`; gamma `0.95`; epsilon linear 1.0 at window 0 to 0.05 at window 4,500, then 0.05; FIFO replay capacity 512; one online update plus four uniform replay updates per window, sampling with replacement; evaluation epsilon 0 and no updates.
