# Pi4B true-n16 Q-training implementation contract

Status: **IMPLEMENTED; SCIENTIFIC INPUTS FROZEN BY THE CORRECTIVE PROTOCOL**

This contract describes the input and lineage checks implemented by
`scripts/train_pi4b8_q_true_n16.py`. Scientific choices are frozen separately
in `PI4B8_CORRECTED_TRUE_N16_PROTOCOL_V1.md`; this file does not authorize
commissioning or a formal run. The legacy trainer, Q tables, preregistration
and historical V4 evidence remain unchanged.

## Development-only workload

The corrected trainer accepts only:

- one hash-bound `6000 x 8` scheduler-development RiskProxy feature array;
- five hash-bound integer `6000 x 16` batch-index arrays, one for each seed
  5101--5105;
- 16 distinct, in-range base-row indices in every window;
- one hash-bound 6000-value backlog array containing exactly zero in every
  window;
- Pi4B guard, runtime, RiskProxy and fit artifacts with declared SHA-256
  values.

RiskProxy runs once on the 6000 base records. Each training-window risk is the
arithmetic mean of the 16 probabilities selected by that window's explicit
seed-specific batch indices. The trainer preserves the frozen index order and
does not apply a second permutation. It independently verifies the frozen
outcome-blind rule
`order=default_rng(seed).permutation(6000); index[w,j]=order[(16w+j)%6000]`.
Predictions are never copied to make a batch, and ground-truth labels are
neither declared as dependencies nor opened by the trainer.

The zero-backlog input is a stationary control assumption. It does not support
a backlog-adaptation or stateful-queue claim. A future experiment that permits
nonzero or action-dependent backlog requires a separately frozen transition
model and newly trained Q tables.

## State and action consistency

The corrected trainer imports the same pure state, guard, thermal and reward
functions used by the corrected Pi4B worker. It requires:

- action order `FISVDD, LUCID, TinyDL, OI-SVDD+AS-ELM`;
- numeric previous-action values `0..3`, with `none=4`;
- the same 1,280-state bins and reward object as runtime;
- Pi4B `n16/B0` latency and thermal guard inputs;
- masked exploration and masked bootstrap targets for FSSQL-R;
- label-free validation-quality, energy, zero-backlog and switching reward
  components;
- seed identity `5101->5101` through `5105->5105`;
- evaluation with `epsilon=0`, no Q update and no replay update.

Training is restricted to `n16/B0`. Evaluation scope may list B0/B1/B2, but
this must be reported as cross-background evaluation of a B0-trained policy,
not B1/B2 Q training.

Only the five FSSQL-R tables used by the corrected formal matrix are trained;
the repair does not create an unused Unshielded-Q branch.

## Fail-closed authorization

The trainer accepts only status
`FROZEN_BEFORE_PI4B8_TRUE_N16_Q_TRAINING`. The separately frozen corrective
protocol, runtime configuration, restored scheduler-development inputs,
seed-specific batch maps, and zero-backlog asset provide the scientific
freeze. The CLI validates all paths and hashes by default and requires an
additional `--execute-training` flag before writing tables.

Each new table embeds the protocol, runtime, base-risk, batch-index and backlog
hashes. The worker verifies those values against the new Q manifest, checks the
seed mapping, and rejects nonzero-backlog use of these stationary-control
tables.
