# Pi 3B+ Policy Experiment Preregistration

Status: **PREREGISTERED BEFORE Q TRAINING, FINAL-TEST OPENING, AND FORMAL POLICY EXECUTION**.

## Formal matrix

The primary Pi 3B+ matrix contains 160 physical closed-loop runs:

- eight policies: Static-Heavy, Static-Light, Round-Robin, Greedy-Risk,
  Safe-Greedy, Threshold, Unshielded-Q, and FSSQL-R;
- two regime labels: Natural and Near-cap;
- 10 matched seeds: 4101--4110;
- 600 decision windows per run at 100 ms/window (nominal 60 seconds).

The exact 160-row order is frozen in
`configs/pi3b_policy_experiment_matrix.csv`, SHA-256
`6c1645c1c647bc860ba09a8e15d58c44a52bbaa1903f9950ecd5c4240643a51e`.
Within each seed, all 16 policy/regime combinations are permuted once by
NumPy `default_rng(20260909 + evaluation_seed)`. Seed blocks run in ascending
order. This associates no method systematically with an early or late position.

## Workloads and matching

Every policy receives the same ordered 600-record trace for a given seed and
regime. Both frozen workloads use one record/window and B0 (no
experiment-created CPU load). The Near-cap selector independently returned the
same minimum candidate as Natural because LUCID's conservative V2 latency upper
bound is 36.784473 ms. Consequently, the two regime labels do not encode
different physical pressure in this matrix. That adverse lack of regime
separation is frozen and must be reported; it cannot be repaired after policy
outcomes are observed.

The official test is still sealed. After all preflight gates pass, the frozen
seeded permutation will be applied once to official-test row identities. The
scheduler may read features and risk-proxy outputs only. `label`, `attack_cat`,
and `id` are unavailable to policy selection and are used only after inference
to calculate run metrics.

## Frozen actions and policy logic

- Light action: FISVDD, selected as the lowest median V2 fit-cell Q0.99 latency.
- Heavy action: TinyDL, selected as the highest development-validation F1.
- Static policies always select their named role.
- Round-Robin uses the frozen action order FISVDD, LUCID, TinyDL,
  OI-SVDD+AS-ELM.
- Greedy-Risk maximizes the validation-only immediate security score over all
  actions without a shield.
- Safe-Greedy maximizes frozen immediate total reward over the exact V2 safe set.
- Threshold uses no shield and applies the frozen hysteresis below.
- Unshielded-Q greedily queries its mapped frozen table over all actions.
- FSSQL-R greedily queries its mapped frozen table only over the exact V2 safe
  set also used by Safe-Greedy.

FSSQL-R and Safe-Greedy therefore differ in selection objective, not guard.
Formal evaluation uses epsilon 0, no Q update, and no replay update.

## Conventional Threshold controller

- `T_hi = 80.99816539732174 C` (`82 - epsilon_T`)
- `T_lo = 78.99816539732174 C`
- `q_hi = 256` records: highest B0 workload whose TinyDL conservative latency
  upper bound is at most 40 ms
- `q_lo = 64` records
- `p_hi = 0.70`, the preregistered validation-only default

Logic: select FISVDD when `T >= T_hi` or `q >= q_hi`; select TinyDL when
`T <= T_lo`, `q <= q_lo`, and `p >= p_hi`; otherwise retain the previous action
or initialize with FISVDD. These values will not be tuned on formal outcomes.

## Frozen tabular learning revision

The submitted manuscript did not retain numeric Q-learning hyperparameters.
The transparent revision implementation freezes:

- state: risk bin x temperature bin x residual-energy bin x backlog bin x
  previous action;
- 1,280 states, four actions, 5,120 state-action pairs;
- 10 training seeds 3101--3110, each 6,000 windows, checkpoint every 100;
- gamma 0.95;
- alpha `min(0.5, (1 + visit_count)^-0.6)`;
- epsilon linearly 1.0 to 0.05 through window 4,500, then 0.05;
- zero initialization;
- FIFO replay capacity 512 and four uniform replay updates per window;
- training input limited to frozen scheduler-train/risk/calibration artifacts;
- evaluation seed 4101 maps to training seed 3101, and so on through 4110/3110.

The complete state, dynamics, reward, replay, and evaluation definitions are in
`configs/q_learning_protocol.json`, SHA-256
`82894b5435ffc923c0704e39ba0786256ae6329c491d4b0476c0c6365fa1f182`.
Long-horizon training/stability evidence must complete and Q tables must be
committed before the final test can be opened.

## Per-run gates and stop rules

Every run begins with 30 one-second idle samples and may start only when the
temperature is at most the idle median plus 2 C, V2 cooling remains heatsink +
continuous fan + open case top, the corrected wall-outlet power path remains
unchanged, all governors are `performance`, frequency is at least 1350 MHz,
and the current `get_throttled` low nibble is zero.

Any current undervoltage/throttle bit, frequency below 1350 MHz, temperature at
or above 58 C, physical-condition change, hash mismatch, unrelated background
job, incomplete field, or process/I/O failure invalidates the complete attempt
and stops the campaign. Partial attempts are retained and never stitched.
There is no silent or automatic retry.

## Raw evidence

Each window records all fields frozen in `configs/pi3b_policy_experiment.json`,
including complete per-action guard predictions/headroom, selected and previous
actions, safe-set mask/size, realised IDS and end-to-end latency, temperature,
backlog, rewards and components, switching, CPU frequency, throttle state, and
controller component timing. Non-applicable Q-update/replay fields are blank,
not fabricated zeros.

The main latency cap is 40 ms and thermal cap is 82 C. Controller and feature
timing are included in realised end-to-end latency. Whole-device W/J are not
claimed for these Pi 3B+ runs because POWER-Z is not currently on this device.

No guard, workload, threshold, policy, model, reward, state bin, or Q-learning
parameter may be changed after official-test opening because of an adverse or
non-significant result.
