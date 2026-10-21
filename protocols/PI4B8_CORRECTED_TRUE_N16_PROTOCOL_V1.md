# Pi4B8 corrected true-n16 protocol V1

Status: **FROZEN BEFORE CORRECTED Q TRAINING, COMMISSIONING, OR ONLINE EXECUTION**

Date: 21 October 2026

## Scope and evidence boundary

This is an audit-driven corrective protocol. It does not modify or overwrite
the historical V4 package, attempts, or results. Those artifacts remain
provenance only and are excluded from current Pi4B policy, robustness, timing,
and energy claims.

The official UNSW-NB15 test has already been opened once. It is forbidden to
this repair. No official-test feature, label, prediction, metric, or ranking
may select a batch, policy parameter, Q parameter, threshold, workload, or
rerun decision. Only the frozen scheduler-development split, validation
metrics, Pi4B calibration, and existing method-level rules are used below.

## Corrected workload

- Source: the frozen 6,000-row `scheduler_train` development trace restored
  one-to-one from the official UNSW-NB15 **training** CSV.
- For seed `s`, compute `order = default_rng(s).permutation(6000)`.
- Window `w` contains the 16 distinct rows
  `order[(16*w+j) mod 6000]`, `j=0..15`.
- Q training uses 6,000 windows. Online evaluation uses the first 600 windows
  of the same seed-specific schedule.
- All four policies and B0/B1/B2 use the identical schedule for a matched
  seed. The 16 row indices are retained for every window.
- Risk is the arithmetic mean of the 16 frozen RiskProxy probabilities and is
  computed before action selection.
- Labels remain in a physically separate sidecar and are opened only after all
  actions and detector predictions have been produced.

This is a stationary synchronous workload. Each selected detector completes
the current 16-record batch before the next 100-ms decision. Queued backlog is
therefore frozen at zero. The backlog state component is retained for encoding
compatibility, but only bin 0 is used and no backlog-adaptation claim is made.
Threshold receives current queue load `16 + queued_backlog`, not queued backlog
alone.

## Frozen policy semantics

- Action order: FISVDD, LUCID, TinyDL, OI-SVDD+AS-ELM.
- Static-Light always selects FISVDD.
- Safe-Greedy maximizes the frozen immediate label-free utility over the Pi4B
  safe set.
- FSSQL-R uses epsilon 0 and masked argmax over the same Pi4B safe set, with no
  online Q or replay updates.
- Threshold remains an unshielded conventional baseline. It uses
  `T_hi=80.14945503488127 C`, `T_lo=78.14945503488127 C`, `q_hi=512`,
  `q_lo=256`, and the pre-existing validation-only method default `p_hi=0.70`.
  FISVDD is its light action and TinyDL its heavy action. These roles and
  `p_hi` are carried forward without consulting V4 or official-test outcomes.

The Q state encodes risk, temperature, residual-energy proxy, zero backlog,
and previous action. Numeric previous-action values are action indices 0--3;
`none` is 4. This corrects the metadata ordering error in the legacy Pi4B Q
configuration.

## Q-training objective

New Pi4B tables must be trained; the legacy single-record tables are rejected.
Training is development-only, true-n16, B0, five seeds 5101--5105, 6,000
windows per seed, and 600-window episodes. The earlier transparent revision
hyperparameters remain: gamma 0.95, zero initialization, epsilon 1.0 to 0.05
through window 4,500, visit-count learning rate
`min(0.5,(1+visits)^-0.6)`, FIFO replay 512, and four uniform replay updates
per window.

Each episode initializes temperature to the median valid `T_start_C` in the
frozen Pi4B formal-fit `n16/B0` cell and resets it every 600 windows. This is
the Pi4B/true-n16 form of the earlier fit-median initialization rule.

The reward is label-free:

`risk*validation_TPR - (1-risk)*validation_FPR`

minus the frozen energy-proxy, backlog, and switching penalties with weights
0.1, 0.05, and 0.02. The energy proxy uses Pi4B n16/B0 predicted latency
normalized by LUCID, identically in B0/B1/B2:

- FISVDD: 0.3525127703457099
- LUCID: 1.0
- TinyDL: 0.42625115703297095
- OI-SVDD+AS-ELM: 0.5844038622269349

Residual energy starts at 1 and decreases by `0.001*normalized_cost` per
window. It resets every 600 windows. This synthetic proxy is not physical
energy. POWER-Z W/J is retained only as a post-run measurement and never enters
state, reward, or policy selection.

## Guard, load, and hardware execution

The accepted Pi4B latency predictor, thermal predictor, residual margins,
40-ms latency cap, 82-C thermal cap, and 100-ms window remain unchanged.

- B0: no experiment-created CPU load.
- B1: one process pinned to CPU 3, 50% duty in a 100-ms period.
- B2: one process pinned to CPU 3, 100% duty in a 100-ms period.
- The load starts before `RUN_START_ACK`, stabilizes for 0.5 s, remains alive
  through `RUN_END_ACK`, and is then stopped and verified. PID, affinity, duty,
  period, liveness, and cleanup are evidence fields.

The corrected matrix remains four policies by three conditions by five
matched seeds by 600 windows. It may start only after new Q hashes, inputs,
runtime package, remote hash validation, label-isolation tests, commissioning,
POWER-Z alignment, and final execution authorization all pass. A technically
valid but scientifically unfavorable run is never repeated. The manuscript is
not updated with corrected Pi4B results until the complete corrected evidence
matrix passes independent validation.
