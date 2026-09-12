# Single Latency Violation Forensic Record

Status: **RETAINED FORMAL MATRIX V1 OUTCOME; NOT REMOVED, RELABELLED, OR
RETRIED**.

## Evidence identity

- Formal Matrix V1 commit: `6923d20e578497d88a399da243e30dc2db892348`
- Frozen runtime source commit: `50b2539a12a63059645537aed5f807c160f3af4b`
- Policy: `Round-Robin`
- Policy class: unshielded deterministic baseline
- Original regime label: `near_cap`
- Scientific regime interpretation: pre-registered near-cap candidate;
  empirically non-binding
- Seed: `4110`
- Run ordinal and ID: `155`, `pi3bplus_near_cap_round_robin_s4110_a1`
- Window: `520`
- Wall timestamp: `2026-09-09T18:18:31Z`
- Source row: `windows.csv` SHA-256
  `ca41100953f49b6a4228e94bcfbd44bd57db216619bc0b7a5a14c3213abda0bc`

## Workload, action, and guard state

- Workload/background: `n_records=1`, `B0`; backlog start/end `0/0`
- Previous action: `oi_svdd_as_elm_revision`
- Selected action: `fisvdd_revision`
- Switch event: yes
- Measured switch bookkeeping overhead: `0.002813 ms`
- Admissible mask:
  `fisvdd_revision|lucid_revision|tinydl_revision|oi_svdd_as_elm_revision`
- Safe-set cardinality: `4`
- Fallback: `0`
- Selected-action V2 base latency prediction: `3.272850 ms`
- Selected-action residual margin: `1.006697 ms`
- Frozen fixed latency buffer: `10.000000 ms`
- Selected-action guarded latency upper bound:
  `3.272850 + 1.006697 + 10.000000 = 14.279547 ms`
- Guard headroom to the unchanged 40 ms cap: `25.720453 ms`
- All-action guarded headroom, in frozen action order:
  FISVDD `25.720453`, LUCID `3.215527`, TinyDL `26.334774`,
  OI-SVDD+AS-ELM `24.467053` ms
- Guard-margin source SHA-256:
  `1a55ec7f33333b67456d0dddfafd08d46ea8b255f40ea5f56834f9029328e051`

## Realised timing decomposition

The policy log does not contain a field literally named “realised detector
latency.” It separately records preprocessing and selected-model inference;
their sum is reported below as a derived detector-path timing and is not
silently substituted for the full end-to-end measurement.

| Component | Time (ms) |
|---|---:|
| Telemetry acquisition | 43.134651 |
| State construction | 0.293905 |
| Guard evaluation | 0.501820 |
| Action selection | 0.010469 |
| Switch bookkeeping | 0.002813 |
| Logged total controller | 43.943658 |
| Feature preprocessing | 0.463278 |
| Selected-model inference | 0.694474 |
| Derived detector path: feature + inference | 1.157752 |
| Full measured end-to-end | **45.725571** |
| Logging probe, measured after end-to-end stop | 0.318799 |

The 40 ms violation was therefore dominated by a `43.134651 ms` telemetry
acquisition outlier. The selected detector inference was `0.694474 ms`, and
the derived feature-plus-inference path was `1.157752 ms`, both far below the
selected action's `14.279547 ms` guarded upper bound. This record is not a
shielded-policy or threshold-controller event and is not evidence that the V2
action-specific residual bound was exceeded. It instead shows that the formal
end-to-end cap also includes controller/telemetry time that is outside the
action-latency lookup used by the guard.

## Thermal and hardware state

- Start/end temperature: `28.406/28.944 C`
- Predicted/guarded selected-action next temperature:
  `29.065656/30.667491 C`
- Thermal headroom to 82 C: `51.332509 C`
- Minimum sampled CPU frequency: `1400.000 MHz`
- Per-window `get_throttled`: `0x0`
- Per-window undervoltage: `0`
- Independent 1 Hz monitor at the same wall-clock second: `28.944 C`, `0x0`
- Thermal violation: `0`

## Interpretation boundary

This is the only realised latency violation in 96,000 retained formal windows
(`1/96000 = 0.0010416667%`). It occurred in Round-Robin, not FSSQL-R,
Safe-Greedy, or Threshold. It must not be explained as a shield failure or used
to alter the 40 ms cap, the V2 guard, margins, buffers, models, data splits, or
policies. Formal Matrix V1 remains valid non-binding operating-regime evidence:
all four actions were admissible in every window, and its Natural and original
Near-cap labels both used `n=1/B0`. No physical W/J was measured.
