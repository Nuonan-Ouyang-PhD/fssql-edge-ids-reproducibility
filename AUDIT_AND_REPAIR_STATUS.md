# Pi4B V4 code-to-protocol audit and repair status

Date: 21 October 2026

## Evidence decision

The historical Pi4B V4 package, 60 cells and all replacement-attempt records
are retained unchanged as provenance. They are not accepted as a valid
implementation of the declared Pi4B online experiment. No result was deleted,
overwritten or selected according to scientific outcome.

The executed V4 worker is identified by
`protocols/PI4B8_FORMAL_RUNTIME_PACKAGE_MANIFEST_V4.json` as
`pi4b8_online_worker.py` SHA-256
`1e1f2c9254a513b43d20ba7938ae45baa6526c696885ecdec72c3130e385088a`.
The repaired source at `online/pi4b8/pi4b8_online_worker.py` is a later,
non-executed implementation and is not used to validate the historical runs.

The following V4 claims are suspended pending a corrected preregistered rerun:

- Pi4B policy rankings and F1/FPR comparisons;
- Pi4B reward and switching comparisons;
- `n=16` throughput and energy-normalized metrics;
- B1/B2 background-load robustness;
- controller/E2E timing decomposition;
- policy-conditioned POWER-Z energy comparisons.

The audit does not by itself invalidate Pi3B results, Pi4B independent
fit/margin/coverage observations, or the saved one-shot official-test
aggregate. The official test is locked and must not be opened again.

## Confirmed V4 implementation defects

1. LUCID and TinyDL fell back to an implicit 0.5 threshold online instead of
   the frozen validation thresholds.
2. The official-test evaluator wrote to the wrong directory and hard-coded
   thresholds, although its numerical constants matched the frozen file.
3. Each declared `n=16` window processed and scored one record.
4. The worker did not call RiskProxy; Threshold, Safe-Greedy and Static-Light
   did not implement their documented policies; the online safe set omitted
   thermal feasibility.
5. FSSQL-R used a global Q argmax followed by first-safe fallback instead of an
   argmax restricted to the safe set.
6. Online reward was `TP-FP`; energy, backlog and switching components were not
   implemented and labels entered the quantity called reward.
7. Guard fitting used hard-coded scientific choices rather than enforcing all
   frozen predictor-config fields.
8. No contemporaneous evidence proves that B1/B2 CPU load ran during V4.
9. Several timing fields were aliases or zeros rather than independent
   measurements.
10. The public package lacks the raw POWER-Z/handshake archive needed to
    independently reproduce every historical energy value.

## Implemented repair code

- validation thresholds are loaded from `models/MODEL_HYPERPARAMETERS.json`;
- models without a serialized threshold no longer default to 0.5;
- official-test output is scoped to `results/official_test/` and the existing
  one-shot lock prevents a second evaluation;
- true `windows x 16 x features` inputs are required; 600-row V4 inputs fail;
- raw runtime input rejects label, attack-category and audit identifiers;
- RiskProxy produces the frozen per-record risk values before aggregation;
- Static-Light, Threshold, Safe-Greedy and masked FSSQL-R selection are
  implemented as separate policy branches;
- latency and thermal predictions jointly define the safe set;
- label-free reward components are separated from post-prediction scoring;
- invalid/non-binary labels are rejected before conversion;
- pre- and post-action temperature, frequency and throttle telemetry are
  recorded;
- B1/B2 require the pinned CPU 3, 100-ms-period load used by Pi4B calibration,
  with explicit process lifecycle evidence;
- the guard fitter now validates the declared model, features, quantiles,
  buffers, split roles and solver, and reproduces the accepted numeric guard
  artifacts from frozen raw inputs;
- formal authorization, POWER-Z samples, ACK boundaries, rolls, hashes and
  energy integration now fail closed in the validation code.

## Remaining protocol decisions

The corrected worker is intentionally not executable as a formal campaign yet.
The following must be frozen before building a new runtime package:

- deterministic 16-record batching and input hashes;
- a measured/frozen backlog source and dynamics;
- Pi4B Threshold `p_hi` and explicit light/heavy roles;
- reward energy/state semantics for each condition;
- a new true-`n=16` Q-training protocol and tables, or an explicit decision to
  evaluate the old tables only as a legacy single-record transfer baseline;
- a corrected package, launcher/schema contract, commissioning result and
  final execution manifest.

The detailed decision boundary is recorded in
`protocols/PI4B8_CORRECTED_RUNTIME_DECISION_DRAFT.md`. That file is a draft and
does not authorize retraining or hardware execution.

## Verification

The repair test suite covers threshold provenance, one-shot test locking,
config-driven guard fitting, true-`n=16` batching, label isolation, RiskProxy,
policy branches, masked argmax, joint feasibility, reward components,
background-load configuration, telemetry fields, formal authorization and
POWER-Z integration boundaries.

Run:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

No corrected formal run has been started from this repair branch.
