# Pi 3B+ Near-Cap Workload Freeze

Status: **FROZEN BEFORE Q TRAINING, FINAL-TEST OPENING, OR FORMAL POLICY RUNS**.

## Selection provenance

The formal workload is `n=1 | B0`: exactly one input record at the start of
each 100 ms decision window and no experiment-created background CPU load. A
run has 600 windows (nominally 60 seconds), with latency and thermal caps fixed
at 40 ms and 82 C.

This candidate was not chosen after observing policy performance. Commit
`1cc8142` already contained the frozen candidate order
`n=[1,16,64,256,512,1024]`, inner order `background=[B0,B1,B2]`, and the
qualification criteria before this freeze was built. The criteria required at
least one high-utility conservative latency upper bound in inclusive [36,44]
ms, the light-action upper bound below 32 ms, and thermal upper bounds at or
below 82 C. The diagnostic applied that rule once and selected the first
passing candidate, `n=1 | B0`, with LUCID qualifying at 36.784473 ms and FISVDD
at 14.279547 ms.

## Workload generation

For each matched seed 4101--4110, NumPy `default_rng(seed).permutation(N)` is
applied to the frozen input row identities and the first 600 indices are used
without replacement. Every method receives the same ordered trace for a given
regime and seed. The development diagnostic uses only the frozen 6,000-row
scheduler-train source. Formal evaluation will apply the already frozen rule
to official-test rows only after the final-test lock is opened.

Expected scheduler-train trace hashes are embedded in
`configs/near_cap_workload.json`; the combined trace-index artifact SHA-256 is
`b4eccd3f1e3965cfa26f3266b2188292cc6f5617cb372f42f2e7b07e272261b2`.

## Safe-set diagnostic

The policy-blind diagnostic crossed each of the 10 matched 600-window traces
with V2 fit `n1/B0` temperature states in fixed `obs_id` order. It used the
frozen V2 latency/thermal predictors and margins and did not select or compare
any policy.

| Safe-set size | Count | Fraction |
|---:|---:|---:|
| 0 | 0 | 0.000000 |
| 1 | 0 | 0.000000 |
| 2 | 0 | 0.000000 |
| 3 | 0 | 0.000000 |
| 4 | 6000 | 1.000000 |

Each action's admissibility rate is 1.000000. This is an adverse but retained
diagnostic: the predefined workload is near the cap only through LUCID's
conservative latency upper bound and does not make the frozen guard binding in
the sampled calibration states. Formal results therefore cannot be used to
claim that the shield was necessary unless the preregistered online evidence
independently supports that claim. The guard and workload will not be changed
to manufacture action masking.

## Frozen physical and start conditions

- CPU governor: `performance`; observed frequency must remain at least 1350 MHz.
- Cooling: heatsink installed, fan continuously running, case top open.
- Power: corrected/replaced adapter and cable on the stable wall-outlet path.
- Start/cooldown: 30 one-second idle samples; start temperature at most the
  median idle baseline plus 2.0 C; current `get_throttled` low nibble must be 0.
- Emergency stop: 58 C, any current undervoltage/throttle bit, frequency below
  1350 MHz, physical-condition change, hash mismatch, or unrelated background
  work.
- Invalid attempts are retained whole. Partial attempts are never stitched and
  retries are never silent.

## Evidence boundary and hashes

No official-test content, coverage residual, Q table, or policy outcome was
read to select this workload. This is a workload/configuration freeze, not a
policy result and not a new physical measurement.

- Machine config: `configs/near_cap_workload.json`, SHA-256
  `57934ae14513abc4878313651fb8293610be088c1a30da1cfb73a0532c28fed1`
- Generator: `scripts/build_near_cap_freeze.py`, SHA-256
  `b36650d4d60d09dfa98b2ade0a4a37e09f8a076930577d5dd08b091111e11cb5`
- Safe-set rows: SHA-256
  `ecf6f10d96b3a6ae0b1726a4c6624434bd11375f758140b747fbfa361742948b`
- Summary: SHA-256
  `a939e0aa20b49c5e487e7d350e3505bba78deb2e3d0a9d92254135fbbd7070b0`
