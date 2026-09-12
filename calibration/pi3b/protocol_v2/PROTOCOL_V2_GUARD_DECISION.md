# Protocol V2 Guard Decision

Status: accept the V2 guard for calibration-only feasibility diagnostics and
subsequent preregistered policy work under the unchanged V2 physical condition.

The independently validated held-out split contains 2,880 rows. Latency and
joint coverage are both 2,879/2,880 = 0.999653, while thermal coverage is
2,880/2,880 = 1.000000. This exceeds the 0.99 target-report level without
coverage tuning or retry.

FISVDD, TinyDL, and OI-SVDD+AS-ELM each have joint coverage 1.000000. LUCID has
0.998611; its only uncovered row is in `lucid_revision|n256|B2`, whose cell
coverage is 39/40 = 0.975000. This residual event remains in the evidence and
must not be hidden or converted into a guarantee.

Acceptance is conditional on the frozen V2 environment: Pi 3B+ identity,
corrected power path, installed heatsink, continuously running fan, open top
cover, `performance` governor, 1400 MHz frequency invariant, and the exact
frozen predictor/margin artifacts. Any change requires a new calibration
decision. The 0.999653 empirical result is not an unconditional physical safety
guarantee.
