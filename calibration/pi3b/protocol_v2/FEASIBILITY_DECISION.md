# Protocol V2 Calibration-Only Feasibility Decision

Status: feasibility demonstrated; candidate identified but policy evaluation has
not started.

The frozen diagnostic evaluated 18 workload/background candidates and 72
action/candidate cells using only the V2 fit data, frozen guard artifacts, and
previously frozen validation F1 values. It did not read coverage residuals or
policy outcomes.

The first passing candidate in the frozen order is one record/window with B0
background. At that candidate, the FISVDD light-action latency upper bound is
14.279547 ms, below 32 ms. The LUCID high-utility latency upper bound is
36.784473 ms, inside the inclusive 36--44 ms near-cap band. Maximum fit-state
thermal upper bounds for the four actions range from 35.244638 C to 36.636691 C,
all below 82 C. Candidates one record/window with B1 and B2 also pass, but are
not selected because the rule requires the first passing candidate.

This is a calibration-only feasibility result, not a policy outcome and not a
new physical run. A separate frozen `near_cap` artifact and complete policy-run
preregistration are still required before any policy evaluation starts.
