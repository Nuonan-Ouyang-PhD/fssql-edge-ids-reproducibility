# Pi 3B+ Stateful Transition Extension — preregistration freeze

Status: **DESIGN FROZEN — NOT EXECUTED**

Scientific question: does FSSQL-R gain incremental value over Safe-Greedy when selected actions affect future queue/resource states?

This extension is separate from Formal Matrix V1 and Stress Extension V3. It cannot alter either accepted campaign or the V2 guard. No existing FSSQL-R/Safe-Greedy result may be used to choose a workload, transition, seed, or stopping rule.

## Frozen workload and transitions

Use only the native workload parameters supported by the online generator: `n_records ∈ {1, 2, 4, 8, 16}` and background labels `B0`, `B1`, `B2` with their existing frozen duties. The diagnostic and formal sequence is fixed before any policy run:

`(n=16,B2) × 200 windows → (n=16,B1) × 200 windows → (n=16,B0) × 200 windows → repeat once`.

The transition schedule is identical for every policy and matched seed. The stateful runner must use the existing queue state, but replace the current constant-service placeholder with the pre-registered measured service model for each selected action. During B2/B1, arrivals exceed service for at least one admissible action and backlog must accumulate; during B0, service exceeds arrivals for at least one action and backlog must recover. If the frozen service-model preflight cannot demonstrate both accumulation and recovery before policy execution, the extension is invalid and stops without policy runs.

Seeds are `6201–6210`, one run per policy/seed, 1,200 windows per run. Policies are FSSQL-R, Safe-Greedy, Threshold, Unshielded-Q, Round-Robin, and Static-Light. Labels remain outcome-blind for workload selection.

## Acceptance and analysis

Before execution, freeze the service-model hash, transition schedule hash, runner hash, guard/config/model/Q/input hashes, and hardware manifest. A run is valid only with complete windows, no throttle/undervoltage/soft-temperature event, intact trace lineage, and exact transition timestamps. Report backlog accumulation and recovery separately by segment, plus F1, reward components, latency quantiles, action composition, switching, controller overhead, guard headroom, and violations.

The primary comparison is paired seed-level FSSQL-R minus Safe-Greedy on recovery time, peak backlog, cumulative reward, F1, latency, and switching. Use exact/appropriate paired tests and effect sizes; never use pooled-window significance as the sole result. If no incremental learning benefit is demonstrated, stop all additional stateful attempts and conclude that Q-learning did not justify its added complexity within the evaluated envelope.

This file is a preregistration only. It contains no execution result and must not be cited as evidence that transitions or backlog dynamics have been observed.
