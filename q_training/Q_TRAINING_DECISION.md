# Frozen Q-Training and Stability Diagnostic

Status: **120,000 preregistered development windows completed; all 20 Q tables
frozen without hyperparameter tuning**.

Ten Unshielded-Q and ten FSSQL-R tables were trained with seeds 3101--3110 for
6,000 windows each. Checkpoints were recorded every 100 windows. Training used
only the frozen scheduler-train risk sequence, development-validation action
quality, and the frozen Pi 3B+ V2 predictors/margins. The official test was not
opened or transformed.

The theoretical state space has 1,280 states and 5,120 state-action pairs.
Observed training support was sparse:

- Unshielded-Q visited 34--38 states and 116--124 state-action pairs per seed.
- FSSQL-R visited 34--36 states and 115--122 state-action pairs per seed.

At window 6,000, mean checkpoint summaries across the 10 seeds were:

| Policy | rolling reward | absolute TD error | absolute Q update | greedy-policy change rate |
|---|---:|---:|---:|---:|
| Unshielded-Q | 0.518533 | 0.405581 | 0.008650 | 0.000313 |
| FSSQL-R | 0.526027 | 0.408324 | 0.007742 | 0.000156 |

The low greedy-policy change rate indicates late-budget action-policy stability
on visited states, but the nonzero TD error and sparse visitation do **not**
establish numerical convergence over the full theoretical state space. This
adverse limitation is retained. The fixed 6,000-window budget is complete and
will not be extended or tuned using formal outcomes.

Independent replay reproduced `checkpoints.csv` byte-for-byte and all arrays in
the 20 tables exactly. NPZ container bytes are not asserted across replay
because ZIP metadata can vary; the committed containers are individually
SHA-256 pinned in `Q_TABLE_MANIFEST.json`.

Formal evaluation maps seed 4101 to table seed 3101 through seed 4110 to 3110,
uses epsilon zero, and performs no Q or replay update. These tables are revision
artifacts and are not represented as recovered historical tables.

- Q config SHA-256: `82894b5435ffc923c0704e39ba0786256ae6329c491d4b0476c0c6365fa1f182`
- Trainer SHA-256: `87d3c6409a2f6781690a8fa1831e2a5c6c8f4d42d11aa35c9f6d32ed592df207`
- Q manifest SHA-256: `f0a45e60c00260ffcc38dd56a7472bdf0f9978a645574edcf2e9c2e61d01fca9`
- Checkpoints SHA-256: `22468b1f1acaa653035fa2a946dfe25ca1d337de1dfc69d2d6c95c9406785de9`
