# Scheduler-train label restoration audit

Status: **PASS — frozen development-label restoration; official test sealed**

The existing `scheduler_train_6000_raw_features.csv` was mapped to the
official UNSW-NB15 **training** CSV using the pre-existing ordered
`SOURCE_INDEX.csv`. The index contains exactly 6,000 unique zero-based source
row indices. No fuzzy matching, model prediction, or official test access was
used.

Independent checks passed:

1. 6,000/6,000 runtime rows mapped.
2. Mapping is one-to-one (`nunique(source_row_index) = 6000`).
3. Runtime row order equals `SOURCE_INDEX.csv` order.
4. All 6,000 × 42 model-input feature cells match the source training rows
   exactly after removing `id`, `attack_cat`, and `label` (0 mismatches).
5. Every source row is from `data/raw/UNSW_NB15_training-set.csv`; the
   official test CSV was not opened and contributes no rows.
6. `attack_cat`, `label`, source IDs, and audit fields are excluded from model
   input, preprocessing, state construction, risk scoring, guard evaluation,
   action selection, Q-learning, and reward computation.
7. Labels are retained only for post-action TP/FP/FN/TN/F1 scoring.
8. Labeled artifact, lineage, and SHA-256 manifest are generated together and
   must be hash-checked before any Pi4B8 formal run.

Artifacts:

- `scheduler_train_6000_labeled_online_source.csv`
- `scheduler_train_6000_label_lineage.csv`
- `SCHEDULER_TRAIN_6000_LABEL_RESTORATION_SHA256.json`

Scientific boundary: this is a frozen labeled development/scheduler-train
workload for online deployment, controller-overhead, robustness, and physical
energy evaluation. Its F1 is not an independent held-out generalization
estimate. Independent final detector generalization remains reserved for the
sealed official test.
