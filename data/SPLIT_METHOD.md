# Development Split Method

The Full Rebuild protocol specifies a seed-20260909 stratified 60/20/20 split
of the official training file, but does not define the stratification label or
how exact duplicate feature rows are handled.

This implementation stratifies on `attack_cat`, with Normal retained as its own
class. Exact duplicate feature vectors are assigned to one split as a group.
This prevents byte-identical feature vectors from appearing in multiple
development partitions. Five shuffled stratified group folds use seed 20260909;
folds 0-2 form `model_train`, fold 3 forms `validation`, and fold 4 forms
`scheduler_train`.

The resulting proportions may differ slightly from exact 60/20/20 because
duplicate groups are indivisible. Duplicate records are retained, not silently
removed. Feature groups containing inconsistent labels are retained in one
split and disclosed in the leakage audit.

The official test file remains sealed. Cross-source duplicate analysis against
that file is deferred until the final-test lock exists; no test rows or labels
are used to determine this development split.

