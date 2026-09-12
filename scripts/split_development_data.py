#!/usr/bin/env python3
"""Create the frozen group-safe development split from official UNSW train."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
MANIFEST = ROOT / "data" / "SPLIT_MANIFEST.csv"
AUDIT = ROOT / "data" / "LEAKAGE_AUDIT.json"
SEED = 20260909
EXCLUDED = {"id", "label", "attack_cat"}

frame = pd.read_csv(SOURCE)
required = EXCLUDED
missing = sorted(required - set(frame.columns))
if missing:
    raise RuntimeError(f"missing required columns: {missing}")

feature_columns = [column for column in frame.columns if column not in EXCLUDED]
groups = pd.util.hash_pandas_object(
    frame[feature_columns], index=False, categorize=True
).to_numpy(dtype=np.uint64)
strata = frame["attack_cat"].fillna("Normal").astype(str)

folds = np.full(len(frame), -1, dtype=np.int8)
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
for fold, (_, held_out) in enumerate(
    splitter.split(np.zeros(len(frame)), strata, groups)
):
    folds[held_out] = fold
if np.any(folds < 0):
    raise RuntimeError("not every row received a fold")

split_names = np.where(
    folds < 3, "model_train", np.where(folds == 3, "validation", "scheduler_train")
)

group_to_splits: dict[int, set[str]] = {}
for group, split in zip(groups, split_names):
    group_to_splits.setdefault(int(group), set()).add(str(split))
cross_split_groups = sum(len(splits) > 1 for splits in group_to_splits.values())
if cross_split_groups:
    raise RuntimeError(f"{cross_split_groups} exact feature groups cross splits")

with MANIFEST.open("x", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["source_row_index", "source_id", "attack_cat", "label", "feature_group_hex", "fold", "split"])
    for index, row in frame.iterrows():
        writer.writerow([
            index,
            row["id"],
            row["attack_cat"],
            row["label"],
            f"{int(groups[index]):016x}",
            int(folds[index]),
            split_names[index],
        ])

feature_table = pd.DataFrame({"group": groups, "label": frame["label"]})
conflicting_groups = int((feature_table.groupby("group")["label"].nunique() > 1).sum())
counts = pd.crosstab(pd.Series(split_names, name="split"), strata)
split_rows = pd.Series(split_names).value_counts()

audit = {
    "status": "PASS_DEVELOPMENT_SPLIT",
    "source_rows": int(len(frame)),
    "source_columns": list(frame.columns),
    "feature_columns": feature_columns,
    "stratification": "attack_cat including Normal",
    "grouping": "exact feature vector pandas stable uint64 hash",
    "seed": SEED,
    "fold_mapping": {
        "model_train": [0, 1, 2],
        "validation": [3],
        "scheduler_train": [4]
    },
    "split_rows": {key: int(value) for key, value in split_rows.items()},
    "split_proportions": {key: float(value / len(frame)) for key, value in split_rows.items()},
    "attack_category_counts": {
        split: {category: int(value) for category, value in row.items()}
        for split, row in counts.to_dict(orient="index").items()
    },
    "exact_duplicate_extra_rows_excluding_id": int(frame.duplicated([column for column in frame.columns if column != "id"]).sum()),
    "exact_duplicate_extra_feature_rows": int(frame.duplicated(feature_columns).sum()),
    "feature_groups_with_conflicting_binary_labels": conflicting_groups,
    "exact_feature_groups_crossing_splits": cross_split_groups,
    "forbidden_columns_in_features": sorted(EXCLUDED.intersection(feature_columns)),
    "official_test_status": "sealed; cross-source overlap audit deferred until FINAL_TEST_LOCK"
}

with AUDIT.open("x", encoding="utf-8") as stream:
    json.dump(audit, stream, indent=2, ensure_ascii=False)
    stream.write("\n")

print(json.dumps({
    "manifest": str(MANIFEST),
    "audit": str(AUDIT),
    "split_rows": audit["split_rows"],
    "cross_split_groups": cross_split_groups,
}, indent=2))
