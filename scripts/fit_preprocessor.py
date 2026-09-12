#!/usr/bin/env python3
"""Fit the frozen shared preprocessor on model_train only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
SPLIT_MANIFEST = ROOT / "data" / "SPLIT_MANIFEST.csv"
OUTPUT_DIR = ROOT / "preprocessing"
MODEL_OUTPUT = OUTPUT_DIR / "preprocessor.joblib"
FEATURE_OUTPUT = OUTPUT_DIR / "FEATURE_NAMES.txt"
PROVENANCE_OUTPUT = ROOT / "data" / "PREPROCESSOR_PROVENANCE.json"
EXCLUDED = {"id", "label", "attack_cat"}


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


for output in (MODEL_OUTPUT, FEATURE_OUTPUT, PROVENANCE_OUTPUT):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {output}")

frame = pd.read_csv(SOURCE)
manifest = pd.read_csv(SPLIT_MANIFEST)
if len(frame) != len(manifest):
    raise RuntimeError("source and split manifest row counts differ")
if not np.array_equal(
    manifest["source_row_index"].to_numpy(), np.arange(len(frame))
):
    raise RuntimeError("split manifest is not aligned to source row order")

feature_columns = [column for column in frame.columns if column not in EXCLUDED]
categorical_columns = [
    column for column in feature_columns if not pd.api.types.is_numeric_dtype(frame[column])
]
numeric_columns = [column for column in feature_columns if column not in categorical_columns]
train_mask = manifest["split"].eq("model_train").to_numpy()
train_features = frame.loc[train_mask, feature_columns]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_columns),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
                dtype=np.float32,
            ),
            categorical_columns,
        ),
    ],
    sparse_threshold=1.0,
    verbose_feature_names_out=True,
)
preprocessor.fit(train_features)
feature_names = list(preprocessor.get_feature_names_out())

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(preprocessor, MODEL_OUTPUT, compress=3)
with FEATURE_OUTPUT.open("x", encoding="utf-8") as stream:
    stream.write("\n".join(feature_names))
    stream.write("\n")

categorical_cardinality = {
    column: int(len(categories))
    for column, categories in zip(
        categorical_columns,
        preprocessor.named_transformers_["categorical"].categories_,
    )
}
provenance = {
    "status": "FROZEN_DEVELOPMENT_PREPROCESSOR",
    "fit_scope": "model_train only",
    "fit_rows": int(train_mask.sum()),
    "source_file": SOURCE.relative_to(ROOT).as_posix(),
    "source_sha256": sha256(SOURCE),
    "split_manifest": SPLIT_MANIFEST.relative_to(ROOT).as_posix(),
    "split_manifest_sha256": sha256(SPLIT_MANIFEST),
    "excluded_columns": sorted(EXCLUDED),
    "numeric_columns": numeric_columns,
    "categorical_columns": categorical_columns,
    "categorical_cardinality_model_train": categorical_cardinality,
    "unknown_category_policy": "ignore and emit all-zero indicators for that field",
    "numeric_transform": "StandardScaler fit on model_train",
    "categorical_transform": "OneHotEncoder fit on model_train",
    "output_feature_count": len(feature_names),
    "output_cast_for_models": "float32 at model input boundary",
    "artifact": MODEL_OUTPUT.relative_to(ROOT).as_posix(),
    "artifact_sha256": sha256(MODEL_OUTPUT),
    "feature_names_file": FEATURE_OUTPUT.relative_to(ROOT).as_posix(),
    "feature_names_sha256": sha256(FEATURE_OUTPUT),
    "software": {
        "python": "3.9.6",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    },
    "official_test_access": "not read or transformed",
}
with PROVENANCE_OUTPUT.open("x", encoding="utf-8") as stream:
    json.dump(provenance, stream, indent=2, ensure_ascii=False)
    stream.write("\n")

print(json.dumps({
    "fit_rows": provenance["fit_rows"],
    "output_feature_count": provenance["output_feature_count"],
    "artifact_sha256": provenance["artifact_sha256"],
}, indent=2))
