#!/usr/bin/env python3
"""Export and verify a pure NumPy form of the frozen preprocessor."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.preprocessor_numpy import PortablePreprocessor  # noqa: E402


SOURCE = ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
SPLITS = ROOT / "data" / "SPLIT_MANIFEST.csv"
SKLEARN_ARTIFACT = ROOT / "preprocessing" / "preprocessor.joblib"
PORTABLE_ARTIFACT = ROOT / "preprocessing" / "preprocessor_portable.npz"
PARITY_OUTPUT = ROOT / "preprocessing" / "PREPROCESSOR_PORTABLE_PARITY.json"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


if PORTABLE_ARTIFACT.exists() or PARITY_OUTPUT.exists():
    raise FileExistsError("refusing to overwrite portable preprocessor outputs")

preprocessor = joblib.load(SKLEARN_ARTIFACT)
numeric_columns = preprocessor.transformers_[0][2]
categorical_columns = preprocessor.transformers_[1][2]
numeric = preprocessor.named_transformers_["numeric"]
categorical = preprocessor.named_transformers_["categorical"]
parameters = {
    "numeric_columns": np.asarray(numeric_columns, dtype=str),
    "categorical_columns": np.asarray(categorical_columns, dtype=str),
    "numeric_mean": numeric.mean_,
    "numeric_scale": numeric.scale_,
    "output_features": np.asarray(len(preprocessor.get_feature_names_out())),
}
for column, categories in zip(categorical_columns, categorical.categories_):
    parameters[f"categories_{column}"] = np.asarray(categories, dtype=str)
np.savez_compressed(PORTABLE_ARTIFACT, **parameters)

frame = pd.read_csv(SOURCE)
splits = pd.read_csv(SPLITS)
sample = frame.loc[splits["split"].eq("validation")].iloc[:512]
feature_frame = sample.drop(columns=["id", "attack_cat", "label"])
expected = preprocessor.transform(feature_frame).toarray().astype(np.float32)
rows = feature_frame.to_dict(orient="records")
observed = PortablePreprocessor(PORTABLE_ARTIFACT).transform(rows)
maximum_difference = float(np.max(np.abs(expected - observed)))
result = {
    "status": "PASS" if maximum_difference <= 1e-6 else "FAIL",
    "scope": "first 512 validation rows; official test not accessed",
    "maximum_absolute_difference": maximum_difference,
    "artifact": PORTABLE_ARTIFACT.relative_to(ROOT).as_posix(),
    "artifact_sha256": sha256(PORTABLE_ARTIFACT),
}
with PARITY_OUTPUT.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps(result, indent=2))
