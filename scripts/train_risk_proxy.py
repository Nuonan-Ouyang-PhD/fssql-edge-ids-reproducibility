#!/usr/bin/env python3
"""Train the separate lightweight risk proxy without final-test access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
SPLITS = ROOT / "data" / "SPLIT_MANIFEST.csv"
ARTIFACT = ROOT / "risk_proxy" / "risk_proxy.npz"
SPEC = ROOT / "models" / "RISK_PROXY_SPEC.json"
SEED = 20260909
FEATURES = [
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "ct_state_ttl",
    "ct_dst_src_ltm",
    "is_sm_ips_ports",
]


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    output = np.empty_like(values, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


if ARTIFACT.exists() or SPEC.exists():
    raise FileExistsError("refusing to overwrite frozen risk-proxy output")

frame = pd.read_csv(SOURCE)
manifest = pd.read_csv(SPLITS)
fit_mask = manifest["split"].eq("model_train") & manifest["fold"].isin([0, 1])
calibration_mask = manifest["split"].eq("model_train") & manifest["fold"].eq(2)
validation_mask = manifest["split"].eq("validation")

scaler = StandardScaler().fit(frame.loc[fit_mask, FEATURES])
fit_x = scaler.transform(frame.loc[fit_mask, FEATURES])
fit_y = frame.loc[fit_mask, "label"].to_numpy(dtype=np.int64)
base = LogisticRegression(
    C=1.0,
    class_weight="balanced",
    max_iter=1000,
    random_state=SEED,
).fit(fit_x, fit_y)

calibration_score = base.decision_function(
    scaler.transform(frame.loc[calibration_mask, FEATURES])
)
calibration_y = frame.loc[calibration_mask, "label"].to_numpy(dtype=np.int64)
platt = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(
    calibration_score[:, None], calibration_y
)

validation_score = base.decision_function(
    scaler.transform(frame.loc[validation_mask, FEATURES])
)
validation_probability = sigmoid(
    validation_score * platt.coef_.reshape(-1)[0] + platt.intercept_[0]
)
validation_y = frame.loc[validation_mask, "label"].to_numpy(dtype=np.int64)

ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    ARTIFACT,
    feature_names=np.asarray(FEATURES),
    scaler_mean=scaler.mean_.astype(np.float64),
    scaler_scale=scaler.scale_.astype(np.float64),
    base_coef=base.coef_.astype(np.float64),
    base_intercept=base.intercept_.astype(np.float64),
    calibration_coef=platt.coef_.astype(np.float64),
    calibration_intercept=platt.intercept_.astype(np.float64),
)
specification = {
    "status": "FROZEN_DEVELOPMENT_RISK_PROXY",
    "purpose": "independent cheap p_t estimator; not a member of the four-action IDS pool",
    "features": FEATURES,
    "forbidden_inputs": ["id", "attack_cat", "label", "official-test labels"],
    "base_model": "balanced logistic regression",
    "base_fit_scope": "model_train folds 0 and 1",
    "base_fit_rows": int(fit_mask.sum()),
    "probability_calibration": "Platt logistic calibration on disjoint model_train fold 2",
    "calibration_rows": int(calibration_mask.sum()),
    "validation_scope": "validation fold 3; reporting only",
    "validation_rows": int(validation_mask.sum()),
    "validation_metrics": {
        "roc_auc": float(roc_auc_score(validation_y, validation_probability)),
        "brier_score": float(brier_score_loss(validation_y, validation_probability)),
        "log_loss": float(log_loss(validation_y, validation_probability)),
    },
    "artifact": ARTIFACT.relative_to(ROOT).as_posix(),
    "artifact_sha256": sha256(ARTIFACT),
    "official_test_access": "not read or transformed",
    "remaining_acceptance": "measure Pi-local inference overhead during device calibration",
}
with SPEC.open("x", encoding="utf-8") as stream:
    json.dump(specification, stream, indent=2)
    stream.write("\n")
print(json.dumps(specification, indent=2))
