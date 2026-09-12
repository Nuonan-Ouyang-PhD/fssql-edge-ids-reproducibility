#!/usr/bin/env python3
"""NumPy inference for the frozen lightweight risk proxy."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class RiskProxy:
    def __init__(self, artifact: str | Path):
        self.parameters = dict(np.load(artifact, allow_pickle=False))
        self.feature_names = self.parameters["feature_names"].tolist()

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        scaled = (features - self.parameters["scaler_mean"]) / self.parameters["scaler_scale"]
        base = scaled @ self.parameters["base_coef"].reshape(-1)
        base += self.parameters["base_intercept"].reshape(-1)[0]
        logits = base * self.parameters["calibration_coef"].reshape(-1)[0]
        logits += self.parameters["calibration_intercept"].reshape(-1)[0]
        positive = logits >= 0
        output = np.empty_like(logits)
        output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exponential = np.exp(logits[~positive])
        output[~positive] = exponential / (1.0 + exponential)
        return output
