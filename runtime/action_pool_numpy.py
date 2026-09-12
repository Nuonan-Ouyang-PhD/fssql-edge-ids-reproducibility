#!/usr/bin/env python3
"""Dependency-light NumPy inference for the four frozen action artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


class NumpyActionModel:
    def __init__(self, artifact: str | Path):
        self.path = Path(artifact)
        self.parameters = dict(np.load(self.path, allow_pickle=False))
        self.kind = self.path.stem

    @property
    def threshold(self) -> float:
        # Some frozen probability artifacts intentionally omit a serialized
        # threshold; the preregistered decision threshold is 0.5 in that case.
        return float(self.parameters.get("threshold", 0.5))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if self.kind == "fisvdd_revision":
            return self._fisvdd_probability(features)
        if self.kind == "lucid_revision":
            return self._lucid_probability(features)
        if self.kind == "tinydl_revision":
            return self._tinydl_probability(features)
        if self.kind == "oi_svdd_as_elm_revision":
            return self._hybrid_probability(features)
        raise ValueError(f"unknown action artifact: {self.path.name}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.predict_proba(features) >= self.threshold).astype(np.int8)

    def _svdd_score(self, features: np.ndarray) -> np.ndarray:
        vectors = self.parameters["support_vectors"]
        sigma = float(self.parameters["sigma"])
        scores = []
        for start in range(0, len(features), 1024):
            batch = features[start:start + 1024]
            squared_distance = (
                np.square(batch).sum(axis=1)[:, None]
                + np.square(vectors).sum(axis=1)[None, :]
                - 2.0 * batch @ vectors.T
            )
            similarities = np.exp(
                -np.maximum(squared_distance, 0.0) / (2.0 * sigma * sigma)
            )
            scores.append(
                float(self.parameters["boundary"])
                - similarities @ self.parameters["alpha"]
            )
        return np.concatenate(scores)

    def _fisvdd_probability(self, features: np.ndarray) -> np.ndarray:
        raw = self._svdd_score(features)
        logits = raw * self.parameters["calibration_coef"].reshape(-1)[0]
        logits += self.parameters["calibration_intercept"].reshape(-1)[0]
        return sigmoid(logits)

    def _lucid_probability(self, features: np.ndarray) -> np.ndarray:
        kernels = self.parameters["convolution.weight"][:, 0, :]
        bias = self.parameters["convolution.bias"]
        windows = sliding_window_view(features, window_shape=3, axis=1)
        convolution = np.einsum("nlk,fk->nfl", windows, kernels) + bias[None, :, None]
        pooled = np.maximum(convolution, 0).max(axis=2)
        logits = pooled @ self.parameters["output.weight"].reshape(-1)
        logits += self.parameters["output.bias"].reshape(-1)[0]
        return sigmoid(logits)

    def _tinydl_probability(self, features: np.ndarray) -> np.ndarray:
        hidden = np.maximum(
            features @ self.parameters["layers.0.weight"].T + self.parameters["layers.0.bias"],
            0,
        )
        hidden = np.maximum(
            hidden @ self.parameters["layers.3.weight"].T + self.parameters["layers.3.bias"],
            0,
        )
        logits = hidden @ self.parameters["layers.6.weight"].reshape(-1)
        logits += self.parameters["layers.6.bias"].reshape(-1)[0]
        return sigmoid(logits)

    def _hybrid_probability(self, features: np.ndarray) -> np.ndarray:
        svdd = self._svdd_score(features)
        hidden = np.tanh(
            features @ self.parameters["elm_weights"] + self.parameters["elm_bias"]
        )
        elm = hidden @ self.parameters["elm_output"]
        scores = np.column_stack((svdd, elm))
        logits = scores @ self.parameters["fusion_coef"].reshape(-1)
        logits += self.parameters["fusion_intercept"].reshape(-1)[0]
        return sigmoid(logits)
