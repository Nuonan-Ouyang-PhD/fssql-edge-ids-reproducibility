#!/usr/bin/env python3
"""Train and validate the four clean-room revision action models."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist, pdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
SPLITS = ROOT / "data" / "SPLIT_MANIFEST.csv"
PREPROCESSOR = ROOT / "preprocessing" / "preprocessor.joblib"
OUTPUT = ROOT / "models" / "artifacts"
SEED = 20260909
EXCLUDED = ["id", "attack_cat", "label"]


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


class IncrementalSVDD:
    """Clean implementation of the published Gaussian-kernel update."""

    def __init__(self, sigma: float, eps_close: float = 1e-8, eps_outlier: float = 1e-8):
        self.sigma = float(sigma)
        self.eps_close = eps_close
        self.eps_outlier = eps_outlier

    def fit(self, data: np.ndarray) -> "IncrementalSVDD":
        self.inverse = np.ones((1, 1), dtype=np.float64)
        self.alpha = np.ones(1, dtype=np.float64)
        self.support_vectors = data[:1].astype(np.float64, copy=True)
        self.boundary = 1.0
        for point in data[1:]:
            similarity = self._similarity(point[None, :], self.support_vectors)[0]
            maximum = float(similarity.max())
            violation = self.boundary - float(self.alpha @ similarity)
            if maximum < self.eps_outlier or maximum > 1.0 - self.eps_close or violation <= 0:
                continue
            projection = self.inverse @ similarity
            beta = 1.0 - float(similarity @ projection)
            if beta <= 1e-12:
                continue
            upper_left = self.inverse + np.outer(projection, projection) / beta
            off_diagonal = -projection[:, None] / beta
            self.inverse = np.block([
                [upper_left, off_diagonal],
                [off_diagonal.T, np.array([[1.0 / beta]])],
            ])
            self.support_vectors = np.vstack((self.support_vectors, point))
            self.alpha = self.inverse.sum(axis=1)
            while float(self.alpha.min()) < 0 and len(self.alpha) > 1:
                remove = int(self.alpha.argmin())
                keep = np.arange(len(self.alpha)) != remove
                self.support_vectors = self.support_vectors[keep]
                kernel = self._similarity(self.support_vectors, self.support_vectors)
                self.inverse = np.linalg.inv(kernel + np.eye(len(kernel)) * 1e-10)
                self.alpha = self.inverse.sum(axis=1)
            total = float(self.alpha.sum())
            self.boundary = 1.0 / total
            self.alpha /= total
        return self

    def raw_score(self, data: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        scores = []
        for start in range(0, len(data), batch_size):
            similarity = self._similarity(
                data[start:start + batch_size], self.support_vectors
            )
            scores.append(self.boundary - similarity @ self.alpha)
        return np.concatenate(scores)

    def _similarity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        distances = cdist(left, right, metric="sqeuclidean")
        return np.exp(-distances / (2.0 * self.sigma * self.sigma))


class LucidRevision(nn.Module):
    def __init__(self, kernels: int = 32, dropout: float = 0.2):
        super().__init__()
        self.convolution = nn.Conv1d(1, kernels, kernel_size=3)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(kernels, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.convolution(features[:, None, :]))
        pooled = torch.amax(hidden, dim=2)
        return self.output(self.dropout(pooled)).squeeze(1)


class TinyDLRevision(nn.Module):
    def __init__(self, inputs: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputs, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(1)


def train_torch_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    learning_rate: float = 1e-3,
    epochs: int = 20,
    patience: int = 4,
    batch_size: int = 1024,
) -> tuple[nn.Module, list[dict[str, float]]]:
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    positive_weight = float((train_y == 0).sum() / (train_y == 1).sum())
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    validation_tensor = torch.from_numpy(validation_x)
    validation_target = torch.from_numpy(validation_y.astype(np.float32))
    best_state = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        training_loss = 0.0
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), labels)
            loss.backward()
            optimizer.step()
            training_loss += float(loss.detach()) * len(features)
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(validation_tensor), validation_target))
        history.append({
            "epoch": epoch,
            "train_loss": training_loss / len(train_x),
            "validation_loss": validation_loss,
        })
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def predict_torch(model: nn.Module, features: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            logits = model(torch.from_numpy(features[start:start + batch_size]))
            output.append(torch.sigmoid(logits).numpy())
    return np.concatenate(output)


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    best = None
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = probabilities >= threshold
        score = f1_score(labels, predictions, zero_division=0)
        candidate = (score, -threshold, threshold, predictions)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    _, _, threshold, predictions = best
    return float(threshold), {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def save_torch_portable(model: nn.Module, prefix: str) -> list[Path]:
    state_path = OUTPUT / f"{prefix}.pt"
    npz_path = OUTPUT / f"{prefix}.npz"
    torch.save(model.state_dict(), state_path)
    arrays = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    np.savez_compressed(npz_path, **arrays)
    return [state_path, npz_path]


def train_elm(features: np.ndarray, labels: np.ndarray, hidden: int, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED + 300)
    weights = rng.normal(0.0, 1.0 / math.sqrt(features.shape[1]), (features.shape[1], hidden))
    bias = rng.normal(0.0, 0.1, hidden)
    gram = np.eye(hidden, dtype=np.float64) * ridge
    target = np.zeros(hidden, dtype=np.float64)
    for start in range(0, len(features), 4096):
        hidden_values = np.tanh(features[start:start + 4096] @ weights + bias)
        gram += hidden_values.T @ hidden_values
        target += hidden_values.T @ labels[start:start + 4096]
    output_weights = np.linalg.solve(gram, target)
    return weights.astype(np.float32), bias.astype(np.float32), output_weights.astype(np.float32)


def elm_score(features: np.ndarray, weights: np.ndarray, bias: np.ndarray, output_weights: np.ndarray) -> np.ndarray:
    output = []
    for start in range(0, len(features), 4096):
        hidden = np.tanh(features[start:start + 4096] @ weights + bias)
        output.append(hidden @ output_weights)
    return np.concatenate(output)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen model directory: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

    frame = pd.read_csv(SOURCE)
    split_manifest = pd.read_csv(SPLITS)
    preprocessor = joblib.load(PREPROCESSOR)
    feature_frame = frame.drop(columns=EXCLUDED)
    train_mask = split_manifest["split"].eq("model_train").to_numpy()
    validation_mask = split_manifest["split"].eq("validation").to_numpy()
    train_x = preprocessor.transform(feature_frame.loc[train_mask]).toarray().astype(np.float32)
    validation_x = preprocessor.transform(feature_frame.loc[validation_mask]).toarray().astype(np.float32)
    train_y = frame.loc[train_mask, "label"].to_numpy(dtype=np.int64)
    validation_y = frame.loc[validation_mask, "label"].to_numpy(dtype=np.int64)

    rng = np.random.default_rng(SEED)
    train_indices = np.flatnonzero(train_mask)
    normal_indices = train_indices[frame.loc[train_mask, "label"].to_numpy() == 0]
    normal_groups = split_manifest.loc[normal_indices, "feature_group_hex"].to_numpy()
    _, unique_positions = np.unique(normal_groups, return_index=True)
    unique_normal_source_indices = normal_indices[unique_positions]
    svdd_source_indices = rng.choice(unique_normal_source_indices, size=2048, replace=False)
    source_to_train_position = {source: position for position, source in enumerate(train_indices)}
    svdd_train_x = train_x[[source_to_train_position[index] for index in svdd_source_indices]]
    sigma_sample = svdd_train_x[rng.choice(len(svdd_train_x), size=512, replace=False)]
    sigma = float(np.median(pdist(sigma_sample)))

    metrics_rows = []
    hyperparameters = {
        "status": "FROZEN_AFTER_VALIDATION",
        "seed": SEED,
        "input_features": int(train_x.shape[1]),
        "training_rows": int(len(train_x)),
        "validation_rows": int(len(validation_x)),
        "official_test_access": "not read or transformed",
        "actions": {},
    }
    created = []

    started = time.monotonic()
    fisvdd = IncrementalSVDD(sigma=sigma).fit(svdd_train_x)
    calibration_positions = rng.choice(len(train_x), size=20000, replace=False)
    calibration = LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000)
    calibration.fit(
        fisvdd.raw_score(train_x[calibration_positions])[:, None],
        train_y[calibration_positions],
    )
    fisvdd_probability = calibration.predict_proba(fisvdd.raw_score(validation_x)[:, None])[:, 1]
    threshold, metrics = select_threshold(validation_y, fisvdd_probability)
    fisvdd_path = OUTPUT / "fisvdd_revision.npz"
    np.savez_compressed(
        fisvdd_path,
        support_vectors=fisvdd.support_vectors.astype(np.float32),
        alpha=fisvdd.alpha.astype(np.float64),
        boundary=np.array(fisvdd.boundary),
        sigma=np.array(fisvdd.sigma),
        calibration_coef=calibration.coef_.astype(np.float64),
        calibration_intercept=calibration.intercept_.astype(np.float64),
        threshold=np.array(threshold),
    )
    created.append(fisvdd_path)
    hyperparameters["actions"]["FISVDD"] = {
        "normal_only_unique_training_rows": 2048,
        "sigma_rule": "median pairwise distance of seeded 512-row subset",
        "sigma": sigma,
        "support_vectors": int(len(fisvdd.support_vectors)),
        "score_calibration": "balanced logistic regression on seeded 20000 model_train rows",
        "decision_threshold_selected_on_validation": threshold,
        "training_seconds_mac": time.monotonic() - started,
    }
    metrics_rows.append({"action": "FISVDD", "threshold": threshold, **metrics})

    for name, model, settings in [
        ("LUCID", LucidRevision(kernels=32, dropout=0.2), {"kernels": 32, "kernel_size": 3, "dropout": 0.2}),
        ("TinyDL", TinyDLRevision(train_x.shape[1], dropout=0.1), {"hidden_layers": [32, 32], "dropout": 0.1}),
    ]:
        started = time.monotonic()
        trained, history = train_torch_model(
            model, train_x, train_y, validation_x, validation_y
        )
        probabilities = predict_torch(trained, validation_x)
        threshold, metrics = select_threshold(validation_y, probabilities)
        prefix = name.lower() + "_revision"
        paths = save_torch_portable(trained, prefix)
        created.extend(paths)
        history_path = OUTPUT / f"{prefix}_history.json"
        with history_path.open("x", encoding="utf-8") as stream:
            json.dump(history, stream, indent=2)
            stream.write("\n")
        created.append(history_path)
        hyperparameters["actions"][name] = {
            **settings,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 1024,
            "maximum_epochs": 20,
            "early_stopping_patience": 4,
            "epochs_completed": len(history),
            "weighted_binary_cross_entropy": True,
            "decision_threshold_selected_on_validation": threshold,
            "training_seconds_mac": time.monotonic() - started,
        }
        metrics_rows.append({"action": name, "threshold": threshold, **metrics})

    started = time.monotonic()
    elm_weights, elm_bias, elm_output = train_elm(train_x, train_y, hidden=128, ridge=0.01)
    train_elm_output = elm_score(train_x[calibration_positions], elm_weights, elm_bias, elm_output)
    train_svdd_output = fisvdd.raw_score(train_x[calibration_positions])
    fusion = LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000)
    fusion.fit(
        np.column_stack((train_svdd_output, train_elm_output)),
        train_y[calibration_positions],
    )
    validation_elm_output = elm_score(validation_x, elm_weights, elm_bias, elm_output)
    validation_svdd_output = fisvdd.raw_score(validation_x)
    hybrid_probability = fusion.predict_proba(
        np.column_stack((validation_svdd_output, validation_elm_output))
    )[:, 1]
    threshold, metrics = select_threshold(validation_y, hybrid_probability)
    hybrid_path = OUTPUT / "oi_svdd_as_elm_revision.npz"
    np.savez_compressed(
        hybrid_path,
        support_vectors=fisvdd.support_vectors.astype(np.float32),
        alpha=fisvdd.alpha.astype(np.float64),
        boundary=np.array(fisvdd.boundary),
        sigma=np.array(fisvdd.sigma),
        elm_weights=elm_weights,
        elm_bias=elm_bias,
        elm_output=elm_output,
        fusion_coef=fusion.coef_.astype(np.float64),
        fusion_intercept=fusion.intercept_.astype(np.float64),
        threshold=np.array(threshold),
    )
    created.append(hybrid_path)
    hyperparameters["actions"]["OI-SVDD+AS-ELM"] = {
        "svdd_stage": "same frozen FISVDD boundary",
        "elm_hidden_units": 128,
        "elm_activation": "tanh",
        "elm_ridge": 0.01,
        "elm_fit": "seeded random hidden layer; chunk-wise sufficient-statistic accumulation",
        "fusion": "balanced logistic head over SVDD and ELM scores on seeded 20000 model_train rows",
        "online_evaluation_updates": "disabled to prevent label leakage",
        "decision_threshold_selected_on_validation": threshold,
        "training_seconds_mac": time.monotonic() - started,
    }
    metrics_rows.append({"action": "OI-SVDD+AS-ELM", "threshold": threshold, **metrics})

    hyperparameters_path = ROOT / "models" / "MODEL_HYPERPARAMETERS.json"
    with hyperparameters_path.open("x", encoding="utf-8") as stream:
        json.dump(hyperparameters, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    created.append(hyperparameters_path)

    metrics_path = ROOT / "models" / "VALIDATION_METRICS.csv"
    with metrics_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics_rows[0]))
        writer.writeheader()
        writer.writerows(metrics_rows)
    created.append(metrics_path)

    hashes = {
        path.relative_to(ROOT).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(created)
    }
    hashes_path = ROOT / "models" / "MODEL_SHA256.json"
    with hashes_path.open("x", encoding="utf-8") as stream:
        json.dump({
            "status": "DEVELOPMENT_MODELS_FROZEN",
            "official_test_access": "not read or transformed",
            "files": hashes,
        }, stream, indent=2)
        stream.write("\n")

    print(json.dumps({"metrics": metrics_rows, "hashes": str(hashes_path)}, indent=2))


if __name__ == "__main__":
    main()
