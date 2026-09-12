#!/usr/bin/env python3
"""Pure NumPy transform matching the frozen scikit-learn preprocessor."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


class PortablePreprocessor:
    def __init__(self, artifact: str | Path):
        parameters = dict(np.load(artifact, allow_pickle=False))
        self.numeric_columns = parameters["numeric_columns"].tolist()
        self.categorical_columns = parameters["categorical_columns"].tolist()
        self.numeric_mean = parameters["numeric_mean"].astype(np.float64)
        self.numeric_scale = parameters["numeric_scale"].astype(np.float64)
        self.categories = {
            column: parameters[f"categories_{column}"].tolist()
            for column in self.categorical_columns
        }
        self.category_to_index = {
            column: {
                str(category): index for index, category in enumerate(self.categories[column])
            }
            for column in self.categorical_columns
        }
        self.output_features = int(parameters["output_features"])

    def transform(self, rows: Sequence[Mapping[str, object]]) -> np.ndarray:
        output = np.zeros((len(rows), self.output_features), dtype=np.float32)
        numeric = np.asarray([
            [float(row[column]) for column in self.numeric_columns]
            for row in rows
        ], dtype=np.float64)
        output[:, :len(self.numeric_columns)] = (
            (numeric - self.numeric_mean) / self.numeric_scale
        ).astype(np.float32)
        offset = len(self.numeric_columns)
        for column in self.categorical_columns:
            category_to_index = self.category_to_index[column]
            for row_index, row in enumerate(rows):
                category_index = category_to_index.get(str(row[column]))
                if category_index is not None:
                    output[row_index, offset + category_index] = 1.0
            offset += len(category_to_index)
        return output
