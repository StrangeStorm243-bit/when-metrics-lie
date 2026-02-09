"""
Tiny binary classifier for Phase 5 examples (LAZY, no import-time training).

This exists purely to exercise the Phase 5 model-adapter pipeline end-to-end.
It trains a LogisticRegression on data/demo_binary.csv using y_score as the sole
feature (not a meaningful ML model).

Key property: importing this module should NOT train the model.
Training happens on first predict/predict_proba call.

The lazy model object is exposed as MODULE-level ``MODEL`` so the Phase 5 example
spec can reference it via import_path: "examples.demo_model:MODEL".

Requirements: sklearn, pandas, numpy (already in project deps).
Runtime: training is tiny (< 0.5s on modern machines).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression


_REPO_ROOT = Path(__file__).resolve().parent.parent
_CSV_PATH = _REPO_ROOT / "data" / "demo_binary.csv"


def _train_model() -> LogisticRegression:
    df = pd.read_csv(_CSV_PATH)
    X = df[["y_score"]].to_numpy(dtype=float)
    y = df["y_true"].to_numpy(dtype=int)
    # Deterministic, CPU-only.
    return LogisticRegression(random_state=42).fit(X, y)


class LazyDemoModel(BaseEstimator):
    """
    Minimal sklearn-like model wrapper that trains lazily on first use.
    Implements predict_proba/predict, which is enough for ModelAdapter to
    generate prediction surfaces.
    """

    def __init__(self) -> None:
        self._model: Optional[LogisticRegression] = None

    def _ensure_model(self) -> LogisticRegression:
        if self._model is None:
            self._model = _train_model()
        return self._model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        model = self._ensure_model()
        return model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        model = self._ensure_model()
        return model.predict(X)


# Object for import-path usage: "examples.demo_model:MODEL"
MODEL = LazyDemoModel()

