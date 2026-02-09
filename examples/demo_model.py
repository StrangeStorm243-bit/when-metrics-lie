"""Tiny fitted binary classifier for Phase 5 examples.

Trains a LogisticRegression on data/demo_binary.csv using y_score as the sole
feature (a demo of the Phase 5 model-adapter pipeline, not a meaningful model).
The fitted estimator is exposed as MODULE-level ``MODEL``.

Requirements: sklearn, pandas, numpy (all already in project deps).
Runtime: < 0.5 s on any modern machine.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CSV_PATH = _REPO_ROOT / "data" / "demo_binary.csv"

def get_model() -> LogisticRegression:
    """Return a fitted sklearn binary classifier — deterministic, CPU-only."""
    df = pd.read_csv(_CSV_PATH)
    X = df[["y_score"]].to_numpy(dtype=float)
    y = df["y_true"].to_numpy(dtype=int)
    return LogisticRegression(random_state=42).fit(X, y)


def _predict_proba(X: np.ndarray) -> np.ndarray:
    # Lazily train at call time to avoid training on import.
    return get_model().predict_proba(X)


# Expose sklearn-like surface methods on the factory for ModelAdapter import path usage.
get_model.predict_proba = _predict_proba  # type: ignore[attr-defined]
