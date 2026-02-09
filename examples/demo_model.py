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

_df = pd.read_csv(_CSV_PATH)
_X = _df[["y_score"]].to_numpy(dtype=float)
_y = _df["y_true"].to_numpy(dtype=int)

MODEL: LogisticRegression = LogisticRegression(random_state=42).fit(_X, _y)
"""Fitted sklearn binary classifier — deterministic, CPU-only."""
