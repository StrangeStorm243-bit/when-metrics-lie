from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class LoadedBinaryDataset:
    y_true: pd.Series
    y_score: pd.Series
    subgroup: Optional[pd.Series] = None


def _validate_probability_series(s: pd.Series, name: str) -> None:
    if s.isna().any():
        raise ValueError(f"{name} contains NaNs.")
    if ((s < 0) | (s > 1)).any():
        bad = s[(s < 0) | (s > 1)].head(5).tolist()
        raise ValueError(f"{name} must be in [0, 1]. Example bad values: {bad}")


def _validate_binary_labels(s: pd.Series, name: str) -> None:
    if s.isna().any():
        raise ValueError(f"{name} contains NaNs.")
    uniq = set(s.unique().tolist())
    if not uniq.issubset({0, 1, False, True}):
        raise ValueError(f"{name} must be binary (0/1). Found unique values: {sorted(list(uniq))}")


def load_binary_csv(
    path: str,
    y_true_col: str,
    y_score_col: str,
    subgroup_col: str | None = None,
) -> LoadedBinaryDataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    df = pd.read_csv(p)

    for col in [y_true_col, y_score_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'. Available: {list(df.columns)}")

    y_true = df[y_true_col]
    y_score = df[y_score_col]

    _validate_binary_labels(y_true, y_true_col)
    _validate_probability_series(y_score, y_score_col)

    subgroup = None
    if subgroup_col:
        if subgroup_col not in df.columns:
            raise ValueError(f"Missing subgroup column '{subgroup_col}'. Available: {list(df.columns)}")
        subgroup = df[subgroup_col]

    return LoadedBinaryDataset(y_true=y_true.astype(int), y_score=y_score.astype(float), subgroup=subgroup)
