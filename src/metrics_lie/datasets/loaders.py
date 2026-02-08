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
    X: Optional[pd.DataFrame] = None
    feature_cols: Optional[list[str]] = None


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
    *,
    feature_cols: list[str] | None = None,
    require_features: bool = False,
    allow_missing_score: bool = False,
) -> LoadedBinaryDataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    df = pd.read_csv(p)

    if y_true_col not in df.columns:
        raise ValueError(f"Missing required column '{y_true_col}'. Available: {list(df.columns)}")
    if y_score_col not in df.columns:
        if allow_missing_score:
            df[y_score_col] = 0.0
        else:
            raise ValueError(f"Missing required column '{y_score_col}'. Available: {list(df.columns)}")

    y_true = df[y_true_col]
    y_score = df[y_score_col]

    _validate_binary_labels(y_true, y_true_col)
    _validate_probability_series(y_score, y_score_col)

    subgroup = None
    if subgroup_col:
        if subgroup_col not in df.columns:
            raise ValueError(f"Missing subgroup column '{subgroup_col}'. Available: {list(df.columns)}")
        subgroup = df[subgroup_col]

    X = None
    resolved_feature_cols = None
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}. Available: {list(df.columns)}")
        resolved_feature_cols = list(feature_cols)
        X = df[resolved_feature_cols]
    elif require_features:
        excluded = {y_true_col, y_score_col}
        if subgroup_col:
            excluded.add(subgroup_col)
        inferred = [c for c in df.columns if c not in excluded]
        if not inferred:
            raise ValueError(
                "No feature columns inferred. Provide feature_cols explicitly."
            )
        resolved_feature_cols = inferred
        X = df[resolved_feature_cols]

    return LoadedBinaryDataset(
        y_true=y_true.astype(int),
        y_score=y_score.astype(float),
        subgroup=subgroup,
        X=X,
        feature_cols=resolved_feature_cols,
    )
