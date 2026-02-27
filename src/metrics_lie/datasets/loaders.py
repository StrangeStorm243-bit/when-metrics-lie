from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from metrics_lie.validation import (
    validate_binary_labels,
    validate_no_inf,
    validate_no_nan,
    validate_probability_range,
)


@dataclass(frozen=True)
class LoadedBinaryDataset:
    y_true: pd.Series
    y_score: pd.Series
    subgroup: Optional[pd.Series] = None
    X: Optional[pd.DataFrame] = None
    feature_cols: Optional[list[str]] = None


def _validate_probability_series(s: pd.Series, name: str) -> None:
    arr = np.asarray(s, dtype=float)
    validate_probability_range(arr, name)


def _validate_numeric_series(s: pd.Series, name: str) -> None:
    """Require numeric dtype, no NaN, no Inf (for score surfaces)."""
    if not pd.api.types.is_numeric_dtype(s):
        raise ValueError(f"{name} must be numeric. Got dtype: {s.dtype}")
    arr = np.asarray(s, dtype=float)
    validate_no_nan(arr, name)
    validate_no_inf(arr, name)


def _validate_binary_labels(s: pd.Series, name: str) -> None:
    if s.isna().any():
        raise ValueError(f"{name} contains NaNs.")
    arr = np.asarray(s)
    validate_binary_labels(arr, name)


def load_binary_csv(
    path: str,
    y_true_col: str,
    y_score_col: str,
    subgroup_col: str | None = None,
    *,
    feature_cols: list[str] | None = None,
    require_features: bool = False,
    allow_missing_score: bool = False,
    score_validation: Literal["probability", "score", "label", "none"] = "probability",
) -> LoadedBinaryDataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    df = pd.read_csv(p)

    if y_true_col not in df.columns:
        raise ValueError(
            f"Missing required column '{y_true_col}'. Available: {list(df.columns)}"
        )
    if y_score_col not in df.columns:
        if allow_missing_score:
            df[y_score_col] = 0.0
        else:
            raise ValueError(
                f"Missing required column '{y_score_col}'. Available: {list(df.columns)}"
            )

    y_true = df[y_true_col]
    y_score = df[y_score_col]

    _validate_binary_labels(y_true, y_true_col)
    if score_validation == "score":
        _validate_numeric_series(y_score, y_score_col)
    elif score_validation == "label":
        _validate_binary_labels(y_score, y_score_col)
    elif score_validation in ("probability", "none"):
        if score_validation == "probability":
            _validate_probability_series(y_score, y_score_col)

    subgroup = None
    if subgroup_col:
        if subgroup_col not in df.columns:
            raise ValueError(
                f"Missing subgroup column '{subgroup_col}'. Available: {list(df.columns)}"
            )
        subgroup = df[subgroup_col]

    X = None
    resolved_feature_cols = None
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns: {missing}. Available: {list(df.columns)}"
            )
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


@dataclass(frozen=True)
class LoadedDataset:
    y_true: pd.Series
    y_score: pd.Series
    subgroup: Optional[pd.Series] = None
    X: Optional[pd.DataFrame] = None
    feature_cols: Optional[list[str]] = None


def load_dataset(
    path: str,
    y_true_col: str,
    y_score_col: str,
    task_type: str = "binary_classification",
    subgroup_col: str | None = None,
    *,
    feature_cols: list[str] | None = None,
    require_features: bool = False,
    allow_missing_score: bool = False,
) -> LoadedDataset:
    """Load a dataset for any task type."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    df = pd.read_csv(p)

    if y_true_col not in df.columns:
        raise ValueError(
            f"Missing required column '{y_true_col}'. Available: {list(df.columns)}"
        )
    if y_score_col not in df.columns:
        if allow_missing_score:
            df[y_score_col] = 0.0
        else:
            raise ValueError(
                f"Missing required column '{y_score_col}'. Available: {list(df.columns)}"
            )

    y_true = df[y_true_col]
    y_score = df[y_score_col]

    # Task-type-specific validation
    if task_type == "binary_classification":
        _validate_binary_labels(y_true, y_true_col)
        _validate_probability_series(y_score, y_score_col)
    elif task_type in ("multiclass_classification", "multilabel_classification"):
        arr = np.asarray(y_true)
        validate_no_nan(arr, y_true_col)
    elif task_type == "regression":
        arr_t = np.asarray(y_true, dtype=float)
        validate_no_nan(arr_t, y_true_col)
        validate_no_inf(arr_t, y_true_col)
        arr_s = np.asarray(y_score, dtype=float)
        validate_no_nan(arr_s, y_score_col)
        validate_no_inf(arr_s, y_score_col)
    else:
        arr = np.asarray(y_true)
        validate_no_nan(arr, y_true_col)

    subgroup = None
    if subgroup_col:
        if subgroup_col not in df.columns:
            raise ValueError(
                f"Missing subgroup column '{subgroup_col}'. Available: {list(df.columns)}"
            )
        subgroup = df[subgroup_col]

    X = None
    resolved_feature_cols = None
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns: {missing}. Available: {list(df.columns)}"
            )
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

    return LoadedDataset(
        y_true=y_true,
        y_score=y_score,
        subgroup=subgroup,
        X=X,
        feature_cols=resolved_feature_cols,
    )
