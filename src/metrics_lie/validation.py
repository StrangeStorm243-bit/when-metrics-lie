"""Shared data validation helpers.

These functions are the single source of truth for array-level validation
used by both the dataset loader (loaders.py) and the surface validator
(model/surface.py).
"""
from __future__ import annotations

import numpy as np


def validate_no_nan(values: np.ndarray, name: str) -> None:
    """Raise ValueError if *values* contains any NaN."""
    if np.isnan(values).any():
        raise ValueError(f"{name} contains NaNs.")


def validate_no_inf(values: np.ndarray, name: str) -> None:
    """Raise ValueError if *values* contains any Inf."""
    if np.isinf(values).any():
        raise ValueError(f"{name} contains Inf values.")


def validate_numeric_dtype(values: np.ndarray, name: str) -> None:
    """Raise ValueError if *values* is not a numeric dtype."""
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{name} must be numeric. Got dtype: {values.dtype}")


def validate_binary_labels(values: np.ndarray, name: str) -> None:
    """Raise ValueError if *values* is not binary {0, 1}."""
    validate_no_nan(values, name)
    uniq = set(np.unique(values).tolist())
    if not uniq.issubset({0, 1, False, True}):
        raise ValueError(
            f"{name} must be binary (0/1). Found unique values: {sorted(list(uniq))}"
        )


def validate_labels(values: np.ndarray, name: str) -> None:
    """Raise ValueError if *values* contains NaN. Accepts any integer labels."""
    validate_no_nan(values, name)


def validate_probability_range(values: np.ndarray, name: str) -> None:
    """Raise ValueError if *values* has entries outside [0, 1]."""
    validate_no_nan(values, name)
    out_of_range = (values < 0) | (values > 1)
    if np.any(out_of_range):
        bad = values[out_of_range][:5].tolist()
        raise ValueError(
            f"{name} must be in [0, 1]. Example bad values: {bad}"
        )
