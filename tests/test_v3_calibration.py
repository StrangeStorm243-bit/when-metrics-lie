"""Tests for multiclass calibration functions."""
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.diagnostics.calibration import multiclass_brier_score, multiclass_ece


def test_multiclass_brier_perfect():
    y_true = np.array([0, 1, 2])
    y_proba = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert multiclass_brier_score(y_true, y_proba) == pytest.approx(0.0)


def test_multiclass_brier_worst():
    y_true = np.array([0, 1, 2])
    y_proba = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    assert multiclass_brier_score(y_true, y_proba) > 0.5


def test_multiclass_brier_formula():
    y_true = np.array([0, 1])
    y_proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    # sample 0: (0.7-1)^2+(0.2-0)^2+(0.1-0)^2 = 0.09+0.04+0.01 = 0.14
    # sample 1: (0.1-0)^2+(0.6-1)^2+(0.3-0)^2 = 0.01+0.16+0.09 = 0.26
    # mean = 0.20
    assert multiclass_brier_score(y_true, y_proba) == pytest.approx(0.20)


def test_multiclass_ece_perfect():
    rng = np.random.default_rng(42)
    n, n_classes = 1000, 3
    y_true = rng.integers(0, n_classes, size=n)
    # Build a well-calibrated model: ~90% correct with confidence 0.9
    # and ~10% incorrect (wrong class gets highest prob).
    # This way avg accuracy ≈ confidence in the 0.9 bin → low ECE.
    y_proba = np.full((n, n_classes), 0.05)
    n_wrong = int(n * 0.10)
    wrong_indices = rng.choice(n, size=n_wrong, replace=False)
    for i in range(n):
        if i in wrong_indices:
            # Incorrect: assign 0.9 to a wrong class
            wrong_class = (y_true[i] + 1) % n_classes
            y_proba[i, wrong_class] = 0.9
            y_proba[i, y_true[i]] = 0.05
        else:
            y_proba[i, y_true[i]] = 0.9
        # Normalize remaining
        remaining = 1.0 - 0.9
        other_classes = [k for k in range(n_classes) if y_proba[i, k] != 0.9]
        for k in other_classes:
            y_proba[i, k] = remaining / len(other_classes)
    assert multiclass_ece(y_true, y_proba) < 0.05


def test_multiclass_ece_returns_float():
    y_true = np.array([0, 1, 2, 0])
    y_proba = np.array(
        [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
    )
    ece = multiclass_ece(y_true, y_proba)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0
