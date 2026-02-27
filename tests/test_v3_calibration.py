"""Tests for multiclass calibration functions."""
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.diagnostics.calibration import multiclass_brier_score


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
