"""Tests for compute_metric routing with new metric categories."""
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.metrics.core import compute_metric, METRICS


def test_compute_regression_metric():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.0, 3.2])
    result = compute_metric("mae", METRICS["mae"], y_true, y_pred)
    assert result == pytest.approx(0.1, abs=0.01)


def test_compute_multiclass_label_metric():
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 1, 0])
    result = compute_metric("macro_f1", METRICS["macro_f1"], y_true, y_pred)
    assert 0.0 <= result <= 1.0


def test_compute_binary_threshold_metric_still_works():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.2, 0.4, 0.6, 0.8])
    result = compute_metric("accuracy", METRICS["accuracy"], y_true, y_score, threshold=0.5)
    assert result == 1.0
