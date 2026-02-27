"""Tests for subgroup diagnostics with multiclass metrics."""
from __future__ import annotations
import numpy as np
from metrics_lie.diagnostics.subgroups import safe_metric_for_group
from metrics_lie.metrics.core import METRICS

def test_safe_metric_macro_f1_multiclass():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 1, 1, 0, 0, 2, 2, 0])
    result = safe_metric_for_group("macro_f1", METRICS["macro_f1"], y_true, y_pred)
    assert result is not None
    assert 0.0 <= result <= 1.0

def test_safe_metric_macro_auc_needs_both_classes():
    y_true = np.array([0, 0, 0])
    y_proba = np.array([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.7, 0.2, 0.1]])
    result = safe_metric_for_group("macro_auc", METRICS["macro_auc"], y_true, y_proba)
    assert result is None

def test_safe_metric_regression_mae():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
    result = safe_metric_for_group("mae", METRICS["mae"], y_true, y_pred)
    assert result is not None
    assert result > 0
