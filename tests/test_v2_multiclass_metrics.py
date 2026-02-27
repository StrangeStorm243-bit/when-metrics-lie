"""Tests for multiclass metric functions."""
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.metrics.core import METRICS


@pytest.fixture
def multiclass_data():
    """3-class classification data."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 0, 2, 2, 0])
    # Probability matrix (10 samples, 3 classes)
    y_proba = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.5, 0.2],
        [0.1, 0.8, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.1, 0.8],
        [0.5, 0.2, 0.3],
        [0.8, 0.1, 0.1],
        [0.2, 0.2, 0.6],
        [0.1, 0.2, 0.7],
        [0.6, 0.3, 0.1],
    ])
    return y_true, y_pred, y_proba


def test_macro_f1_registered():
    assert "macro_f1" in METRICS


def test_weighted_f1_registered():
    assert "weighted_f1" in METRICS


def test_macro_precision_registered():
    assert "macro_precision" in METRICS


def test_macro_recall_registered():
    assert "macro_recall" in METRICS


def test_macro_auc_registered():
    assert "macro_auc" in METRICS


def test_cohens_kappa_registered():
    assert "cohens_kappa" in METRICS


def test_top_k_accuracy_registered():
    assert "top_k_accuracy" in METRICS


def test_macro_f1_value(multiclass_data):
    y_true, y_pred, _ = multiclass_data
    fn = METRICS["macro_f1"]
    result = fn(y_true, y_pred)
    assert 0.0 <= result <= 1.0


def test_weighted_f1_value(multiclass_data):
    y_true, y_pred, _ = multiclass_data
    fn = METRICS["weighted_f1"]
    result = fn(y_true, y_pred)
    assert 0.0 <= result <= 1.0


def test_macro_auc_value(multiclass_data):
    y_true, _, y_proba = multiclass_data
    fn = METRICS["macro_auc"]
    result = fn(y_true, y_proba)
    assert 0.0 <= result <= 1.0


def test_cohens_kappa_value(multiclass_data):
    y_true, y_pred, _ = multiclass_data
    fn = METRICS["cohens_kappa"]
    result = fn(y_true, y_pred)
    assert -1.0 <= result <= 1.0


def test_top_k_accuracy_value(multiclass_data):
    y_true, _, y_proba = multiclass_data
    fn = METRICS["top_k_accuracy"]
    # top-2 accuracy should be >= top-1 accuracy
    result = fn(y_true, y_proba)
    assert 0.0 <= result <= 1.0
