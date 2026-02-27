"""Tests for multiclass metric functions."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score as sklearn_f1,
    precision_score as sklearn_precision,
    recall_score as sklearn_recall,
    roc_auc_score as sklearn_roc_auc,
    top_k_accuracy_score as sklearn_top_k,
)

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


@pytest.mark.parametrize("metric_id", [
    "macro_f1", "weighted_f1", "macro_precision", "macro_recall",
    "macro_auc", "cohens_kappa", "top_k_accuracy",
])
def test_multiclass_metric_registered(metric_id):
    assert metric_id in METRICS


def test_macro_f1_value(multiclass_data):
    y_true, y_pred, _ = multiclass_data
    expected = sklearn_f1(y_true, y_pred, average="macro", zero_division=0)
    result = METRICS["macro_f1"](y_true, y_pred)
    assert result == pytest.approx(expected)


def test_weighted_f1_value(multiclass_data):
    y_true, y_pred, _ = multiclass_data
    expected = sklearn_f1(y_true, y_pred, average="weighted", zero_division=0)
    result = METRICS["weighted_f1"](y_true, y_pred)
    assert result == pytest.approx(expected)


def test_macro_precision_value(multiclass_data):
    y_true, y_pred, _ = multiclass_data
    expected = sklearn_precision(y_true, y_pred, average="macro", zero_division=0)
    result = METRICS["macro_precision"](y_true, y_pred)
    assert result == pytest.approx(expected)


def test_macro_recall_value(multiclass_data):
    y_true, y_pred, _ = multiclass_data
    expected = sklearn_recall(y_true, y_pred, average="macro", zero_division=0)
    result = METRICS["macro_recall"](y_true, y_pred)
    assert result == pytest.approx(expected)


def test_macro_auc_value(multiclass_data):
    y_true, _, y_proba = multiclass_data
    expected = sklearn_roc_auc(y_true, y_proba, multi_class="ovr", average="macro")
    result = METRICS["macro_auc"](y_true, y_proba)
    assert result == pytest.approx(expected)


def test_cohens_kappa_value(multiclass_data):
    y_true, y_pred, _ = multiclass_data
    expected = cohen_kappa_score(y_true, y_pred)
    result = METRICS["cohens_kappa"](y_true, y_pred)
    assert result == pytest.approx(expected)


def test_top_k_accuracy_value(multiclass_data):
    y_true, _, y_proba = multiclass_data
    expected = sklearn_top_k(y_true, y_proba, k=2)
    result = METRICS["top_k_accuracy"](y_true, y_proba)
    assert result == pytest.approx(expected)


def test_macro_auc_rejects_1d_input(multiclass_data):
    y_true, y_pred, _ = multiclass_data
    with pytest.raises(ValueError, match="requires 2D probability matrix"):
        METRICS["macro_auc"](y_true, y_pred)
