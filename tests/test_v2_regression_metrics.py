"""Tests for regression metric functions."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    max_error as sklearn_max_error,
    r2_score,
)

from metrics_lie.metrics.core import METRICS, REGRESSION_METRICS


@pytest.fixture
def regression_data():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
    return y_true, y_pred


@pytest.mark.parametrize("metric_id", ["mae", "mse", "rmse", "r2", "max_error"])
def test_regression_metric_registered(metric_id):
    assert metric_id in METRICS


def test_regression_metrics_category():
    expected = {"mae", "mse", "rmse", "r2", "max_error"}
    assert expected == REGRESSION_METRICS


def test_mae_value(regression_data):
    y_true, y_pred = regression_data
    expected = mean_absolute_error(y_true, y_pred)
    result = METRICS["mae"](y_true, y_pred)
    assert result == pytest.approx(expected)


def test_mse_value(regression_data):
    y_true, y_pred = regression_data
    expected = mean_squared_error(y_true, y_pred)
    result = METRICS["mse"](y_true, y_pred)
    assert result == pytest.approx(expected)


def test_rmse_value(regression_data):
    y_true, y_pred = regression_data
    mse = mean_squared_error(y_true, y_pred)
    rmse = METRICS["rmse"](y_true, y_pred)
    assert rmse == pytest.approx(np.sqrt(mse))


def test_r2_value(regression_data):
    y_true, y_pred = regression_data
    expected = r2_score(y_true, y_pred)
    result = METRICS["r2"](y_true, y_pred)
    assert result == pytest.approx(expected)


def test_max_error_value(regression_data):
    y_true, y_pred = regression_data
    expected = sklearn_max_error(y_true, y_pred)
    result = METRICS["max_error"](y_true, y_pred)
    assert result == pytest.approx(expected)
