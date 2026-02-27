"""Tests for metrics/scenarios catalog helpers."""
from __future__ import annotations

from metrics_lie.catalog import list_metrics, list_model_formats, list_scenarios


def test_list_metrics_binary():
    metrics = list_metrics(task="binary_classification")
    assert "auc" in metrics
    assert "f1" in metrics
    assert "macro_f1" not in metrics


def test_list_metrics_multiclass():
    metrics = list_metrics(task="multiclass_classification")
    assert "macro_f1" in metrics
    assert "auc" not in metrics


def test_list_metrics_regression():
    metrics = list_metrics(task="regression")
    assert "mae" in metrics
    assert "auc" not in metrics


def test_list_metrics_all():
    metrics = list_metrics()
    assert "auc" in metrics
    assert "mae" in metrics
    assert "macro_f1" in metrics


def test_list_scenarios_all():
    scenarios = list_scenarios()
    assert "label_noise" in scenarios
    assert "score_noise" in scenarios


def test_list_scenarios_by_task():
    scenarios = list_scenarios(task="regression")
    assert "label_noise" in scenarios
    assert "threshold_gaming" not in scenarios


def test_list_model_formats():
    formats = list_model_formats()
    assert "pickle" in formats
    assert "onnx" in formats
