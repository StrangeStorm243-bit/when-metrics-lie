from __future__ import annotations

import pytest

from metrics_lie.spec import load_experiment_spec


def test_spec_accepts_multiclass_task():
    spec = load_experiment_spec({
        "name": "multiclass test",
        "task": "multiclass_classification",
        "dataset": {"path": "data.csv", "y_true_col": "label", "y_score_col": "pred"},
        "metric": "accuracy",
    })
    assert spec.task == "multiclass_classification"


def test_spec_accepts_regression_task():
    spec = load_experiment_spec({
        "name": "regression test",
        "task": "regression",
        "dataset": {"path": "data.csv", "y_true_col": "target", "y_score_col": "pred"},
        "metric": "accuracy",
    })
    assert spec.task == "regression"


def test_spec_still_defaults_to_binary():
    spec = load_experiment_spec({
        "name": "binary test",
        "dataset": {"path": "data.csv", "y_true_col": "y", "y_score_col": "p"},
        "metric": "auc",
    })
    assert spec.task == "binary_classification"


def test_spec_onnx_model_source():
    spec = load_experiment_spec({
        "name": "onnx test",
        "dataset": {"path": "data.csv", "y_true_col": "y", "y_score_col": "p"},
        "metric": "auc",
        "model_source": {"kind": "onnx", "path": "model.onnx"},
    })
    assert spec.model_source.kind == "onnx"


def test_spec_xgboost_model_source():
    spec = load_experiment_spec({
        "name": "xgb test",
        "dataset": {"path": "data.csv", "y_true_col": "y", "y_score_col": "p"},
        "metric": "auc",
        "model_source": {"kind": "xgboost", "path": "model.ubj"},
    })
    assert spec.model_source.kind == "xgboost"


def test_spec_http_model_source():
    spec = load_experiment_spec({
        "name": "http test",
        "dataset": {"path": "data.csv", "y_true_col": "y", "y_score_col": "p"},
        "metric": "auc",
        "model_source": {"kind": "http", "endpoint": "http://localhost:8080/predict"},
    })
    assert spec.model_source.kind == "http"
    assert spec.model_source.endpoint == "http://localhost:8080/predict"


def test_spec_rejects_invalid_task():
    with pytest.raises(Exception):
        load_experiment_spec({
            "name": "bad",
            "task": "quantum_computing",
            "dataset": {"path": "data.csv", "y_true_col": "y", "y_score_col": "p"},
            "metric": "auc",
        })
