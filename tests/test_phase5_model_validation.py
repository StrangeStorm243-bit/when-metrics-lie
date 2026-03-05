"""Tests for Phase 5 multi-task model validation."""
from __future__ import annotations

import pickle

import numpy as np
import pytest

pytest.importorskip("fastapi")

from web.backend.app.model_validation import (
    ACCEPTED_EXTENSIONS,
    validate_model,
    validate_sklearn_pickle,
)


class _FakeClassifier:
    """Fake binary classifier with predict_proba."""
    n_features_in_ = 3

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)) * 0.6, np.ones(len(X)) * 0.4])


class _FakeMulticlassClassifier:
    """Fake multiclass classifier with 4 classes."""
    n_features_in_ = 3

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([
            np.ones(n) * 0.7,
            np.ones(n) * 0.1,
            np.ones(n) * 0.1,
            np.ones(n) * 0.1,
        ])


class _FakeRegressor:
    """Fake regressor with predict only."""
    n_features_in_ = 3

    def predict(self, X):
        return np.zeros(len(X))


class _FakeBadModel:
    """Model with no predict."""
    pass


def _pickle_model(model) -> bytes:
    return pickle.dumps(model)


# --- Extension acceptance ---

def test_accepted_extensions_include_all_formats():
    assert ".pkl" in ACCEPTED_EXTENSIONS
    assert ".onnx" in ACCEPTED_EXTENSIONS
    assert ".ubj" in ACCEPTED_EXTENSIONS
    assert ".lgb" in ACCEPTED_EXTENSIONS
    assert ".cbm" in ACCEPTED_EXTENSIONS


def test_unsupported_extension_rejected():
    result = validate_model(b"data", ".txt", "binary_classification")
    assert not result.valid
    assert "Unsupported" in result.error


# --- Pickle binary ---

def test_binary_classifier_valid():
    raw = _pickle_model(_FakeClassifier())
    result = validate_model(raw, ".pkl", "binary_classification")
    assert result.valid
    assert result.task_type == "binary_classification"
    assert result.n_classes == 2


def test_multiclass_model_fails_as_binary():
    raw = _pickle_model(_FakeMulticlassClassifier())
    result = validate_model(raw, ".pkl", "binary_classification")
    assert not result.valid
    assert "2 classes" in result.error


# --- Pickle multiclass ---

def test_multiclass_classifier_valid():
    raw = _pickle_model(_FakeMulticlassClassifier())
    result = validate_model(raw, ".pkl", "multiclass_classification")
    assert result.valid
    assert result.task_type == "multiclass_classification"
    assert result.n_classes == 4


def test_binary_model_fails_as_multiclass():
    raw = _pickle_model(_FakeClassifier())
    result = validate_model(raw, ".pkl", "multiclass_classification")
    assert not result.valid
    assert "3+ classes" in result.error


# --- Pickle regression ---

def test_regression_model_valid():
    raw = _pickle_model(_FakeRegressor())
    result = validate_model(raw, ".pkl", "regression")
    assert result.valid
    assert result.task_type == "regression"
    assert result.n_classes is None


def test_regression_doesnt_require_proba():
    """Regression models don't need predict_proba."""
    raw = _pickle_model(_FakeRegressor())
    result = validate_model(raw, ".pkl", "regression")
    assert result.valid
    # Regressor has predict but not predict_proba
    assert result.capabilities["predict"] is True


# --- Validation errors ---

def test_no_predict_rejected():
    raw = _pickle_model(_FakeBadModel())
    result = validate_model(raw, ".pkl", "binary_classification")
    assert not result.valid
    assert "predict()" in result.error


def test_invalid_pickle_bytes():
    result = validate_model(b"not a pickle", ".pkl", "binary_classification")
    assert not result.valid
    assert "Invalid pickle" in result.error


# --- Boosting format passthrough ---

def test_boosting_format_accepted():
    """Boosting formats are accepted if non-empty (validated at inference time)."""
    result = validate_model(b"model data", ".ubj", "binary_classification")
    assert result.valid
    assert result.model_class == "xgboost_model"


def test_lightgbm_format_accepted():
    result = validate_model(b"model data", ".lgb", "regression")
    assert result.valid
    assert result.task_type == "regression"


# --- Backward compatibility ---

def test_validate_sklearn_pickle_alias():
    """Legacy validate_sklearn_pickle still works."""
    raw = _pickle_model(_FakeClassifier())
    result = validate_sklearn_pickle(raw)
    assert result.valid
    assert result.task_type == "binary_classification"
