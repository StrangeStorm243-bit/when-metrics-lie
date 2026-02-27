from __future__ import annotations

from metrics_lie.task_types import TaskType


def test_task_type_values():
    assert TaskType.BINARY_CLASSIFICATION == "binary_classification"
    assert TaskType.MULTICLASS_CLASSIFICATION == "multiclass_classification"
    assert TaskType.REGRESSION == "regression"
    assert TaskType.RANKING == "ranking"


def test_task_type_from_string():
    assert TaskType("binary_classification") == TaskType.BINARY_CLASSIFICATION
    assert TaskType("regression") == TaskType.REGRESSION


def test_task_type_is_classification():
    assert TaskType.BINARY_CLASSIFICATION.is_classification is True
    assert TaskType.MULTICLASS_CLASSIFICATION.is_classification is True
    assert TaskType.REGRESSION.is_classification is False
    assert TaskType.RANKING.is_classification is False


def test_task_type_is_binary():
    assert TaskType.BINARY_CLASSIFICATION.is_binary is True
    assert TaskType.MULTICLASS_CLASSIFICATION.is_binary is False
    assert TaskType.REGRESSION.is_binary is False
    assert TaskType.RANKING.is_binary is False


def test_task_type_is_regression():
    assert TaskType.REGRESSION.is_regression is True
    assert TaskType.BINARY_CLASSIFICATION.is_regression is False
    assert TaskType.MULTICLASS_CLASSIFICATION.is_regression is False


def test_task_type_supports_threshold():
    assert TaskType.BINARY_CLASSIFICATION.supports_threshold is True
    assert TaskType.MULTICLASS_CLASSIFICATION.supports_threshold is False
    assert TaskType.REGRESSION.supports_threshold is False
    assert TaskType.RANKING.supports_threshold is False
    assert TaskType.MULTILABEL_CLASSIFICATION.supports_threshold is True


def test_task_type_all_members():
    names = [t.value for t in TaskType]
    assert len(names) >= 4
    assert "binary_classification" in names
