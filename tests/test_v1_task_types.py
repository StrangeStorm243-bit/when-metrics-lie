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


def test_task_type_all_members():
    names = [t.value for t in TaskType]
    assert len(names) >= 4
    assert "binary_classification" in names
