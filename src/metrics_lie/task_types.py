from __future__ import annotations

from enum import Enum


class TaskType(str, Enum):
    """Supported ML task types for model evaluation."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    RANKING = "ranking"

    @property
    def is_classification(self) -> bool:
        """Return True if this task type is a classification variant."""
        return self in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        )

    @property
    def is_binary(self) -> bool:
        """Return True if this task type is binary classification."""
        return self == TaskType.BINARY_CLASSIFICATION

    @property
    def is_regression(self) -> bool:
        """Return True if this task type is regression."""
        return self == TaskType.REGRESSION

    @property
    def supports_threshold(self) -> bool:
        """Return True if this task type supports threshold-based decisions."""
        return self in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION)
