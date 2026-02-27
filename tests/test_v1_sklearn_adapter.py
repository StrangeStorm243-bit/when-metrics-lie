from __future__ import annotations

import numpy as np

from metrics_lie.model.adapter import SklearnAdapter, ModelAdapter
from metrics_lie.model.sources import ModelSourceCallable
from metrics_lie.model.metadata import ModelMetadata
from metrics_lie.task_types import TaskType


def _make_callable_source(name: str = "test") -> ModelSourceCallable:
    def fn(X: np.ndarray) -> np.ndarray:
        return np.array([0.5] * len(X))

    return ModelSourceCallable(fn=fn, name=name)


def test_sklearn_adapter_is_model_adapter():
    assert SklearnAdapter is ModelAdapter


def test_sklearn_adapter_task_type():
    source = _make_callable_source()
    adapter = SklearnAdapter(source)
    assert adapter.task_type == TaskType.BINARY_CLASSIFICATION


def test_sklearn_adapter_metadata():
    source = _make_callable_source()
    adapter = SklearnAdapter(source)
    meta = adapter.metadata
    assert isinstance(meta, ModelMetadata)
    assert meta.model_format == "callable"


def test_sklearn_adapter_predict_raw():
    source = _make_callable_source()
    adapter = SklearnAdapter(source)
    X = np.array([[1.0, 2.0]])
    raw = adapter.predict_raw(X)
    assert "probabilities" in raw
