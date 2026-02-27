from __future__ import annotations

from typing import Any

import numpy as np

from metrics_lie.model.protocol import ModelAdapterProtocol
from metrics_lie.model.metadata import ModelMetadata
from metrics_lie.model.surface import PredictionSurface, SurfaceType, CalibrationState
from metrics_lie.task_types import TaskType


class FakeAdapter:
    """A minimal adapter that satisfies the protocol."""

    @property
    def task_type(self) -> TaskType:
        return TaskType.BINARY_CLASSIFICATION

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_class="Fake",
            model_module="test",
            model_format="memory",
        )

    def predict(self, X: np.ndarray) -> PredictionSurface:
        return PredictionSurface(
            surface_type=SurfaceType.LABEL,
            values=np.zeros(X.shape[0], dtype=int),
            dtype=np.dtype("int64"),
            n_samples=X.shape[0],
            class_names=("negative", "positive"),
            positive_label=1,
            threshold=None,
            calibration_state=CalibrationState.UNKNOWN,
            model_hash=None,
            is_deterministic=True,
        )

    def predict_proba(self, X: np.ndarray) -> PredictionSurface | None:
        return None

    def predict_raw(self, X: np.ndarray) -> dict[str, Any]:
        return {"labels": np.zeros(X.shape[0])}

    def get_all_surfaces(self, X: np.ndarray) -> dict[SurfaceType, PredictionSurface]:
        return {SurfaceType.LABEL: self.predict(X)}


def test_fake_adapter_satisfies_protocol():
    adapter: ModelAdapterProtocol = FakeAdapter()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    surface = adapter.predict(X)
    assert surface.surface_type == SurfaceType.LABEL
    assert surface.n_samples == 2
    assert adapter.task_type == TaskType.BINARY_CLASSIFICATION


def test_protocol_metadata():
    adapter: ModelAdapterProtocol = FakeAdapter()
    meta = adapter.metadata
    assert meta.model_class == "Fake"
    assert meta.model_format == "memory"


def test_protocol_predict_raw():
    adapter: ModelAdapterProtocol = FakeAdapter()
    X = np.array([[1.0]])
    raw = adapter.predict_raw(X)
    assert "labels" in raw
