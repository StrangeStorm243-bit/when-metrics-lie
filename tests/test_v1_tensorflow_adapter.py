"""Tests for TensorFlow/Keras adapter."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from metrics_lie.model.surface import SurfaceType
from metrics_lie.task_types import TaskType


tf = pytest.importorskip("tensorflow")


@pytest.fixture()
def binary_keras_model(tmp_path: Any) -> str:
    """Create a simple Keras binary classification model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation="relu", input_shape=(4,)),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    # Train briefly on random data so weights are initialized
    X = np.random.default_rng(42).standard_normal((20, 4)).astype(np.float32)
    y = np.random.default_rng(42).integers(0, 2, size=20)
    model.fit(X, y, epochs=1, verbose=0)
    path = str(tmp_path / "binary_model.keras")
    model.save(path)
    return path


@pytest.fixture()
def multiclass_keras_model(tmp_path: Any) -> str:
    """Create a simple Keras multiclass (5-class) model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation="relu", input_shape=(4,)),
        tf.keras.layers.Dense(5, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    X = np.random.default_rng(42).standard_normal((20, 4)).astype(np.float32)
    y = np.random.default_rng(42).integers(0, 5, size=20)
    model.fit(X, y, epochs=1, verbose=0)
    path = str(tmp_path / "multi_model.keras")
    model.save(path)
    return path


class TestTensorFlowAdapterBinary:
    """Tests for TensorFlow adapter with binary classification models."""

    def test_predict_returns_label_surface(self, binary_keras_model: str) -> None:
        from metrics_lie.model.adapters.tensorflow_adapter import TensorFlowAdapter

        adapter = TensorFlowAdapter(
            path=binary_keras_model,
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        X = np.random.default_rng(42).standard_normal((10, 4)).astype(np.float32)
        surface = adapter.predict(X)

        assert surface.surface_type == SurfaceType.LABEL
        assert surface.n_samples == 10
        assert set(np.unique(surface.values)).issubset({0, 1})

    def test_predict_proba_returns_probability_surface(
        self, binary_keras_model: str
    ) -> None:
        from metrics_lie.model.adapters.tensorflow_adapter import TensorFlowAdapter

        adapter = TensorFlowAdapter(
            path=binary_keras_model,
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        X = np.random.default_rng(42).standard_normal((10, 4)).astype(np.float32)
        surface = adapter.predict_proba(X)

        assert surface is not None
        assert surface.surface_type == SurfaceType.PROBABILITY
        assert surface.n_samples == 10
        assert np.all(surface.values >= 0.0)
        assert np.all(surface.values <= 1.0)

    def test_metadata(self, binary_keras_model: str) -> None:
        from metrics_lie.model.adapters.tensorflow_adapter import TensorFlowAdapter

        adapter = TensorFlowAdapter(
            path=binary_keras_model,
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        meta = adapter.metadata

        assert meta.model_module == "tensorflow"
        assert meta.model_format == "keras"
        assert meta.model_hash is not None
        assert meta.model_hash.startswith("sha256:")
        assert "predict" in meta.capabilities
        assert "predict_proba" in meta.capabilities

    def test_task_type_property(self, binary_keras_model: str) -> None:
        from metrics_lie.model.adapters.tensorflow_adapter import TensorFlowAdapter

        adapter = TensorFlowAdapter(
            path=binary_keras_model,
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        assert adapter.task_type == TaskType.BINARY_CLASSIFICATION

    def test_get_all_surfaces(self, binary_keras_model: str) -> None:
        from metrics_lie.model.adapters.tensorflow_adapter import TensorFlowAdapter

        adapter = TensorFlowAdapter(
            path=binary_keras_model,
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        X = np.random.default_rng(42).standard_normal((10, 4)).astype(np.float32)
        surfaces = adapter.get_all_surfaces(X)

        assert SurfaceType.LABEL in surfaces
        assert SurfaceType.PROBABILITY in surfaces

    def test_predict_raw(self, binary_keras_model: str) -> None:
        from metrics_lie.model.adapters.tensorflow_adapter import TensorFlowAdapter

        adapter = TensorFlowAdapter(
            path=binary_keras_model,
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        X = np.random.default_rng(42).standard_normal((10, 4)).astype(np.float32)
        raw = adapter.predict_raw(X)

        assert "labels" in raw
        assert "probabilities" in raw
        assert raw["labels"].shape == (10,)


class TestTensorFlowAdapterMulticlass:
    """Tests for TensorFlow adapter with multiclass models."""

    def test_predict_multiclass(self, multiclass_keras_model: str) -> None:
        from metrics_lie.model.adapters.tensorflow_adapter import TensorFlowAdapter

        adapter = TensorFlowAdapter(
            path=multiclass_keras_model,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
        )
        X = np.random.default_rng(42).standard_normal((10, 4)).astype(np.float32)
        surface = adapter.predict(X)

        assert surface.surface_type == SurfaceType.LABEL
        assert surface.n_samples == 10

    def test_predict_proba_multiclass(self, multiclass_keras_model: str) -> None:
        from metrics_lie.model.adapters.tensorflow_adapter import TensorFlowAdapter

        adapter = TensorFlowAdapter(
            path=multiclass_keras_model,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
        )
        X = np.random.default_rng(42).standard_normal((10, 4)).astype(np.float32)
        surface = adapter.predict_proba(X)

        assert surface is not None
        assert surface.surface_type == SurfaceType.PROBABILITY
        assert surface.n_samples == 10
