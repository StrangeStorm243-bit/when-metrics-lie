"""Tests for HuggingFace pipeline adapter."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from metrics_lie.model.surface import SurfaceType
from metrics_lie.task_types import TaskType


transformers = pytest.importorskip("transformers")


@pytest.fixture()
def mock_pipeline() -> MagicMock:
    """Create a mock HuggingFace text-classification pipeline."""
    pipe = MagicMock()
    # Simulate text-classification pipeline output with return_all_scores=True:
    # returns list[list[dict]] — one inner list per sample with all class scores.
    def _mock_call(texts: Any, **kwargs: Any) -> list[list[dict[str, Any]]]:
        if isinstance(texts, str):
            texts = [texts]
        return [
            [
                {"label": "NEGATIVE", "score": 0.1 + 0.3 * (i % 2)},
                {"label": "POSITIVE", "score": 0.9 - 0.3 * (i % 2)},
            ]
            for i in range(len(texts))
        ]

    pipe.side_effect = _mock_call
    pipe.model = MagicMock()
    pipe.model.config = MagicMock()
    pipe.model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    pipe.model.config._name_or_path = "mock-model"
    return pipe


class TestHuggingFaceAdapter:
    """Tests for HuggingFace text-classification adapter."""

    def test_predict_returns_label_surface(self, mock_pipeline: MagicMock) -> None:
        from metrics_lie.model.adapters.huggingface_adapter import HuggingFaceAdapter

        adapter = HuggingFaceAdapter(
            pipeline=mock_pipeline,
            task_type=TaskType.BINARY_CLASSIFICATION,
            positive_label_name="POSITIVE",
        )
        texts = ["good movie", "bad movie", "ok movie"]
        surface = adapter.predict(texts)

        assert surface.surface_type == SurfaceType.LABEL
        assert surface.n_samples == 3
        assert set(np.unique(surface.values)).issubset({0, 1})

    def test_predict_proba_returns_probability_surface(
        self, mock_pipeline: MagicMock
    ) -> None:
        from metrics_lie.model.adapters.huggingface_adapter import HuggingFaceAdapter

        adapter = HuggingFaceAdapter(
            pipeline=mock_pipeline,
            task_type=TaskType.BINARY_CLASSIFICATION,
            positive_label_name="POSITIVE",
        )
        texts = ["good movie", "bad movie", "ok movie"]
        surface = adapter.predict_proba(texts)

        assert surface is not None
        assert surface.surface_type == SurfaceType.PROBABILITY
        assert surface.n_samples == 3
        assert np.all(surface.values >= 0.0)
        assert np.all(surface.values <= 1.0)

    def test_metadata(self, mock_pipeline: MagicMock) -> None:
        from metrics_lie.model.adapters.huggingface_adapter import HuggingFaceAdapter

        adapter = HuggingFaceAdapter(
            pipeline=mock_pipeline,
            task_type=TaskType.BINARY_CLASSIFICATION,
            positive_label_name="POSITIVE",
        )
        meta = adapter.metadata

        assert meta.model_module == "transformers"
        assert meta.model_format == "huggingface"
        assert "predict" in meta.capabilities
        assert "predict_proba" in meta.capabilities

    def test_task_type_property(self, mock_pipeline: MagicMock) -> None:
        from metrics_lie.model.adapters.huggingface_adapter import HuggingFaceAdapter

        adapter = HuggingFaceAdapter(
            pipeline=mock_pipeline,
            task_type=TaskType.BINARY_CLASSIFICATION,
            positive_label_name="POSITIVE",
        )
        assert adapter.task_type == TaskType.BINARY_CLASSIFICATION

    def test_get_all_surfaces(self, mock_pipeline: MagicMock) -> None:
        from metrics_lie.model.adapters.huggingface_adapter import HuggingFaceAdapter

        adapter = HuggingFaceAdapter(
            pipeline=mock_pipeline,
            task_type=TaskType.BINARY_CLASSIFICATION,
            positive_label_name="POSITIVE",
        )
        texts = ["good movie", "bad movie", "ok movie"]
        surfaces = adapter.get_all_surfaces(texts)

        assert SurfaceType.LABEL in surfaces
        assert SurfaceType.PROBABILITY in surfaces

    def test_predict_raw(self, mock_pipeline: MagicMock) -> None:
        from metrics_lie.model.adapters.huggingface_adapter import HuggingFaceAdapter

        adapter = HuggingFaceAdapter(
            pipeline=mock_pipeline,
            task_type=TaskType.BINARY_CLASSIFICATION,
            positive_label_name="POSITIVE",
        )
        texts = ["good movie", "bad movie", "ok movie"]
        raw = adapter.predict_raw(texts)

        assert "labels" in raw
        assert "scores" in raw
        assert len(raw["labels"]) == 3

    def test_from_model_path(self) -> None:
        """Test the class method that loads a pipeline from model path."""
        from metrics_lie.model.adapters.huggingface_adapter import HuggingFaceAdapter

        mock_pipe = MagicMock()
        mock_pipe.model = MagicMock()
        mock_pipe.model.config = MagicMock()
        mock_pipe.model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        mock_pipe.model.config._name_or_path = "test-model"

        with patch("metrics_lie.model.adapters.huggingface_adapter.transformers") as mock_tf:
            mock_tf.pipeline.return_value = mock_pipe
            adapter = HuggingFaceAdapter.from_model_path(
                model_path="test-model",
                task_type=TaskType.BINARY_CLASSIFICATION,
                positive_label_name="POSITIVE",
            )
            mock_tf.pipeline.assert_called_once_with(
                "text-classification",
                model="test-model",
                return_all_scores=True,
            )
            assert adapter.task_type == TaskType.BINARY_CLASSIFICATION
