"""Tests for PyTorch TorchScript adapter."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from metrics_lie.model.surface import SurfaceType
from metrics_lie.task_types import TaskType


torch = pytest.importorskip("torch")


@pytest.fixture()
def binary_torchscript_model(tmp_path: Any) -> str:
    """Create a simple TorchScript binary classification model."""

    class BinaryModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = self.linear(x)
            return torch.nn.functional.softmax(logits, dim=1)

    model = BinaryModel()
    model.eval()
    scripted = torch.jit.script(model)
    path = str(tmp_path / "binary_model.pt")
    scripted.save(path)
    return path


@pytest.fixture()
def multiclass_torchscript_model(tmp_path: Any) -> str:
    """Create a simple TorchScript multiclass (5-class) model."""

    class MultiModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = self.linear(x)
            return torch.nn.functional.softmax(logits, dim=1)

    model = MultiModel()
    model.eval()
    scripted = torch.jit.script(model)
    path = str(tmp_path / "multi_model.pt")
    scripted.save(path)
    return path


class TestPyTorchAdapterBinary:
    """Tests for PyTorch adapter with binary classification models."""

    def test_predict_returns_label_surface(self, binary_torchscript_model: str) -> None:
        from metrics_lie.model.adapters.pytorch_adapter import PyTorchAdapter

        adapter = PyTorchAdapter(
            path=binary_torchscript_model,
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        X = np.random.default_rng(42).standard_normal((10, 4))
        surface = adapter.predict(X)

        assert surface.surface_type == SurfaceType.LABEL
        assert surface.n_samples == 10
        assert set(np.unique(surface.values)).issubset({0, 1})

    def test_predict_proba_returns_probability_surface(
        self, binary_torchscript_model: str
    ) -> None:
        from metrics_lie.model.adapters.pytorch_adapter import PyTorchAdapter

        adapter = PyTorchAdapter(
            path=binary_torchscript_model,
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        X = np.random.default_rng(42).standard_normal((10, 4))
        surface = adapter.predict_proba(X)

        assert surface is not None
        assert surface.surface_type == SurfaceType.PROBABILITY
        assert surface.n_samples == 10
        assert np.all(surface.values >= 0.0)
        assert np.all(surface.values <= 1.0)

    def test_metadata(self, binary_torchscript_model: str) -> None:
        from metrics_lie.model.adapters.pytorch_adapter import PyTorchAdapter

        adapter = PyTorchAdapter(
            path=binary_torchscript_model,
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        meta = adapter.metadata

        assert meta.model_class == "RecursiveScriptModule"
        assert meta.model_module == "torch"
        assert meta.model_format == "torchscript"
        assert meta.model_hash is not None
        assert meta.model_hash.startswith("sha256:")
        assert "predict" in meta.capabilities
        assert "predict_proba" in meta.capabilities

    def test_task_type_property(self, binary_torchscript_model: str) -> None:
        from metrics_lie.model.adapters.pytorch_adapter import PyTorchAdapter

        adapter = PyTorchAdapter(
            path=binary_torchscript_model,
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        assert adapter.task_type == TaskType.BINARY_CLASSIFICATION

    def test_get_all_surfaces(self, binary_torchscript_model: str) -> None:
        from metrics_lie.model.adapters.pytorch_adapter import PyTorchAdapter

        adapter = PyTorchAdapter(
            path=binary_torchscript_model,
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        X = np.random.default_rng(42).standard_normal((10, 4))
        surfaces = adapter.get_all_surfaces(X)

        assert SurfaceType.LABEL in surfaces
        assert SurfaceType.PROBABILITY in surfaces

    def test_predict_raw(self, binary_torchscript_model: str) -> None:
        from metrics_lie.model.adapters.pytorch_adapter import PyTorchAdapter

        adapter = PyTorchAdapter(
            path=binary_torchscript_model,
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        X = np.random.default_rng(42).standard_normal((10, 4))
        raw = adapter.predict_raw(X)

        assert "labels" in raw
        assert "probabilities" in raw
        assert raw["labels"].shape == (10,)
        assert raw["probabilities"].shape[0] == 10


class TestPyTorchAdapterMulticlass:
    """Tests for PyTorch adapter with multiclass models."""

    def test_predict_multiclass(self, multiclass_torchscript_model: str) -> None:
        from metrics_lie.model.adapters.pytorch_adapter import PyTorchAdapter

        adapter = PyTorchAdapter(
            path=multiclass_torchscript_model,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
        )
        X = np.random.default_rng(42).standard_normal((10, 4))
        surface = adapter.predict(X)

        assert surface.surface_type == SurfaceType.LABEL
        assert surface.n_samples == 10

    def test_predict_proba_multiclass(self, multiclass_torchscript_model: str) -> None:
        from metrics_lie.model.adapters.pytorch_adapter import PyTorchAdapter

        adapter = PyTorchAdapter(
            path=multiclass_torchscript_model,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
        )
        X = np.random.default_rng(42).standard_normal((10, 4))
        surface = adapter.predict_proba(X)

        assert surface is not None
        assert surface.surface_type == SurfaceType.PROBABILITY
        # For multiclass, probabilities are the full matrix flattened isn't right;
        # we return positive class col for binary, full matrix for multiclass
        assert surface.n_samples == 10
