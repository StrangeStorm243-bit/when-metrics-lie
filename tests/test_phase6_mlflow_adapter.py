"""Tests for Phase 6 MLflow model adapter."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from metrics_lie.model.surface import SurfaceType
from metrics_lie.task_types import TaskType


def _make_mlflow_mocks():
    """Create properly linked mlflow and mlflow.pyfunc mocks.

    The key is that ``import mlflow.pyfunc`` inside the adapter resolves
    ``mlflow`` from sys.modules and then accesses ``.pyfunc`` on it.  So
    sys.modules["mlflow"].pyfunc must be the **same** object as
    sys.modules["mlflow.pyfunc"].
    """
    mock_pyfunc = MagicMock()
    mock_mlflow = MagicMock()
    mock_mlflow.pyfunc = mock_pyfunc
    return mock_mlflow, mock_pyfunc


def test_mlflow_adapter_import_error():
    """Raises ImportError with install hint when mlflow is missing."""
    try:
        import mlflow  # noqa: F401
        pytest.skip("mlflow is installed -- cannot test missing import")
    except ImportError:
        pass

    with pytest.raises(ImportError, match="pip install"):
        from metrics_lie.model.adapters.mlflow_adapter import MLflowAdapter
        MLflowAdapter(uri="runs:/fake/model")


def test_mlflow_adapter_predict_binary():
    """MLflow adapter returns LABEL surface for binary classification."""
    mock_mlflow, mock_pyfunc = _make_mlflow_mocks()

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1, 0])
    mock_model.metadata = MagicMock()
    mock_model.metadata.flavors = {"sklearn": {}}
    mock_pyfunc.load_model.return_value = mock_model

    with patch.dict("sys.modules", {"mlflow": mock_mlflow, "mlflow.pyfunc": mock_pyfunc}):
        from metrics_lie.model.adapters.mlflow_adapter import MLflowAdapter

        adapter = MLflowAdapter(
            uri="runs:/abc123/model",
            task_type=TaskType.BINARY_CLASSIFICATION,
        )

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        surface = adapter.predict(X)

    assert surface.surface_type == SurfaceType.LABEL
    assert surface.n_samples == 3
    assert np.array_equal(surface.values, np.array([0, 1, 0]))


def test_mlflow_adapter_predict_proba_binary():
    """MLflow adapter returns PROBABILITY surface when output is 2D."""
    mock_mlflow, mock_pyfunc = _make_mlflow_mocks()

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    mock_model.metadata = MagicMock()
    mock_model.metadata.flavors = {}
    mock_pyfunc.load_model.return_value = mock_model

    with patch.dict("sys.modules", {"mlflow": mock_mlflow, "mlflow.pyfunc": mock_pyfunc}):
        from metrics_lie.model.adapters.mlflow_adapter import MLflowAdapter

        adapter = MLflowAdapter(
            uri="runs:/abc123/model",
            task_type=TaskType.BINARY_CLASSIFICATION,
        )

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        surface = adapter.predict_proba(X)

    assert surface is not None
    assert surface.surface_type == SurfaceType.PROBABILITY
    assert surface.n_samples == 3
    assert surface.values[1] == pytest.approx(0.7)  # positive class prob


def test_mlflow_adapter_predict_regression():
    """MLflow adapter returns CONTINUOUS surface for regression."""
    mock_mlflow, mock_pyfunc = _make_mlflow_mocks()

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1.5, 2.3, 0.8])
    mock_model.metadata = MagicMock()
    mock_model.metadata.flavors = {}
    mock_pyfunc.load_model.return_value = mock_model

    with patch.dict("sys.modules", {"mlflow": mock_mlflow, "mlflow.pyfunc": mock_pyfunc}):
        from metrics_lie.model.adapters.mlflow_adapter import MLflowAdapter

        adapter = MLflowAdapter(
            uri="runs:/abc123/model",
            task_type=TaskType.REGRESSION,
        )

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        surface = adapter.predict(X)

    assert surface.surface_type == SurfaceType.CONTINUOUS
    assert surface.n_samples == 3


def test_mlflow_adapter_metadata():
    """MLflow adapter metadata detects sklearn flavor."""
    mock_mlflow, mock_pyfunc = _make_mlflow_mocks()

    mock_model = MagicMock()
    mock_model.metadata = MagicMock()
    mock_model.metadata.flavors = {"sklearn": {"version": "1.3"}}
    mock_pyfunc.load_model.return_value = mock_model

    with patch.dict("sys.modules", {"mlflow": mock_mlflow, "mlflow.pyfunc": mock_pyfunc}):
        from metrics_lie.model.adapters.mlflow_adapter import MLflowAdapter

        adapter = MLflowAdapter(uri="runs:/abc/model")

    meta = adapter.metadata

    assert meta.model_format == "mlflow"
    assert meta.model_class == "mlflow.sklearn"


def test_mlflow_adapter_get_all_surfaces():
    """get_all_surfaces returns both label and probability surfaces."""
    mock_mlflow, mock_pyfunc = _make_mlflow_mocks()

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
    mock_model.metadata = MagicMock()
    mock_model.metadata.flavors = {}
    mock_pyfunc.load_model.return_value = mock_model

    with patch.dict("sys.modules", {"mlflow": mock_mlflow, "mlflow.pyfunc": mock_pyfunc}):
        from metrics_lie.model.adapters.mlflow_adapter import MLflowAdapter

        adapter = MLflowAdapter(uri="runs:/abc/model")
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        surfaces = adapter.get_all_surfaces(X)

    assert len(surfaces) >= 1
