from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from metrics_lie.model.adapters.http_adapter import HTTPAdapter
from metrics_lie.model.surface import SurfaceType
from metrics_lie.task_types import TaskType


def _mock_response(predictions):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"predictions": predictions}
    return resp


def test_http_adapter_predict():
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response([
            {"label": 0, "probability": [0.8, 0.2]},
            {"label": 1, "probability": [0.3, 0.7]},
        ])
        adapter = HTTPAdapter(
            endpoint="http://localhost:8080/predict",
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        surface = adapter.predict(X)
        assert surface.n_samples == 2
        assert surface.surface_type == SurfaceType.LABEL


def test_http_adapter_predict_proba():
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response([
            {"label": 0, "probability": [0.8, 0.2]},
            {"label": 1, "probability": [0.3, 0.7]},
        ])
        adapter = HTTPAdapter(
            endpoint="http://localhost:8080/predict",
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        surface = adapter.predict_proba(X)
        assert surface is not None
        assert surface.surface_type == SurfaceType.PROBABILITY


def test_http_adapter_metadata():
    adapter = HTTPAdapter(
        endpoint="http://localhost:8080/predict",
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    meta = adapter.metadata
    assert meta.model_format == "http"
    assert meta.model_class == "HTTPModel"
