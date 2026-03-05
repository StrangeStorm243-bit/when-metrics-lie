"""Tests for Phase 6 KServe V2 protocol support in HTTP adapter."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from metrics_lie.model.adapters.http_adapter import HTTPAdapter
from metrics_lie.model.surface import SurfaceType
from metrics_lie.task_types import TaskType


def _mock_kserve_response(outputs):
    """Create a mock response with KServe V2 format."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"outputs": outputs}
    return resp


def _mock_custom_response(predictions):
    """Create a mock response with custom format."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"predictions": predictions}
    return resp


# --- Protocol detection ---

def test_auto_detect_kserve_from_url():
    """URL with /v2/ triggers KServe protocol."""
    adapter = HTTPAdapter(
        endpoint="http://localhost:8080/v2/models/mymodel/infer",
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    assert adapter._protocol == "kserve_v2"


def test_custom_protocol_default():
    """Default URL uses custom protocol."""
    adapter = HTTPAdapter(
        endpoint="http://localhost:8080/predict",
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    assert adapter._protocol == "custom"


def test_explicit_kserve_protocol():
    """Explicit protocol='kserve_v2' is respected."""
    adapter = HTTPAdapter(
        endpoint="http://localhost:8080/myapi",
        protocol="kserve_v2",
        model_name="mymodel",
    )
    assert adapter._protocol == "kserve_v2"


# --- KServe URL building ---

def test_kserve_url_with_model_name():
    adapter = HTTPAdapter(
        endpoint="http://localhost:8080",
        protocol="kserve_v2",
        model_name="my_model",
    )
    url = adapter._build_kserve_url()
    assert url == "http://localhost:8080/v2/models/my_model/infer"


def test_kserve_url_without_model_name():
    """When no model_name, use endpoint as-is."""
    adapter = HTTPAdapter(
        endpoint="http://localhost:8080/v2/models/existing/infer",
        protocol="kserve_v2",
    )
    url = adapter._build_kserve_url()
    assert url == "http://localhost:8080/v2/models/existing/infer"


# --- KServe request format ---

def test_kserve_request_format():
    adapter = HTTPAdapter(
        endpoint="http://localhost:8080",
        protocol="kserve_v2",
        model_name="m",
    )
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    req = adapter._format_kserve_request(X)

    assert "inputs" in req
    assert len(req["inputs"]) == 1
    inp = req["inputs"][0]
    assert inp["name"] == "input"
    assert inp["shape"] == [2, 2]
    assert inp["datatype"] == "FP64"
    assert inp["data"] == [1.0, 2.0, 3.0, 4.0]


# --- KServe response parsing ---

def test_parse_kserve_2d_probabilities():
    """2D output [n, classes] is parsed as probabilities."""
    adapter = HTTPAdapter(
        endpoint="http://localhost:8080",
        protocol="kserve_v2",
        model_name="m",
    )
    outputs = [{"name": "output", "shape": [2, 2], "data": [0.8, 0.2, 0.3, 0.7]}]
    preds = adapter._parse_kserve_response({"outputs": outputs})

    assert len(preds) == 2
    assert preds[0]["label"] == 0
    assert preds[0]["probability"] == [0.8, 0.2]
    assert preds[1]["label"] == 1
    assert preds[1]["probability"] == [0.3, 0.7]


def test_parse_kserve_1d_scores():
    """1D output [n] with values in [0,1] is treated as probabilities."""
    adapter = HTTPAdapter(
        endpoint="http://localhost:8080",
        protocol="kserve_v2",
        model_name="m",
        threshold=0.5,
    )
    outputs = [{"name": "output", "shape": [3], "data": [0.9, 0.3, 0.6]}]
    preds = adapter._parse_kserve_response({"outputs": outputs})

    assert len(preds) == 3
    assert preds[0]["label"] == 1  # 0.9 >= 0.5
    assert preds[1]["label"] == 0  # 0.3 < 0.5
    assert preds[2]["label"] == 1  # 0.6 >= 0.5


# --- End-to-end with mock ---

def test_kserve_predict_end_to_end():
    """Full predict() flow with KServe V2 mock."""
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_kserve_response(
            [{"name": "output", "shape": [2, 2], "data": [0.8, 0.2, 0.3, 0.7]}]
        )
        adapter = HTTPAdapter(
            endpoint="http://localhost:8080",
            protocol="kserve_v2",
            model_name="m",
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        surface = adapter.predict(X)

        assert surface.surface_type == SurfaceType.LABEL
        assert surface.n_samples == 2

        # Verify the URL was correct
        call_args = mock_post.call_args
        assert "/v2/models/m/infer" in call_args[0][0]


def test_kserve_predict_proba_end_to_end():
    """Full predict_proba() flow with KServe V2 mock."""
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_kserve_response(
            [{"name": "output", "shape": [2, 2], "data": [0.8, 0.2, 0.3, 0.7]}]
        )
        adapter = HTTPAdapter(
            endpoint="http://localhost:8080",
            protocol="kserve_v2",
            model_name="m",
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        surface = adapter.predict_proba(X)

        assert surface is not None
        assert surface.surface_type == SurfaceType.PROBABILITY
        assert surface.n_samples == 2


# --- Backward compatibility ---

def test_custom_protocol_still_works():
    """Existing custom protocol calls are unchanged."""
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_custom_response([
            {"label": 0, "probability": [0.8, 0.2]},
            {"label": 1, "probability": [0.3, 0.7]},
        ])
        adapter = HTTPAdapter(
            endpoint="http://localhost:8080/predict",
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        surface = adapter.predict(X)

        assert surface.surface_type == SurfaceType.LABEL
        assert surface.n_samples == 2

        # URL should be the endpoint directly (no /v2/ rewrite)
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:8080/predict"
