from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.task_types import TaskType

onnxruntime = pytest.importorskip("onnxruntime")


def _create_simple_onnx_model(tmp_path):
    skl2onnx = pytest.importorskip("skl2onnx")
    from sklearn.linear_model import LogisticRegression
    from skl2onnx.common.data_types import FloatTensorType

    X_train = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.float32)
    y_train = np.array([0, 1, 0, 1])
    model = LogisticRegression()
    model.fit(X_train, y_train)

    onnx_model = skl2onnx.convert_sklearn(
        model, "test_model", [("input", FloatTensorType([None, 2]))]
    )
    path = tmp_path / "model.onnx"
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return str(path)


def test_onnx_adapter_loads(tmp_path):
    from metrics_lie.model.adapters.onnx_adapter import ONNXAdapter

    path = _create_simple_onnx_model(tmp_path)
    adapter = ONNXAdapter(path=path, task_type=TaskType.BINARY_CLASSIFICATION)
    assert adapter.task_type == TaskType.BINARY_CLASSIFICATION


def test_onnx_adapter_predict(tmp_path):
    from metrics_lie.model.adapters.onnx_adapter import ONNXAdapter

    path = _create_simple_onnx_model(tmp_path)
    adapter = ONNXAdapter(path=path, task_type=TaskType.BINARY_CLASSIFICATION)
    X = np.array([[0, 0], [1, 1]], dtype=np.float32)
    surface = adapter.predict(X)
    assert surface.n_samples == 2
    assert surface.surface_type.value == "label"


def test_onnx_adapter_predict_proba(tmp_path):
    from metrics_lie.model.adapters.onnx_adapter import ONNXAdapter

    path = _create_simple_onnx_model(tmp_path)
    adapter = ONNXAdapter(path=path, task_type=TaskType.BINARY_CLASSIFICATION)
    X = np.array([[0, 0], [1, 1]], dtype=np.float32)
    surface = adapter.predict_proba(X)
    assert surface is not None
    assert surface.surface_type.value == "probability"
    assert np.all(surface.values >= 0) and np.all(surface.values <= 1)


def test_onnx_adapter_metadata(tmp_path):
    from metrics_lie.model.adapters.onnx_adapter import ONNXAdapter

    path = _create_simple_onnx_model(tmp_path)
    adapter = ONNXAdapter(path=path, task_type=TaskType.BINARY_CLASSIFICATION)
    meta = adapter.metadata
    assert meta.model_format == "onnx"
