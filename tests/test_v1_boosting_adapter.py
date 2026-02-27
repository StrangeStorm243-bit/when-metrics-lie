from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.task_types import TaskType

xgb = pytest.importorskip("xgboost")


def _create_xgb_model(tmp_path):
    X_train = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.float32)
    y_train = np.array([0, 1, 0, 1])
    model = xgb.XGBClassifier(n_estimators=5, eval_metric="logloss")
    model.fit(X_train, y_train)
    path = tmp_path / "model.ubj"
    model.save_model(str(path))
    return str(path)


def test_xgb_adapter_loads(tmp_path):
    from metrics_lie.model.adapters.boosting_adapter import BoostingAdapter
    path = _create_xgb_model(tmp_path)
    adapter = BoostingAdapter(
        path=path, framework="xgboost", task_type=TaskType.BINARY_CLASSIFICATION,
    )
    assert adapter.task_type == TaskType.BINARY_CLASSIFICATION


def test_xgb_adapter_predict(tmp_path):
    from metrics_lie.model.adapters.boosting_adapter import BoostingAdapter
    path = _create_xgb_model(tmp_path)
    adapter = BoostingAdapter(
        path=path, framework="xgboost", task_type=TaskType.BINARY_CLASSIFICATION,
    )
    X = np.array([[0, 0], [1, 1]], dtype=np.float32)
    surface = adapter.predict(X)
    assert surface.n_samples == 2
    assert surface.surface_type.value == "label"


def test_xgb_adapter_predict_proba(tmp_path):
    from metrics_lie.model.adapters.boosting_adapter import BoostingAdapter
    path = _create_xgb_model(tmp_path)
    adapter = BoostingAdapter(
        path=path, framework="xgboost", task_type=TaskType.BINARY_CLASSIFICATION,
    )
    X = np.array([[0, 0], [1, 1]], dtype=np.float32)
    surface = adapter.predict_proba(X)
    assert surface is not None
    assert surface.surface_type.value == "probability"


def test_xgb_adapter_metadata(tmp_path):
    from metrics_lie.model.adapters.boosting_adapter import BoostingAdapter
    path = _create_xgb_model(tmp_path)
    adapter = BoostingAdapter(
        path=path, framework="xgboost", task_type=TaskType.BINARY_CLASSIFICATION,
    )
    meta = adapter.metadata
    assert meta.model_format == "xgboost"
