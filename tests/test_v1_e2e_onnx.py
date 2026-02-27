from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

onnxruntime = pytest.importorskip("onnxruntime")
skl2onnx = pytest.importorskip("skl2onnx")


def _create_onnx_model_and_data(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from skl2onnx.common.data_types import FloatTensorType

    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 3)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = LogisticRegression()
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]

    onnx_model = skl2onnx.convert_sklearn(
        model, "test", [("input", FloatTensorType([None, 3]))]
    )
    model_path = tmp_path / "model.onnx"
    with open(model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    df["y_true"] = y
    df["y_score"] = proba
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    return str(model_path), str(csv_path)


def test_e2e_onnx_binary_classification(tmp_path):
    from metrics_lie.execution import run_from_spec_dict

    model_path, csv_path = _create_onnx_model_and_data(tmp_path)
    spec = {
        "name": "ONNX E2E test",
        "task": "binary_classification",
        "dataset": {
            "path": csv_path,
            "y_true_col": "y_true",
            "y_score_col": "y_score",
            "feature_cols": ["f1", "f2", "f3"],
        },
        "model_source": {
            "kind": "onnx",
            "path": model_path,
        },
        "metric": "auc",
        "scenarios": [
            {"id": "label_noise", "params": {"p": 0.05}},
            {"id": "score_noise", "params": {"sigma": 0.02}},
        ],
        "n_trials": 10,
        "seed": 42,
    }
    run_id = run_from_spec_dict(spec)
    assert run_id is not None
    assert len(run_id) == 10

    from metrics_lie.utils.paths import get_run_dir
    paths = get_run_dir(run_id)
    assert paths.results_json.exists()

    result = json.loads(paths.results_json.read_text())
    assert result["metric_name"] == "auc"
    assert result["run_id"] == run_id
    assert len(result["scenarios"]) == 2
    assert result["baseline"]["mean"] > 0
