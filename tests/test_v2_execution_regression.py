"""Tests for regression execution pipeline."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from metrics_lie.db.session import DB_PATH, engine, init_db
from metrics_lie.execution import run_from_spec_dict
from metrics_lie.schema import ResultBundle
from metrics_lie.utils.paths import get_run_dir


@pytest.fixture(autouse=True)
def fresh_db():
    engine.dispose()
    if DB_PATH.exists():
        DB_PATH.unlink()
    init_db()


def test_regression_execution(tmp_path: Path):
    """Run regression through the full pipeline."""
    rng = np.random.default_rng(42)
    n = 50
    X = rng.standard_normal((n, 2))
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + rng.normal(0, 0.5, n)
    model = LinearRegression().fit(X, y)
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    df = pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "y_true": y, "y_score": np.zeros(n)})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    spec = {
        "name": "regression_test",
        "task": "regression",
        "dataset": {
            "source": "csv",
            "path": str(csv_path),
            "y_true_col": "y_true",
            "y_score_col": "y_score",
            "feature_cols": ["f1", "f2"],
        },
        "metric": "mae",
        "model_source": {"kind": "pickle", "path": str(model_path)},
        "scenarios": [
            {"id": "label_noise", "params": {"p": 0.1}},
            {"id": "score_noise", "params": {"sigma": 0.5}},
        ],
        "n_trials": 5,
        "seed": 42,
    }

    run_id = run_from_spec_dict(spec)
    bundle_json = get_run_dir(run_id).results_json.read_text(encoding="utf-8")
    bundle = ResultBundle.model_validate_json(bundle_json)

    assert bundle.task_type == "regression"
    assert bundle.metric_name == "mae"
    assert "mae" in bundle.applicable_metrics
    assert "mse" in bundle.applicable_metrics
    assert "rmse" in bundle.applicable_metrics
    assert "auc" not in bundle.applicable_metrics
    assert "f1" not in bundle.applicable_metrics
    assert "macro_f1" not in bundle.applicable_metrics
    assert "threshold_sweep" not in bundle.analysis_artifacts
