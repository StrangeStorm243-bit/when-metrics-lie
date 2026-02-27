"""End-to-end tests for Phase 2 multi-task pipeline."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression

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


def test_e2e_multiclass_with_scenarios(tmp_path: Path):
    """Full multiclass pipeline with label_noise and score_noise scenarios."""
    rng = np.random.default_rng(42)
    n = 90
    X = rng.standard_normal((n, 3))
    y = np.array([0]*30 + [1]*30 + [2]*30)
    model = LogisticRegression(random_state=0, max_iter=300).fit(X, y)
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    df = pd.DataFrame({
        "f1": X[:, 0], "f2": X[:, 1], "f3": X[:, 2],
        "y_true": y, "y_score": np.zeros(n),
        "group": (["A"]*15 + ["B"]*15) * 3,
    })
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    spec = {
        "name": "e2e_multiclass",
        "task": "multiclass_classification",
        "dataset": {
            "source": "csv", "path": str(csv_path),
            "y_true_col": "y_true", "y_score_col": "y_score",
            "feature_cols": ["f1", "f2", "f3"], "subgroup_col": "group",
        },
        "metric": "macro_f1",
        "model_source": {"kind": "pickle", "path": str(model_path)},
        "scenarios": [
            {"id": "label_noise", "params": {"p": 0.1}},
            {"id": "score_noise", "params": {"sigma": 0.05}},
            {"id": "class_imbalance", "params": {"target_pos_rate": 0.15}},
        ],
        "n_trials": 5,
        "seed": 42,
    }

    run_id = run_from_spec_dict(spec)
    bundle = ResultBundle.model_validate_json(
        get_run_dir(run_id).results_json.read_text(encoding="utf-8")
    )
    assert bundle.task_type == "multiclass_classification"
    assert len(bundle.scenarios) > 0
    assert bundle.baseline is not None
    assert bundle.baseline.mean > 0.0


def test_e2e_regression_with_scenarios(tmp_path: Path):
    """Full regression pipeline with label_noise and score_noise scenarios."""
    rng = np.random.default_rng(42)
    n = 50
    X = rng.standard_normal((n, 2))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(0, 0.5, n)
    model = LinearRegression().fit(X, y)
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    df = pd.DataFrame({
        "f1": X[:, 0], "f2": X[:, 1],
        "y_true": y, "y_score": np.zeros(n),
    })
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    spec = {
        "name": "e2e_regression",
        "task": "regression",
        "dataset": {
            "source": "csv", "path": str(csv_path),
            "y_true_col": "y_true", "y_score_col": "y_score",
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
    bundle = ResultBundle.model_validate_json(
        get_run_dir(run_id).results_json.read_text(encoding="utf-8")
    )
    assert bundle.task_type == "regression"
    assert "mae" in bundle.applicable_metrics
    assert "r2" in bundle.applicable_metrics
    assert len(bundle.scenarios) > 0
    # No binary analysis artifacts
    assert "threshold_sweep" not in bundle.analysis_artifacts
    assert "metric_disagreements" not in bundle.analysis_artifacts
