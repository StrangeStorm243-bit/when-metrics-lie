"""Tests for multiclass execution pipeline."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

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


def test_multiclass_execution(tmp_path: Path):
    """Run multiclass classification through the full pipeline."""
    rng = np.random.default_rng(42)
    n = 60
    X = rng.standard_normal((n, 2))
    y = np.array([0]*20 + [1]*20 + [2]*20)
    model = LogisticRegression(random_state=0, max_iter=200).fit(X, y)
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    df = pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "y_true": y, "y_score": np.zeros(n)})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    spec = {
        "name": "multiclass_test",
        "task": "multiclass_classification",
        "dataset": {
            "source": "csv",
            "path": str(csv_path),
            "y_true_col": "y_true",
            "y_score_col": "y_score",
            "feature_cols": ["f1", "f2"],
        },
        "metric": "macro_f1",
        "model_source": {"kind": "pickle", "path": str(model_path)},
        "scenarios": [
            {"id": "label_noise", "params": {"p": 0.1}},
            {"id": "score_noise", "params": {"sigma": 0.05}},
        ],
        "n_trials": 5,
        "seed": 42,
    }

    run_id = run_from_spec_dict(spec)
    bundle_json = get_run_dir(run_id).results_json.read_text(encoding="utf-8")
    bundle = ResultBundle.model_validate_json(bundle_json)

    assert bundle.task_type == "multiclass_classification"
    assert bundle.metric_name == "macro_f1"
    assert "macro_f1" in bundle.applicable_metrics
    assert "auc" not in bundle.applicable_metrics
    assert "f1" not in bundle.applicable_metrics
