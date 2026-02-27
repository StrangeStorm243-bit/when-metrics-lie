"""Tests for public SDK entry points."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from metrics_lie.db.session import DB_PATH, engine, init_db
from metrics_lie.schema import ResultBundle


@pytest.fixture(autouse=True)
def fresh_db():
    engine.dispose()
    if DB_PATH.exists():
        DB_PATH.unlink()
    init_db()


@pytest.fixture
def binary_fixtures(tmp_path: Path):
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegression(random_state=0).fit(X, y)
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    df = pd.DataFrame(
        {"f1": X[:, 0], "f2": X[:, 1], "y_true": y, "y_score": np.zeros(n)}
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path), str(model_path)


def test_evaluate_returns_result_bundle(binary_fixtures):
    from metrics_lie.sdk import evaluate

    csv_path, model_path = binary_fixtures
    result = evaluate(
        name="sdk_test",
        dataset=csv_path,
        model=model_path,
        metric="auc",
        scenarios=[{"type": "label_noise", "noise_rate": 0.1}],
        n_trials=3,
        seed=42,
    )
    assert isinstance(result, ResultBundle)
    assert result.metric_name == "auc"
    assert result.run_id is not None


def test_evaluate_file(binary_fixtures, tmp_path: Path):
    from metrics_lie.sdk import evaluate_file

    csv_path, model_path = binary_fixtures
    spec = {
        "name": "file_test",
        "dataset": {
            "source": "csv",
            "path": csv_path,
            "y_true_col": "y_true",
            "y_score_col": "y_score",
            "feature_cols": ["f1", "f2"],
        },
        "metric": "auc",
        "model_source": {"kind": "pickle", "path": model_path},
        "scenarios": [],
        "n_trials": 1,
        "seed": 42,
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    result = evaluate_file(str(spec_path))
    assert isinstance(result, ResultBundle)


def test_compare_bundles_sdk(binary_fixtures):
    from metrics_lie.sdk import compare, evaluate

    csv_path, model_path = binary_fixtures
    result_a = evaluate(
        name="compare_a",
        dataset=csv_path,
        model=model_path,
        metric="auc",
        n_trials=3,
        seed=42,
    )
    result_b = evaluate(
        name="compare_b",
        dataset=csv_path,
        model=model_path,
        metric="auc",
        n_trials=3,
        seed=99,
    )
    report = compare(result_a, result_b)
    assert "baseline_delta" in report
    assert "decision" in report


def test_score_sdk(binary_fixtures):
    from metrics_lie.sdk import evaluate, score

    csv_path, model_path = binary_fixtures
    scenarios = [{"type": "label_noise", "noise_rate": 0.1}]
    result_a = evaluate(
        name="score_a",
        dataset=csv_path,
        model=model_path,
        metric="auc",
        scenarios=scenarios,
        n_trials=3,
        seed=42,
    )
    result_b = evaluate(
        name="score_b",
        dataset=csv_path,
        model=model_path,
        metric="auc",
        scenarios=scenarios,
        n_trials=3,
        seed=99,
    )
    scorecard = score(result_a, result_b, profile="balanced")
    assert hasattr(scorecard, "total_score")
    assert hasattr(scorecard, "profile_name")
