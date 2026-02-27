"""End-to-end integration tests for Phase 3 diagnostics generalization."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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


@pytest.fixture
def multiclass_fixtures(tmp_path: Path):
    rng = np.random.default_rng(42)
    n = 60
    X = rng.standard_normal((n, 2))
    y = np.array([0] * 20 + [1] * 20 + [2] * 20)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    model_path = tmp_path / "mc_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    df = pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "y_true": y, "y_score": np.zeros(n)})
    csv_path = tmp_path / "mc_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path), str(model_path)


@pytest.fixture
def regression_fixtures(tmp_path: Path):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 3))
    y = X[:, 0] * 2 + X[:, 1] - 0.5 * X[:, 2] + rng.normal(0, 0.1, 80)
    model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
    model_path = tmp_path / "reg_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    df = pd.DataFrame({"f0": X[:, 0], "f1": X[:, 1], "f2": X[:, 2]})
    df["y_true"] = y
    df["y_score"] = model.predict(X)
    csv_path = tmp_path / "reg_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path), str(model_path)


class TestMulticlassFullAnalysis:
    def test_multiclass_has_failure_modes(self, multiclass_fixtures):
        csv_path, model_path = multiclass_fixtures
        spec = {
            "name": "mc_analysis",
            "task": "multiclass_classification",
            "dataset": {
                "source": "csv", "path": csv_path,
                "y_true_col": "y_true", "y_score_col": "y_score",
                "feature_cols": ["f1", "f2"],
            },
            "metric": "macro_f1",
            "model_source": {"kind": "pickle", "path": model_path},
            "scenarios": [{"id": "label_noise", "params": {"p": 0.1}}],
            "n_trials": 3,
            "seed": 42,
        }
        run_id = run_from_spec_dict(spec)
        bundle = ResultBundle.model_validate_json(
            get_run_dir(run_id).results_json.read_text(encoding="utf-8")
        )
        aa = bundle.analysis_artifacts
        assert "failure_modes" in aa, f"Missing failure_modes, got: {list(aa.keys())}"
        assert aa["failure_modes"]["total_samples"] > 0

    def test_multiclass_has_dashboard(self, multiclass_fixtures):
        csv_path, model_path = multiclass_fixtures
        spec = {
            "name": "mc_dashboard",
            "task": "multiclass_classification",
            "dataset": {
                "source": "csv", "path": csv_path,
                "y_true_col": "y_true", "y_score_col": "y_score",
                "feature_cols": ["f1", "f2"],
            },
            "metric": "macro_f1",
            "model_source": {"kind": "pickle", "path": model_path},
            "scenarios": [{"id": "label_noise", "params": {"p": 0.1}}],
            "n_trials": 3,
            "seed": 42,
        }
        run_id = run_from_spec_dict(spec)
        bundle = ResultBundle.model_validate_json(
            get_run_dir(run_id).results_json.read_text(encoding="utf-8")
        )
        aa = bundle.analysis_artifacts
        assert "dashboard_summary" in aa

    def test_multiclass_calibration_in_notes(self, multiclass_fixtures):
        csv_path, model_path = multiclass_fixtures
        spec = {
            "name": "mc_cal",
            "task": "multiclass_classification",
            "dataset": {
                "source": "csv", "path": csv_path,
                "y_true_col": "y_true", "y_score_col": "y_score",
                "feature_cols": ["f1", "f2"],
            },
            "metric": "macro_f1",
            "model_source": {"kind": "pickle", "path": model_path},
            "scenarios": [],
            "n_trials": 1,
            "seed": 42,
        }
        run_id = run_from_spec_dict(spec)
        bundle = ResultBundle.model_validate_json(
            get_run_dir(run_id).results_json.read_text(encoding="utf-8")
        )
        baseline_diag = bundle.notes.get("baseline_diagnostics", {})
        assert "multiclass_brier" in baseline_diag or "multiclass_ece" in baseline_diag


class TestRegressionFullAnalysis:
    def test_regression_has_failure_modes_and_dashboard(self, regression_fixtures):
        csv_path, model_path = regression_fixtures
        spec = {
            "name": "reg_analysis",
            "task": "regression",
            "dataset": {
                "source": "csv", "path": csv_path,
                "y_true_col": "y_true", "y_score_col": "y_score",
                "feature_cols": ["f0", "f1", "f2"],
            },
            "metric": "mae",
            "model_source": {"kind": "pickle", "path": model_path},
            "scenarios": [{"id": "label_noise", "params": {"p": 0.1}}],
            "n_trials": 3,
            "seed": 42,
        }
        run_id = run_from_spec_dict(spec)
        bundle = ResultBundle.model_validate_json(
            get_run_dir(run_id).results_json.read_text(encoding="utf-8")
        )
        aa = bundle.analysis_artifacts
        assert "failure_modes" in aa
        assert "dashboard_summary" in aa

    def test_regression_has_sensitivity(self, regression_fixtures):
        csv_path, model_path = regression_fixtures
        spec = {
            "name": "reg_sens",
            "task": "regression",
            "dataset": {
                "source": "csv", "path": csv_path,
                "y_true_col": "y_true", "y_score_col": "y_score",
                "feature_cols": ["f0", "f1", "f2"],
            },
            "metric": "mae",
            "model_source": {"kind": "pickle", "path": model_path},
            "scenarios": [{"id": "score_noise", "params": {"sigma": 0.05}}],
            "n_trials": 3,
            "seed": 42,
        }
        run_id = run_from_spec_dict(spec)
        bundle = ResultBundle.model_validate_json(
            get_run_dir(run_id).results_json.read_text(encoding="utf-8")
        )
        aa = bundle.analysis_artifacts
        assert "sensitivity" in aa


class TestBackwardCompatibility:
    def test_binary_still_has_full_analysis(self, tmp_path: Path):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 2))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=0).fit(X, y)
        model_path = tmp_path / "bin_model.pkl"
        with model_path.open("wb") as f:
            pickle.dump(model, f)
        df = pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "y_true": y, "y_score": np.zeros(n)})
        csv_path = tmp_path / "bin_data.csv"
        df.to_csv(csv_path, index=False)

        spec = {
            "name": "bin_compat",
            "task": "binary_classification",
            "dataset": {
                "source": "csv", "path": str(csv_path),
                "y_true_col": "y_true", "y_score_col": "y_score",
                "feature_cols": ["f1", "f2"],
            },
            "metric": "auc",
            "model_source": {"kind": "pickle", "path": str(model_path)},
            "scenarios": [{"id": "label_noise", "params": {"p": 0.1}}],
            "n_trials": 3,
            "seed": 42,
        }
        run_id = run_from_spec_dict(spec)
        bundle = ResultBundle.model_validate_json(
            get_run_dir(run_id).results_json.read_text(encoding="utf-8")
        )
        aa = bundle.analysis_artifacts
        assert "threshold_sweep" in aa
        assert "sensitivity" in aa
        assert "metric_disagreements" in aa
        assert "failure_modes" in aa
        assert "dashboard_summary" in aa
