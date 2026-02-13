"""Phase 9B tests: LABEL surface ingestion + scenario compatibility filter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from metrics_lie.datasets.loaders import load_binary_csv
from metrics_lie.execution import run_from_spec_dict
from metrics_lie.utils.paths import get_run_dir


def _create_label_csv(tmp_path: Path, n: int = 100) -> Path:
    """Create a CSV with binary 0/1 predictions (y_score column)."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, n)
    y_score = np.random.randint(0, 2, n)
    group = np.where(np.arange(n) < 50, "A", "B")
    csv_path = tmp_path / "label_data.csv"
    with open(csv_path, "w") as f:
        f.write("y_true,y_score,group\n")
        for yt, ys, g in zip(y_true, y_score, group):
            f.write(f"{yt},{ys},{g}\n")
    return csv_path


def test_label_surface_creates_prediction_surface(tmp_path: Path) -> None:
    """surface_type=label with 0/1 predictions produces prediction_surface with surface_type=label."""
    csv_path = _create_label_csv(tmp_path)
    spec_dict = {
        "name": "test_label_ingest",
        "task": "binary_classification",
        "dataset": {
            "source": "csv",
            "path": str(csv_path),
            "y_true_col": "y_true",
            "y_score_col": "y_score",
            "subgroup_col": "group",
        },
        "metric": "accuracy",
        "surface_source": {"kind": "csv_columns", "surface_type": "label"},
        "scenarios": [{"id": "label_noise", "params": {"p": 0.1}}],
        "n_trials": 10,
        "seed": 42,
    }
    run_id = run_from_spec_dict(spec_dict)
    bundle = json.loads(get_run_dir(run_id).results_json.read_text(encoding="utf-8"))
    assert bundle.get("prediction_surface") is not None
    assert bundle["prediction_surface"]["surface_type"] == "label"


def test_label_surface_resolves_classification_metrics(tmp_path: Path) -> None:
    """Applicable metrics for LABEL include accuracy, f1, precision, recall, matthews_corrcoef; not auc/logloss/ece."""
    csv_path = _create_label_csv(tmp_path)
    spec_dict = {
        "name": "test_label_metrics",
        "task": "binary_classification",
        "dataset": {
            "source": "csv",
            "path": str(csv_path),
            "y_true_col": "y_true",
            "y_score_col": "y_score",
        },
        "metric": "accuracy",
        "surface_source": {"kind": "csv_columns", "surface_type": "label"},
        "scenarios": [],
        "n_trials": 10,
        "seed": 42,
    }
    run_id = run_from_spec_dict(spec_dict)
    bundle = json.loads(get_run_dir(run_id).results_json.read_text(encoding="utf-8"))
    applicable = set(bundle.get("applicable_metrics", []))
    for required in ("accuracy", "f1", "precision", "recall", "matthews_corrcoef"):
        assert required in applicable, f"{required} should be applicable for LABEL"
    for excluded in ("auc", "logloss", "ece"):
        assert excluded not in applicable, f"{excluded} should not be applicable for LABEL"


def test_label_surface_rejects_non_binary(tmp_path: Path) -> None:
    """CSV with float y_score (e.g. 0.7) for label surface raises ValueError."""
    csv_path = tmp_path / "non_binary.csv"
    with open(csv_path, "w") as f:
        f.write("y_true,y_score,group\n")
        f.write("0,0,A\n")
        f.write("1,0.7,B\n")
    with pytest.raises(ValueError, match="binary"):
        load_binary_csv(
            path=str(csv_path),
            y_true_col="y_true",
            y_score_col="y_score",
            score_validation="label",
        )


def test_label_surface_filters_score_noise(tmp_path: Path) -> None:
    """Spec includes score_noise scenario; for LABEL surface it is filtered out (not in bundle scenarios)."""
    csv_path = _create_label_csv(tmp_path)
    spec_dict = {
        "name": "test_label_filter",
        "task": "binary_classification",
        "dataset": {
            "source": "csv",
            "path": str(csv_path),
            "y_true_col": "y_true",
            "y_score_col": "y_score",
        },
        "metric": "accuracy",
        "surface_source": {"kind": "csv_columns", "surface_type": "label"},
        "scenarios": [
            {"id": "label_noise", "params": {"p": 0.1}},
            {"id": "score_noise", "params": {"sigma": 0.1}},
        ],
        "n_trials": 10,
        "seed": 42,
    }
    run_id = run_from_spec_dict(spec_dict)
    bundle = json.loads(get_run_dir(run_id).results_json.read_text(encoding="utf-8"))
    scenario_ids = [s["scenario_id"] for s in bundle.get("scenarios", [])]
    assert "score_noise" not in scenario_ids
    assert "label_noise" in scenario_ids


def test_label_surface_allows_label_noise(tmp_path: Path) -> None:
    """Spec includes label_noise; for LABEL surface it runs and produces results."""
    csv_path = _create_label_csv(tmp_path)
    spec_dict = {
        "name": "test_label_noise",
        "task": "binary_classification",
        "dataset": {
            "source": "csv",
            "path": str(csv_path),
            "y_true_col": "y_true",
            "y_score_col": "y_score",
        },
        "metric": "accuracy",
        "surface_source": {"kind": "csv_columns", "surface_type": "label"},
        "scenarios": [{"id": "label_noise", "params": {"p": 0.1}}],
        "n_trials": 10,
        "seed": 42,
    }
    run_id = run_from_spec_dict(spec_dict)
    bundle = json.loads(get_run_dir(run_id).results_json.read_text(encoding="utf-8"))
    assert len(bundle.get("scenarios", [])) >= 1
    assert bundle["scenarios"][0]["scenario_id"] == "label_noise"


def test_label_surface_determinism(tmp_path: Path) -> None:
    """Two identical LABEL runs produce identical bundles."""
    csv_path = _create_label_csv(tmp_path)
    spec_dict = {
        "name": "test_label_determinism",
        "task": "binary_classification",
        "dataset": {
            "source": "csv",
            "path": str(csv_path),
            "y_true_col": "y_true",
            "y_score_col": "y_score",
            "subgroup_col": "group",
        },
        "metric": "accuracy",
        "surface_source": {"kind": "csv_columns", "surface_type": "label"},
        "scenarios": [{"id": "label_noise", "params": {"p": 0.1}}],
        "n_trials": 10,
        "seed": 42,
    }
    run_id_1 = run_from_spec_dict(spec_dict)
    run_id_2 = run_from_spec_dict(spec_dict)
    bundle_1 = json.loads(get_run_dir(run_id_1).results_json.read_text(encoding="utf-8"))
    bundle_2 = json.loads(get_run_dir(run_id_2).results_json.read_text(encoding="utf-8"))
    for b in [bundle_1, bundle_2]:
        b.pop("run_id", None)
        b.pop("created_at", None)
        for sr in b.get("scenarios", []):
            for artifact in sr.get("artifacts", []):
                artifact.pop("path", None)
        if "notes" in b and "spec_path" in b.get("notes", {}):
            b["notes"].pop("spec_path", None)
    assert bundle_1 == bundle_2


def test_scenario_filter_does_not_affect_probability(tmp_path: Path) -> None:
    """Probability surface with all 4 scenarios still runs all 4 (no filtering)."""
    np.random.seed(42)
    csv_path = tmp_path / "prob.csv"
    n = 80
    y_true = np.random.randint(0, 2, n)
    y_score = np.clip(y_true + np.random.normal(0, 0.2, n), 0, 1)
    with open(csv_path, "w") as f:
        f.write("y_true,y_score\n")
        for yt, ys in zip(y_true, y_score):
            f.write(f"{yt},{ys:.6f}\n")
    spec_dict = {
        "name": "test_prob_all_scenarios",
        "task": "binary_classification",
        "dataset": {
            "source": "csv",
            "path": str(csv_path),
            "y_true_col": "y_true",
            "y_score_col": "y_score",
        },
        "metric": "auc",
        "surface_source": {
            "kind": "csv_columns",
            "surface_type": "probability",
            "threshold": 0.5,
        },
        "scenarios": [
            {"id": "label_noise", "params": {"p": 0.1}},
            {"id": "score_noise", "params": {"sigma": 0.1}},
            {"id": "class_imbalance", "params": {"target_positive_rate": 0.3}},
            {"id": "threshold_gaming", "params": {}},
        ],
        "n_trials": 10,
        "seed": 42,
    }
    run_id = run_from_spec_dict(spec_dict)
    bundle = json.loads(get_run_dir(run_id).results_json.read_text(encoding="utf-8"))
    scenario_ids = {s["scenario_id"] for s in bundle.get("scenarios", [])}
    assert scenario_ids == {"label_noise", "score_noise", "class_imbalance", "threshold_gaming"}


def test_scenario_filter_does_not_affect_phase4_path(tmp_path: Path) -> None:
    """Spec without surface_source is unaffected by scenario filtering (Phase 4 path)."""
    csv_path = tmp_path / "phase4.csv"
    with open(csv_path, "w") as f:
        f.write("y_true,y_score\n")
        f.write("0,0.2\n")
        f.write("1,0.8\n")
    spec_dict = {
        "name": "test_phase4_no_surface",
        "task": "binary_classification",
        "dataset": {
            "source": "csv",
            "path": str(csv_path),
            "y_true_col": "y_true",
            "y_score_col": "y_score",
        },
        "metric": "auc",
        "scenarios": [{"id": "label_noise", "params": {"p": 0.1}}],
        "n_trials": 10,
        "seed": 42,
    }
    run_id = run_from_spec_dict(spec_dict)
    bundle = json.loads(get_run_dir(run_id).results_json.read_text(encoding="utf-8"))
    assert bundle.get("prediction_surface") is None
    assert bundle.get("applicable_metrics") == ["auc"]
    assert len(bundle.get("scenarios", [])) == 1
    assert bundle["scenarios"][0]["scenario_id"] == "label_noise"
