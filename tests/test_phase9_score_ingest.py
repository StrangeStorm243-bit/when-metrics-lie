"""Phase 9A tests: SCORE surface ingestion from CSV."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from metrics_lie.datasets.loaders import load_binary_csv
from metrics_lie.execution import run_from_spec_dict
from metrics_lie.spec import SurfaceSourceSpec
from metrics_lie.utils.paths import get_run_dir


def _create_probability_csv(tmp_path: Path, n: int = 100) -> Path:
    """Create a CSV with probabilities in [0, 1] (Phase 8 style)."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, n)
    y_score = np.clip(y_true + np.random.normal(0, 0.2, n), 0, 1)
    group = np.where(np.arange(n) < 50, "A", "B")
    csv_path = tmp_path / "prob_data.csv"
    with open(csv_path, "w") as f:
        f.write("y_true,y_score,group\n")
        for yt, ys, g in zip(y_true, y_score, group):
            f.write(f"{yt},{ys:.6f},{g}\n")
    return csv_path


def _create_score_csv(tmp_path: Path, n: int = 100, low: float = -2.0, high: float = 3.0) -> Path:
    """Create a CSV with scores in [low, high] (arbitrary range)."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, n)
    y_score = np.random.uniform(low, high, n)
    group = np.where(np.arange(n) < 50, "A", "B")
    csv_path = tmp_path / "score_data.csv"
    with open(csv_path, "w") as f:
        f.write("y_true,y_score,group\n")
        for yt, ys, g in zip(y_true, y_score, group):
            f.write(f"{yt},{ys:.6f},{g}\n")
    return csv_path


def test_score_surface_creates_prediction_surface(tmp_path: Path) -> None:
    """Spec with surface_type=score and CSV with scores in [-2, 3] produces score surface."""
    csv_path = _create_score_csv(tmp_path, low=-2.0, high=3.0)
    spec_dict = {
        "name": "test_score_ingest",
        "task": "binary_classification",
        "dataset": {
            "source": "csv",
            "path": str(csv_path),
            "y_true_col": "y_true",
            "y_score_col": "y_score",
            "subgroup_col": "group",
        },
        "metric": "auc",
        "surface_source": {
            "kind": "csv_columns",
            "surface_type": "score",
            "positive_label": 1,
        },
        "scenarios": [{"id": "label_noise", "params": {"p": 0.1}}],
        "n_trials": 10,
        "seed": 42,
    }
    run_id = run_from_spec_dict(spec_dict)
    run_paths = get_run_dir(run_id)
    bundle = json.loads(run_paths.results_json.read_text(encoding="utf-8"))
    assert bundle.get("prediction_surface") is not None
    assert bundle["prediction_surface"]["surface_type"] == "score"
    assert bundle["prediction_surface"]["n_samples"] == 100


def test_score_surface_resolves_auc_pr_auc_only(tmp_path: Path) -> None:
    """Applicable metrics for SCORE are only auc and pr_auc; no calibration/classification."""
    csv_path = _create_score_csv(tmp_path)
    spec_dict = {
        "name": "test_score_metrics",
        "task": "binary_classification",
        "dataset": {
            "source": "csv",
            "path": str(csv_path),
            "y_true_col": "y_true",
            "y_score_col": "y_score",
        },
        "metric": "auc",
        "surface_source": {"kind": "csv_columns", "surface_type": "score"},
        "scenarios": [],
        "n_trials": 10,
        "seed": 42,
    }
    run_id = run_from_spec_dict(spec_dict)
    run_paths = get_run_dir(run_id)
    bundle = json.loads(run_paths.results_json.read_text(encoding="utf-8"))
    applicable = set(bundle.get("applicable_metrics", []))
    assert applicable.issubset({"auc", "pr_auc"}), f"Expected subset of {{auc, pr_auc}}, got {applicable}"
    for excluded in ("logloss", "ece", "brier_score", "accuracy"):
        assert excluded not in applicable, f"{excluded} should not be applicable for SCORE"


def test_score_surface_skips_probability_artifacts(tmp_path: Path) -> None:
    """SCORE surfaces get threshold_sweep, sensitivity, metric_disagreements, failure_modes (Phase 9C)."""
    csv_path = _create_score_csv(tmp_path)
    spec_dict = {
        "name": "test_score_artifacts",
        "task": "binary_classification",
        "dataset": {
            "source": "csv",
            "path": str(csv_path),
            "y_true_col": "y_true",
            "y_score_col": "y_score",
        },
        "metric": "auc",
        "surface_source": {"kind": "csv_columns", "surface_type": "score"},
        "scenarios": [{"id": "label_noise", "params": {"p": 0.1}}],
        "n_trials": 10,
        "seed": 42,
    }
    run_id = run_from_spec_dict(spec_dict)
    run_paths = get_run_dir(run_id)
    bundle = json.loads(run_paths.results_json.read_text(encoding="utf-8"))
    artifacts = bundle.get("analysis_artifacts", {})
    assert "threshold_sweep" in artifacts
    assert "sensitivity" in artifacts
    assert "metric_disagreements" in artifacts
    assert "failure_modes" in artifacts


def test_score_surface_determinism(tmp_path: Path) -> None:
    """Two identical SCORE runs produce identical bundles (after stripping non-deterministic fields)."""
    csv_path = _create_score_csv(tmp_path)
    spec_dict = {
        "name": "test_score_determinism",
        "task": "binary_classification",
        "dataset": {
            "source": "csv",
            "path": str(csv_path),
            "y_true_col": "y_true",
            "y_score_col": "y_score",
            "subgroup_col": "group",
        },
        "metric": "auc",
        "surface_source": {"kind": "csv_columns", "surface_type": "score"},
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


def test_score_surface_rejects_nan(tmp_path: Path) -> None:
    """CSV with NaN in y_score for score surface raises ValueError."""
    csv_path = tmp_path / "nan_scores.csv"
    with open(csv_path, "w") as f:
        f.write("y_true,y_score,group\n")
        f.write("0,1.5,A\n")
        f.write("1,nan,B\n")
    with pytest.raises(ValueError, match="NaNs"):
        load_binary_csv(
            path=str(csv_path),
            y_true_col="y_true",
            y_score_col="y_score",
            score_validation="score",
        )


def test_score_surface_rejects_inf(tmp_path: Path) -> None:
    """CSV with Inf in y_score for score surface raises ValueError."""
    csv_path = tmp_path / "inf_scores.csv"
    with open(csv_path, "w") as f:
        f.write("y_true,y_score,group\n")
        f.write("0,1.0,A\n")
        f.write("1,inf,B\n")
    with pytest.raises(ValueError, match="Inf"):
        load_binary_csv(
            path=str(csv_path),
            y_true_col="y_true",
            y_score_col="y_score",
            score_validation="score",
        )


def test_score_surface_rejects_threshold() -> None:
    """Spec with surface_type=score and threshold=0.5 raises ValidationError."""
    with pytest.raises(ValidationError, match="threshold"):
        SurfaceSourceSpec(
            kind="csv_columns",
            surface_type="score",
            threshold=0.5,
            positive_label=1,
        )


def test_probability_surface_unchanged(tmp_path: Path) -> None:
    """Regression: surface_type=probability still works identically (Phase 8 behavior)."""
    csv_path = _create_probability_csv(tmp_path)
    spec_dict = {
        "name": "test_prob_unchanged",
        "task": "binary_classification",
        "dataset": {
            "source": "csv",
            "path": str(csv_path),
            "y_true_col": "y_true",
            "y_score_col": "y_score",
            "subgroup_col": "group",
        },
        "metric": "auc",
        "surface_source": {
            "kind": "csv_columns",
            "surface_type": "probability",
            "threshold": 0.5,
            "positive_label": 1,
        },
        "scenarios": [{"id": "label_noise", "params": {"p": 0.1}}],
        "n_trials": 10,
        "seed": 42,
    }
    run_id = run_from_spec_dict(spec_dict)
    run_paths = get_run_dir(run_id)
    bundle = json.loads(run_paths.results_json.read_text(encoding="utf-8"))
    assert bundle["prediction_surface"]["surface_type"] == "probability"
    assert bundle["prediction_surface"]["n_samples"] == 100
    # Probability gets full artifact set
    artifacts = bundle.get("analysis_artifacts", {})
    assert "threshold_sweep" in artifacts
    assert "failure_modes" in artifacts
