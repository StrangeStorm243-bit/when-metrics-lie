"""Phase 8 M1 tests: Surface-from-CSV ingestion + multi-metric activation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from metrics_lie.execution import run_from_spec_dict
from metrics_lie.utils.paths import get_run_dir


def _create_test_csv(tmp_path: Path) -> Path:
    """Create a test CSV with binary classification data."""
    np.random.seed(42)
    n = 100
    y_true = np.random.randint(0, 2, n)
    y_score = np.clip(y_true + np.random.normal(0, 0.2, n), 0, 1)
    group = np.where(np.arange(n) < 50, "A", "B")

    csv_path = tmp_path / "test_data.csv"
    with open(csv_path, "w") as f:
        f.write("y_true,y_score,group\n")
        for yt, ys, g in zip(y_true, y_score, group):
            f.write(f"{yt},{ys:.6f},{g}\n")
    return csv_path


def test_surface_source_creates_prediction_surface(tmp_path: Path) -> None:
    """Spec with surface_source and no model_source creates a prediction surface."""
    csv_path = _create_test_csv(tmp_path)

    spec_dict = {
        "name": "test_surface_ingest",
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

    # Should have a prediction_surface
    assert bundle.get("prediction_surface") is not None
    assert bundle["prediction_surface"]["surface_type"] == "probability"
    assert bundle["prediction_surface"]["n_samples"] == 100


def test_surface_source_resolves_multiple_metrics(tmp_path: Path) -> None:
    """Bundle's applicable_metrics has more than 1 metric when surface_source is set."""
    csv_path = _create_test_csv(tmp_path)

    spec_dict = {
        "name": "test_multi_metric",
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

    # Should have multiple applicable metrics (MetricResolver triggered)
    applicable = bundle.get("applicable_metrics", [])
    assert len(applicable) > 1, f"Expected >1 applicable metrics, got {applicable}"

    # Should have metric_results for each applicable metric
    metric_results = bundle.get("metric_results", {})
    assert len(metric_results) == len(applicable)
    for metric_id in applicable:
        assert metric_id in metric_results, f"Missing metric_results for {metric_id}"


def test_surface_source_preserves_phase4(tmp_path: Path) -> None:
    """Spec without surface_source or model_source preserves single-metric behavior."""
    csv_path = _create_test_csv(tmp_path)

    spec_dict = {
        "name": "test_phase4_preserved",
        "task": "binary_classification",
        "dataset": {
            "source": "csv",
            "path": str(csv_path),
            "y_true_col": "y_true",
            "y_score_col": "y_score",
            "subgroup_col": "group",
        },
        "metric": "auc",
        # NO surface_source, NO model_source
        "scenarios": [{"id": "label_noise", "params": {"p": 0.1}}],
        "n_trials": 10,
        "seed": 42,
    }

    run_id = run_from_spec_dict(spec_dict)
    run_paths = get_run_dir(run_id)
    bundle = json.loads(run_paths.results_json.read_text(encoding="utf-8"))

    # Should have only the specified metric (single-metric mode)
    applicable = bundle.get("applicable_metrics", [])
    assert applicable == ["auc"], f"Expected ['auc'], got {applicable}"

    # No prediction surface (Phase 4 behavior)
    assert bundle.get("prediction_surface") is None


def test_surface_source_determinism(tmp_path: Path) -> None:
    """Two identical runs with surface_source produce identical bundles."""
    csv_path = _create_test_csv(tmp_path)

    spec_dict = {
        "name": "test_determinism",
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

    # Run twice
    run_id_1 = run_from_spec_dict(spec_dict)
    run_id_2 = run_from_spec_dict(spec_dict)

    run_paths_1 = get_run_dir(run_id_1)
    run_paths_2 = get_run_dir(run_id_2)

    bundle_1 = json.loads(run_paths_1.results_json.read_text(encoding="utf-8"))
    bundle_2 = json.loads(run_paths_2.results_json.read_text(encoding="utf-8"))

    # Strip non-deterministic fields
    for b in [bundle_1, bundle_2]:
        b.pop("run_id", None)
        b.pop("created_at", None)
        # Strip artifact paths which contain run_id
        for sr in b.get("scenarios", []):
            for artifact in sr.get("artifacts", []):
                artifact.pop("path", None)
        # Strip notes.spec_path which may vary
        if "notes" in b and "spec_path" in b.get("notes", {}):
            b["notes"].pop("spec_path", None)

    assert bundle_1 == bundle_2, "Bundles should be identical after stripping non-deterministic fields"
