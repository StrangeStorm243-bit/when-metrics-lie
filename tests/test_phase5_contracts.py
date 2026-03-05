"""Tests for Phase 5 contract additions — task_type and task-specific fields."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from web.backend.app.contracts import (
    ExperimentSummary,
    ModelUploadResponse,
    ResultSummary,
    SupportedFormat,
)


def test_experiment_summary_task_type_default():
    """ExperimentSummary defaults to binary_classification."""
    summary = ExperimentSummary(
        id="exp1",
        name="test",
        metric_id="auc",
        stress_suite_id="default",
        status="created",
        created_at="2026-01-01T00:00:00+00:00",
    )
    assert summary.task_type == "binary_classification"


def test_experiment_summary_task_type_explicit():
    summary = ExperimentSummary(
        id="exp1",
        name="test",
        metric_id="mae",
        stress_suite_id="default",
        task_type="regression",
        status="created",
        created_at="2026-01-01T00:00:00+00:00",
    )
    assert summary.task_type == "regression"


def test_result_summary_task_specific_fields_default_none():
    """All task-specific fields default to None."""
    result = ResultSummary(
        experiment_id="exp1",
        run_id="run1",
        headline_score=0.9,
        generated_at="2026-01-01T00:00:00+00:00",
    )
    assert result.task_type == "binary_classification"
    assert result.confusion_matrix is None
    assert result.class_names is None
    assert result.per_class_metrics is None
    assert result.residual_stats is None
    assert result.ranking_metrics is None


def test_result_summary_confusion_matrix_round_trip():
    """Confusion matrix round-trips through the contract."""
    cm = [[50, 10], [5, 35]]
    result = ResultSummary(
        experiment_id="exp1",
        run_id="run1",
        headline_score=0.85,
        task_type="binary_classification",
        confusion_matrix=cm,
        class_names=["0", "1"],
        generated_at="2026-01-01T00:00:00+00:00",
    )
    assert result.confusion_matrix == cm
    assert result.class_names == ["0", "1"]


def test_result_summary_per_class_metrics():
    result = ResultSummary(
        experiment_id="exp1",
        run_id="run1",
        headline_score=0.8,
        task_type="multiclass_classification",
        per_class_metrics={
            "0": {"precision": 0.9, "recall": 0.85, "f1": 0.87, "support": 50},
            "1": {"precision": 0.8, "recall": 0.75, "f1": 0.77, "support": 40},
            "2": {"precision": 0.7, "recall": 0.65, "f1": 0.67, "support": 30},
        },
        generated_at="2026-01-01T00:00:00+00:00",
    )
    assert len(result.per_class_metrics) == 3
    assert result.per_class_metrics["0"]["precision"] == 0.9


def test_result_summary_residual_stats():
    result = ResultSummary(
        experiment_id="exp1",
        run_id="run1",
        headline_score=0.5,
        task_type="regression",
        residual_stats={
            "mean": 0.01, "std": 0.5, "min": -2.0, "max": 1.8,
            "median": 0.0, "mae": 0.3, "rmse": 0.5,
        },
        generated_at="2026-01-01T00:00:00+00:00",
    )
    assert result.residual_stats["mae"] == pytest.approx(0.3)
    assert result.residual_stats["rmse"] == pytest.approx(0.5)


def test_model_upload_response_task_type():
    resp = ModelUploadResponse(
        model_id="abc123",
        original_filename="model.pkl",
        model_class="sklearn.linear_model.LogisticRegression",
        task_type="multiclass_classification",
        n_classes=5,
        capabilities={"predict": True, "predict_proba": True},
        file_size_bytes=1024,
    )
    assert resp.task_type == "multiclass_classification"
    assert resp.n_classes == 5


def test_supported_format_contract():
    fmt = SupportedFormat(
        format_id="pickle",
        name="sklearn Pickle",
        extensions=[".pkl", ".joblib"],
        task_types=["binary_classification", "regression"],
    )
    assert fmt.format_id == "pickle"
    assert ".pkl" in fmt.extensions
