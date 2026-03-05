"""Tests for Phase 5 task-aware bundle transform."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from metrics_lie.schema import MetricSummary, ResultBundle, ScenarioResult

from web.backend.app.bundle_transform import bundle_to_result_summary


def _metric(mean: float = 0.85) -> MetricSummary:
    return MetricSummary(mean=mean, std=0.02, q05=0.81, q50=0.85, q95=0.89, n=200)


def _bundle(
    *,
    task_type: str = "binary_classification",
    baseline_mean: float = 0.90,
    scenarios: list[ScenarioResult] | None = None,
    baseline_diagnostics: dict | None = None,
    task_specific: dict | None = None,
    analysis_artifacts: dict | None = None,
) -> ResultBundle:
    if scenarios is None:
        scenarios = [
            ScenarioResult(scenario_id="label_noise", params={"p": 0.1}, metric=_metric(0.85)),
        ]
    notes: dict = {"phase": "test"}
    if baseline_diagnostics is not None:
        notes["baseline_diagnostics"] = baseline_diagnostics
    if task_specific is not None:
        notes["task_specific"] = task_specific
    return ResultBundle(
        run_id="TESTRUN01",
        experiment_name="test",
        metric_name="auc",
        task_type=task_type,
        baseline=_metric(baseline_mean),
        scenarios=scenarios,
        notes=notes,
        analysis_artifacts=analysis_artifacts or {},
        applicable_metrics=["auc"],
        created_at="2026-01-15T12:00:00+00:00",
    )


# --- task_type passthrough ---

def test_task_type_binary():
    result = bundle_to_result_summary(_bundle(task_type="binary_classification"), "e", "r")
    assert result.task_type == "binary_classification"


def test_task_type_regression():
    result = bundle_to_result_summary(_bundle(task_type="regression"), "e", "r")
    assert result.task_type == "regression"


def test_task_type_multiclass():
    result = bundle_to_result_summary(_bundle(task_type="multiclass_classification"), "e", "r")
    assert result.task_type == "multiclass_classification"


# --- confusion matrix ---

def test_confusion_matrix_extracted():
    ts = {"confusion_matrix": [[45, 5], [10, 40]], "class_names": ["0", "1"]}
    result = bundle_to_result_summary(_bundle(task_specific=ts), "e", "r")
    assert result.confusion_matrix == [[45, 5], [10, 40]]
    assert result.class_names == ["0", "1"]


def test_confusion_matrix_none_when_absent():
    result = bundle_to_result_summary(_bundle(), "e", "r")
    assert result.confusion_matrix is None


# --- per-class metrics ---

def test_per_class_metrics_extracted():
    ts = {
        "per_class_metrics": {
            "0": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 50},
            "1": {"precision": 0.7, "recall": 0.6, "f1": 0.65, "support": 30},
            "2": {"precision": 0.8, "recall": 0.9, "f1": 0.85, "support": 40},
        }
    }
    result = bundle_to_result_summary(
        _bundle(task_type="multiclass_classification", task_specific=ts), "e", "r"
    )
    assert result.per_class_metrics is not None
    assert len(result.per_class_metrics) == 3
    assert result.per_class_metrics["0"]["precision"] == 0.9


# --- residual stats ---

def test_residual_stats_extracted():
    ts = {
        "residual_stats": {
            "mean": 0.01, "std": 0.5, "min": -2.0, "max": 1.8,
            "median": 0.0, "mae": 0.3, "rmse": 0.5,
        }
    }
    result = bundle_to_result_summary(
        _bundle(task_type="regression", task_specific=ts), "e", "r"
    )
    assert result.residual_stats is not None
    assert result.residual_stats["mae"] == pytest.approx(0.3)
    assert result.residual_stats["rmse"] == pytest.approx(0.5)


def test_residual_stats_none_for_classification():
    result = bundle_to_result_summary(_bundle(task_type="binary_classification"), "e", "r")
    assert result.residual_stats is None


# --- severity classification ---

def test_severity_high_for_large_delta():
    """Delta >= 0.1 should be 'high'."""
    bundle = _bundle(
        baseline_mean=0.90,
        scenarios=[ScenarioResult(scenario_id="test", params={}, metric=_metric(0.78))],
    )
    result = bundle_to_result_summary(bundle, "e", "r")
    assert result.scenario_results[0].severity == "high"


def test_severity_med_for_moderate_delta():
    """Delta 0.05-0.1 should be 'med'."""
    bundle = _bundle(
        baseline_mean=0.90,
        scenarios=[ScenarioResult(scenario_id="test", params={}, metric=_metric(0.84))],
    )
    result = bundle_to_result_summary(bundle, "e", "r")
    assert result.scenario_results[0].severity == "med"


def test_severity_low_for_small_delta():
    """Delta 0.01-0.05 should be 'low'."""
    bundle = _bundle(
        baseline_mean=0.90,
        scenarios=[ScenarioResult(scenario_id="test", params={}, metric=_metric(0.88))],
    )
    result = bundle_to_result_summary(bundle, "e", "r")
    assert result.scenario_results[0].severity == "low"


def test_severity_none_for_tiny_delta():
    """Delta < 0.01 should be None."""
    bundle = _bundle(
        baseline_mean=0.90,
        scenarios=[ScenarioResult(scenario_id="test", params={}, metric=_metric(0.899))],
    )
    result = bundle_to_result_summary(bundle, "e", "r")
    assert result.scenario_results[0].severity is None


# --- component scores multi-task ---

def test_multiclass_calibration_components():
    diag = {"multiclass_brier": 0.15, "multiclass_ece": 0.08}
    result = bundle_to_result_summary(
        _bundle(task_type="multiclass_classification", baseline_diagnostics=diag), "e", "r"
    )
    names = [c.name for c in result.component_scores]
    assert "multiclass_brier" in names
    assert "multiclass_ece" in names


def test_regression_component_scores():
    ts = {"residual_stats": {"mae": 0.3, "rmse": 0.5, "mean": 0, "std": 0.5, "min": -1, "max": 1, "median": 0}}
    result = bundle_to_result_summary(
        _bundle(task_type="regression", task_specific=ts), "e", "r"
    )
    names = [c.name for c in result.component_scores]
    assert "mae" in names
    assert "rmse" in names


# --- flags multi-task ---

def test_high_ece_flag_still_works():
    result = bundle_to_result_summary(
        _bundle(baseline_diagnostics={"ece": 0.15}), "e", "r"
    )
    assert any(f.code == "high_ece" for f in result.flags)


def test_multiclass_ece_flag():
    result = bundle_to_result_summary(
        _bundle(
            task_type="multiclass_classification",
            baseline_diagnostics={"multiclass_ece": 0.15},
        ),
        "e", "r",
    )
    assert any(f.code == "high_multiclass_ece" for f in result.flags)


def test_residual_outlier_flag():
    ts = {"residual_stats": {"mean": 0, "std": 1.0, "min": -5.0, "max": 4.0, "median": 0, "mae": 0.8, "rmse": 1.0}}
    result = bundle_to_result_summary(
        _bundle(task_type="regression", task_specific=ts), "e", "r"
    )
    assert any(f.code == "high_residual_outliers" for f in result.flags)


def test_no_residual_flag_when_within_bounds():
    ts = {"residual_stats": {"mean": 0, "std": 1.0, "min": -2.0, "max": 2.0, "median": 0, "mae": 0.5, "rmse": 1.0}}
    result = bundle_to_result_summary(
        _bundle(task_type="regression", task_specific=ts), "e", "r"
    )
    assert not any(f.code == "high_residual_outliers" for f in result.flags)


# --- backward compat: existing test from milestone4 should still pass pattern ---

def test_existing_binary_transform_unchanged():
    """Binary bundles without task_specific still work (backward compat)."""
    bundle = _bundle(
        baseline_diagnostics={"brier": 0.12, "ece": 0.08},
    )
    result = bundle_to_result_summary(bundle, "exp_1", "run_1")
    assert result.headline_score == pytest.approx(0.90)
    assert result.task_type == "binary_classification"
    names = [c.name for c in result.component_scores]
    assert "brier_score" in names
    assert "ece_score" in names
    assert len(result.flags) == 0  # ECE 0.08 < 0.1
