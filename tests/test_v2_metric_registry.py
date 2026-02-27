"""Tests for extended metric registry with task-type awareness."""
from __future__ import annotations

from metrics_lie.metrics.registry import METRIC_REQUIREMENTS, MetricRequirement
from metrics_lie.model.surface import SurfaceType


def test_metric_requirement_has_task_types_field():
    req = METRIC_REQUIREMENTS[0]
    assert hasattr(req, "task_types")


def test_binary_metrics_include_binary_task():
    binary_ids = {"accuracy", "auc", "f1", "precision", "recall", "logloss",
                  "brier_score", "ece", "pr_auc", "matthews_corrcoef"}
    for req in METRIC_REQUIREMENTS:
        if req.metric_id in binary_ids:
            assert req.task_types is not None
            assert "binary_classification" in req.task_types


def test_regression_metrics_exclude_binary():
    regression_ids = {"mae", "mse", "rmse", "r2", "max_error"}
    for req in METRIC_REQUIREMENTS:
        if req.metric_id in regression_ids:
            assert req.task_types is not None
            assert "binary_classification" not in req.task_types
            assert "regression" in req.task_types


def test_multiclass_metrics_include_multiclass():
    mc_ids = {"macro_f1", "weighted_f1", "macro_precision", "macro_recall",
              "macro_auc", "cohens_kappa", "top_k_accuracy"}
    for req in METRIC_REQUIREMENTS:
        if req.metric_id in mc_ids:
            assert req.task_types is not None
            assert "multiclass_classification" in req.task_types


def test_regression_requirements_use_continuous_surface():
    regression_ids = {"mae", "mse", "rmse", "r2", "max_error"}
    for req in METRIC_REQUIREMENTS:
        if req.metric_id in regression_ids:
            assert SurfaceType.CONTINUOUS in req.requires_surface


def test_all_metrics_have_requirements():
    """Every metric in METRICS dict should have a MetricRequirement entry."""
    from metrics_lie.metrics.core import METRICS
    registered_ids = {req.metric_id for req in METRIC_REQUIREMENTS}
    for metric_id in METRICS:
        assert metric_id in registered_ids, f"Missing MetricRequirement for {metric_id}"


def test_r2_requires_min_samples_2():
    """R2 needs at least 2 samples."""
    for req in METRIC_REQUIREMENTS:
        if req.metric_id == "r2":
            assert req.min_samples == 2


def test_regression_requires_both_classes_false():
    """Regression metrics should not require both classes."""
    regression_ids = {"mae", "mse", "rmse", "r2", "max_error"}
    for req in METRIC_REQUIREMENTS:
        if req.metric_id in regression_ids:
            assert req.requires_both_classes is False


def test_metric_id_is_str_not_literal():
    """MetricRequirement.metric_id should be str, not Literal."""
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(MetricRequirement)}
    assert fields["metric_id"].type == "str"


def test_total_requirements_count():
    """Should have 22 total requirements (10 binary + 7 multiclass + 5 regression)."""
    assert len(METRIC_REQUIREMENTS) == 22
