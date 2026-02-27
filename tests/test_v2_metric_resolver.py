"""Tests for task-type-aware MetricResolver."""
from __future__ import annotations

from metrics_lie.metrics.applicability import DatasetProperties, MetricResolver
from metrics_lie.model.surface import SurfaceType


def test_resolver_binary_returns_only_binary_metrics():
    resolver = MetricResolver()
    props = DatasetProperties(
        n_samples=100, n_positive=50, n_negative=50,
        has_subgroups=False, positive_rate=0.5,
    )
    result = resolver.resolve(
        task_type="binary_classification",
        surface_type=SurfaceType.PROBABILITY,
        dataset_props=props,
    )
    regression_metrics = {"mae", "mse", "rmse", "r2", "max_error"}
    multiclass_metrics = {"macro_f1", "weighted_f1", "macro_precision",
                          "macro_recall", "macro_auc", "cohens_kappa", "top_k_accuracy"}
    for m in regression_metrics | multiclass_metrics:
        assert m not in result.metrics, f"{m} should not be in binary results"


def test_resolver_multiclass_returns_multiclass_metrics():
    resolver = MetricResolver()
    props = DatasetProperties(
        n_samples=100, n_positive=0, n_negative=0,
        has_subgroups=False, positive_rate=0.0,
        n_classes=3,
    )
    result = resolver.resolve(
        task_type="multiclass_classification",
        surface_type=SurfaceType.PROBABILITY,
        dataset_props=props,
    )
    assert "macro_f1" in result.metrics
    assert "macro_auc" in result.metrics
    # Binary-only metrics should be excluded
    assert "auc" not in result.metrics
    assert "f1" not in result.metrics


def test_resolver_regression_returns_regression_metrics():
    resolver = MetricResolver()
    props = DatasetProperties(
        n_samples=100, n_positive=0, n_negative=0,
        has_subgroups=False, positive_rate=0.0,
    )
    result = resolver.resolve(
        task_type="regression",
        surface_type=SurfaceType.CONTINUOUS,
        dataset_props=props,
    )
    assert "mae" in result.metrics
    assert "mse" in result.metrics
    assert "rmse" in result.metrics
    assert "r2" in result.metrics
    assert "max_error" in result.metrics
    # No classification metrics
    assert "auc" not in result.metrics
    assert "f1" not in result.metrics
    assert "macro_f1" not in result.metrics


def test_resolver_regression_excludes_imbalance_warnings():
    """Regression should not get severe_imbalance_warning."""
    resolver = MetricResolver()
    props = DatasetProperties(
        n_samples=100, n_positive=0, n_negative=0,
        has_subgroups=False, positive_rate=0.0,
    )
    result = resolver.resolve(
        task_type="regression",
        surface_type=SurfaceType.CONTINUOUS,
        dataset_props=props,
    )
    assert "severe_imbalance_warning" not in result.warnings


def test_dataset_properties_accepts_n_classes():
    props = DatasetProperties(
        n_samples=100, n_positive=30, n_negative=70,
        has_subgroups=False, positive_rate=0.3, n_classes=3,
    )
    assert props.n_classes == 3


def test_dataset_properties_defaults_n_classes_none():
    props = DatasetProperties(
        n_samples=100, n_positive=50, n_negative=50,
        has_subgroups=False, positive_rate=0.5,
    )
    assert props.n_classes is None


def test_resolver_excluded_includes_task_type_reason():
    """Excluded metrics should have task_type reason."""
    resolver = MetricResolver()
    props = DatasetProperties(
        n_samples=100, n_positive=50, n_negative=50,
        has_subgroups=False, positive_rate=0.5,
    )
    result = resolver.resolve(
        task_type="binary_classification",
        surface_type=SurfaceType.PROBABILITY,
        dataset_props=props,
    )
    # macro_f1 should be excluded with task_type reason
    excluded_ids = [e[0] for e in result.excluded]
    assert "macro_f1" in excluded_ids
    macro_f1_reason = next(e[1] for e in result.excluded if e[0] == "macro_f1")
    assert "task_type" in macro_f1_reason
