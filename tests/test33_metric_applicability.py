from __future__ import annotations

from metrics_lie.metrics.applicability import DatasetProperties, MetricResolver
from metrics_lie.model.surface import SurfaceType


def test_metric_resolver_probability_surface() -> None:
    resolver = MetricResolver()
    props = DatasetProperties(
        n_samples=100,
        n_positive=30,
        n_negative=70,
        has_subgroups=False,
        positive_rate=0.3,
    )
    res = resolver.resolve(
        task_type="binary_classification",
        surface_type=SurfaceType.PROBABILITY,
        dataset_props=props,
    )
    assert "auc" in res.metrics
    assert "pr_auc" in res.metrics
    assert "logloss" in res.metrics
    assert "brier_score" in res.metrics
    assert "ece" in res.metrics
    assert "accuracy" in res.metrics


def test_metric_resolver_label_surface_excludes_scores() -> None:
    resolver = MetricResolver()
    props = DatasetProperties(
        n_samples=100,
        n_positive=50,
        n_negative=50,
        has_subgroups=False,
        positive_rate=0.5,
    )
    res = resolver.resolve(
        task_type="binary_classification",
        surface_type=SurfaceType.LABEL,
        dataset_props=props,
    )
    assert "accuracy" in res.metrics
    assert "auc" not in res.metrics
    assert "logloss" not in res.metrics


def test_metric_resolver_missing_class_excludes() -> None:
    resolver = MetricResolver()
    props = DatasetProperties(
        n_samples=20,
        n_positive=0,
        n_negative=20,
        has_subgroups=False,
        positive_rate=0.0,
    )
    res = resolver.resolve(
        task_type="binary_classification",
        surface_type=SurfaceType.PROBABILITY,
        dataset_props=props,
    )
    assert res.metrics == []
    assert any("requires both classes" in reason for _, reason in res.excluded)
