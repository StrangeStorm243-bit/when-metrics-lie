"""Tests for metric direction (higher_is_better) on MetricRequirement."""
from __future__ import annotations
import pytest
from metrics_lie.metrics.registry import METRIC_DIRECTION, METRIC_REQUIREMENTS

HIGHER_IS_BETTER_METRICS = {
    "accuracy", "auc", "f1", "precision", "recall", "pr_auc",
    "matthews_corrcoef", "macro_f1", "weighted_f1", "macro_precision",
    "macro_recall", "macro_auc", "cohens_kappa", "top_k_accuracy", "r2",
}
LOWER_IS_BETTER_METRICS = {
    "logloss", "brier_score", "ece", "mae", "mse", "rmse", "max_error",
}

def _requirements_by_id():
    return {r.metric_id: r for r in METRIC_REQUIREMENTS}

def test_metric_requirement_has_higher_is_better_field():
    req = METRIC_REQUIREMENTS[0]
    assert hasattr(req, "higher_is_better")

@pytest.mark.parametrize("metric_id", sorted(HIGHER_IS_BETTER_METRICS))
def test_higher_is_better_true(metric_id):
    reqs = _requirements_by_id()
    assert metric_id in reqs
    assert reqs[metric_id].higher_is_better is True

@pytest.mark.parametrize("metric_id", sorted(LOWER_IS_BETTER_METRICS))
def test_lower_is_better(metric_id):
    reqs = _requirements_by_id()
    assert metric_id in reqs
    assert reqs[metric_id].higher_is_better is False

def test_metric_direction_dict():
    assert METRIC_DIRECTION["auc"] is True
    assert METRIC_DIRECTION["mae"] is False
    assert len(METRIC_DIRECTION) == len(METRIC_REQUIREMENTS)
