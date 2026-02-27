"""Tests for direction-aware dashboard summary."""
from __future__ import annotations
from metrics_lie.analysis.dashboard import build_dashboard_summary

def test_dashboard_lower_is_better_flags_increase_as_drop():
    metric_results = {"mae": {"mean": 1.0}, "rmse": {"mean": 1.5}}
    scenario_results_by_metric = {
        "mae": [{"scenario_id": "label_noise_0.1", "metric": {"mean": 1.3}}],
        "rmse": [{"scenario_id": "label_noise_0.1", "metric": {"mean": 1.7}}],
    }
    dashboard = build_dashboard_summary(
        primary_metric="mae", surface_type="continuous",
        metric_results=metric_results,
        scenario_results_by_metric=scenario_results_by_metric,
        metric_directions={"mae": False, "rmse": False},
    )
    assert "mae" in dashboard.risk_summary["metrics_with_large_drops"]
    assert "rmse" in dashboard.risk_summary["metrics_with_large_drops"]

def test_dashboard_higher_is_better_flags_decrease_as_drop():
    metric_results = {"auc": {"mean": 0.90}}
    scenario_results_by_metric = {
        "auc": [{"scenario_id": "label_noise_0.1", "metric": {"mean": 0.82}}],
    }
    dashboard = build_dashboard_summary(
        primary_metric="auc", surface_type="probability",
        metric_results=metric_results,
        scenario_results_by_metric=scenario_results_by_metric,
        metric_directions={"auc": True},
    )
    assert "auc" in dashboard.risk_summary["metrics_with_large_drops"]

def test_dashboard_backward_compat_no_direction():
    metric_results = {"auc": {"mean": 0.90}}
    scenario_results_by_metric = {
        "auc": [{"scenario_id": "label_noise_0.1", "metric": {"mean": 0.82}}],
    }
    dashboard = build_dashboard_summary(
        primary_metric="auc", surface_type="probability",
        metric_results=metric_results,
        scenario_results_by_metric=scenario_results_by_metric,
    )
    assert "auc" in dashboard.risk_summary["metrics_with_large_drops"]
