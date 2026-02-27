"""Phase 8: Multi-metric dashboard summary artifact generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Threshold for flagging a metric as having a "large drop"
LARGE_DROP_THRESHOLD = 0.05


@dataclass
class DashboardSummary:
    """Structured multi-metric dashboard summary."""

    version: str
    primary_metric: str
    surface_type: str
    metrics: list[dict[str, Any]]
    risk_summary: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "primary_metric": self.primary_metric,
            "surface_type": self.surface_type,
            "metrics": self.metrics,
            "risk_summary": self.risk_summary,
        }


def build_dashboard_summary(
    *,
    primary_metric: str,
    surface_type: str,
    metric_results: dict[str, Any],
    scenario_results_by_metric: dict[str, list[Any]],
    metric_directions: dict[str, bool] | None = None,
) -> DashboardSummary:
    """Build a multi-metric dashboard summary from run results.

    Args:
        primary_metric: The primary metric ID (e.g., 'auc').
        surface_type: The surface type (e.g., 'probability').
        metric_results: Dict mapping metric_id to MetricSummary-like dict.
        scenario_results_by_metric: Dict mapping metric_id to list of ScenarioResult-like dicts.

    Returns:
        DashboardSummary with deterministic ordering and risk flags.
    """
    metrics_list: list[dict[str, Any]] = []
    metrics_with_large_drops: list[str] = []
    stable_metrics: list[str] = []
    worst_overall_delta: float | None = None
    worst_overall_metric: str | None = None
    worst_overall_scenario: str | None = None

    # Process metrics in sorted order for determinism
    for metric_id in sorted(metric_results.keys()):
        mr = metric_results[metric_id]
        baseline = _get_mean(mr) if isinstance(mr, dict) else None
        if baseline is None:
            continue

        scenarios = scenario_results_by_metric.get(metric_id, [])
        if not scenarios:
            # No scenarios - metric is stable by default
            metrics_list.append({
                "metric_id": metric_id,
                "baseline": baseline,
                "worst_scenario": None,
                "best_scenario": None,
                "scenario_range": 0.0,
                "n_scenarios": 0,
            })
            stable_metrics.append(metric_id)
            continue

        # Determine direction for this metric
        hib = True  # default: higher is better
        if metric_directions is not None and metric_id in metric_directions:
            hib = metric_directions[metric_id]

        # Find worst and best scenarios
        worst_scenario: dict[str, Any] | None = None
        best_scenario: dict[str, Any] | None = None
        worst_degradation_delta = 0.0
        best_delta = float("-inf")

        for sr in scenarios:
            sr_dict = sr.model_dump() if hasattr(sr, "model_dump") else sr
            metric_summary = sr_dict.get("metric", {})
            score = _get_mean(metric_summary)
            if score is None:
                continue

            delta = score - baseline
            scenario_id = sr_dict.get("scenario_id", "unknown")

            # For higher-is-better, degradation = negative delta.
            # For lower-is-better, degradation = positive delta (flip sign).
            degradation_delta = delta if hib else -delta

            if degradation_delta < worst_degradation_delta:
                worst_degradation_delta = degradation_delta
                worst_scenario = {
                    "scenario_id": scenario_id,
                    "score": score,
                    "delta": delta,
                }

            if delta > best_delta:
                best_delta = delta
                best_scenario = {
                    "scenario_id": scenario_id,
                    "score": score,
                    "delta": delta,
                }

        # Calculate scenario range using raw deltas
        worst_raw_delta = worst_scenario["delta"] if worst_scenario else 0.0
        scenario_range = abs(worst_raw_delta - best_delta) if best_delta > float("-inf") else 0.0

        metrics_list.append({
            "metric_id": metric_id,
            "baseline": baseline,
            "worst_scenario": worst_scenario,
            "best_scenario": best_scenario,
            "scenario_range": scenario_range,
            "n_scenarios": len(scenarios),
        })

        # Check for large drops using degradation_delta
        if worst_degradation_delta < -LARGE_DROP_THRESHOLD:
            metrics_with_large_drops.append(metric_id)
            # Track worst overall
            if worst_overall_delta is None or worst_degradation_delta < worst_overall_delta:
                worst_overall_delta = worst_degradation_delta
                worst_overall_metric = metric_id
                worst_overall_scenario = worst_scenario["scenario_id"] if worst_scenario else None
        else:
            stable_metrics.append(metric_id)

    # Sort lists for determinism
    metrics_with_large_drops.sort()
    stable_metrics.sort()

    risk_summary = {
        "metrics_with_large_drops": metrics_with_large_drops,
        "stable_metrics": stable_metrics,
        "worst_overall_delta": worst_overall_delta,
        "worst_overall_metric": worst_overall_metric,
        "worst_overall_scenario": worst_overall_scenario,
    }

    return DashboardSummary(
        version="1.0",
        primary_metric=primary_metric,
        surface_type=surface_type,
        metrics=metrics_list,
        risk_summary=risk_summary,
    )


def _get_mean(obj: dict[str, Any] | Any) -> float | None:
    """Extract mean from a MetricSummary-like object."""
    if isinstance(obj, dict):
        v = obj.get("mean")
        if isinstance(v, (int, float)):
            return float(v)
    elif hasattr(obj, "mean"):
        return float(obj.mean)
    return None
