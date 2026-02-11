from __future__ import annotations

from typing import Any

import numpy as np

from metrics_lie.profiles.schema import DecisionProfile

from .components import DecisionComponents


def _aggregate_mean(values: list[float]) -> float:
    """Compute mean of values."""
    return float(np.mean(values)) if values else 0.0


def _aggregate_percentile(
    values: list[float], percentile: float, use_upper: bool = False
) -> float:
    """
    Compute percentile of values.

    Args:
        values: List of numeric values
        percentile: Percentile value (0-1)
        use_upper: If True, use (1 - percentile) for upper tail
    """
    if not values:
        return 0.0
    p = (1.0 - percentile) if use_upper else percentile
    return float(np.percentile(values, p * 100))


def _aggregate_worst_case(values: list[float], use_min: bool = False) -> float:
    """
    Compute worst-case (min or max) of values.

    Args:
        values: List of numeric values
        use_min: If True, return min; else return max
    """
    if not values:
        return 0.0
    return float(min(values)) if use_min else float(max(values))


def extract_components(
    compare_report: dict[str, Any], profile: DecisionProfile
) -> DecisionComponents:
    """
    Extract and aggregate decision components from a compare report
    according to the DecisionProfile's aggregation rules.

    Args:
        compare_report: Phase 2.3 compare report dict with scenario_deltas
        profile: DecisionProfile defining aggregation strategy

    Returns:
        DecisionComponents with aggregated values

    Raises:
        ValueError: If scenario subset is requested but none available
    """
    scenario_deltas = compare_report.get("scenario_deltas", {})

    # A) Determine scenario IDs to use
    agg_config = profile.aggregation
    scenario_scope = agg_config.get("scenario_scope", "all")

    if scenario_scope == "all":
        scenario_ids = sorted(scenario_deltas.keys())
    elif scenario_scope == "subset":
        requested = set(agg_config.get("scenario_subset", []))
        available = set(scenario_deltas.keys())
        scenario_ids = sorted(requested & available)
        if not scenario_ids:
            raise ValueError(
                f"Profile requests scenario subset {list(requested)}, "
                f"but none are available in compare report. Available: {list(available)}"
            )
    else:
        raise ValueError(f"Unknown scenario_scope: {scenario_scope}")

    if not scenario_ids:
        raise ValueError("No scenarios available in compare report")

    # B) Collect per-scenario values for each component
    component_lists: dict[str, list[float]] = {
        "metric_mean_delta": [],
        "ece_mean_delta": [],
        "brier_mean_delta": [],
        "subgroup_gap_delta": [],
        "sensitivity_abs_delta": [],
    }

    per_scenario: dict[str, dict[str, float | None]] = {}

    for sid in scenario_ids:
        delta = scenario_deltas[sid]
        per_scenario[sid] = {}

        # Extract each component
        for comp_name in component_lists.keys():
            value = delta.get(comp_name)
            if value is not None and isinstance(value, (int, float)):
                component_lists[comp_name].append(float(value))
                per_scenario[sid][comp_name] = float(value)
            else:
                per_scenario[sid][comp_name] = None

    # C) Aggregate each list according to profile.aggregation.mode
    mode = agg_config.get("mode", "worst_case")
    percentile = agg_config.get("percentile")

    aggregated: dict[str, float | None] = {}

    # Define aggregation direction per component
    # metric_mean_delta: lower is worse (use min for worst_case, lower percentile)
    # Others: higher is worse (use max for worst_case, upper percentile)
    for comp_name, values in component_lists.items():
        if not values:
            aggregated[comp_name] = None
            continue

        if mode == "mean":
            aggregated[comp_name] = _aggregate_mean(values)
        elif mode == "worst_case":
            use_min = comp_name == "metric_mean_delta"
            aggregated[comp_name] = _aggregate_worst_case(values, use_min=use_min)
        elif mode == "percentile":
            if percentile is None:
                raise ValueError(
                    "aggregation.percentile is required when mode='percentile'"
                )
            use_upper = comp_name != "metric_mean_delta"
            aggregated[comp_name] = _aggregate_percentile(
                values, percentile, use_upper=use_upper
            )
        else:
            raise ValueError(f"Unknown aggregation mode: {mode}")

    # D) Handle metric_inflation_delta (optional)
    metric_gaming_delta = compare_report.get("metric_gaming_delta")
    if isinstance(metric_gaming_delta, dict):
        delta_inflation = metric_gaming_delta.get("delta_inflation")
        if delta_inflation is not None and isinstance(delta_inflation, (int, float)):
            aggregated["metric_inflation_delta"] = float(delta_inflation)
        else:
            aggregated["metric_inflation_delta"] = None
    else:
        aggregated["metric_inflation_delta"] = None

    # E) Build meta
    meta: dict[str, Any] = {
        "used_scenarios": scenario_ids,
        "per_scenario": per_scenario,
        "aggregation_mode": mode,
    }
    if mode == "percentile" and percentile is not None:
        meta["percentile"] = percentile

    # Build scenario_scope echo
    scenario_scope_echo: dict[str, Any] = {
        "scope": scenario_scope,
        "scenario_ids": scenario_ids,
    }
    if scenario_scope == "subset":
        scenario_scope_echo["requested_subset"] = agg_config.get("scenario_subset", [])

    return DecisionComponents(
        profile_name=profile.name,
        aggregation=agg_config.copy(),
        scenario_scope=scenario_scope_echo,
        components=aggregated,
        meta=meta,
    )
