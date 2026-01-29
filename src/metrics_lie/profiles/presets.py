from __future__ import annotations

from .schema import DecisionProfile

# Phase 2.7.1: Built-in decision profile presets

BALANCED = DecisionProfile(
    name="balanced",
    description="Balanced evaluation emphasizing worst-case scenarios with calibration and subgroup considerations",
    aggregation={
        "mode": "worst_case",
        "scenario_scope": "all",
    },
    objectives={
        "primary": "metric_mean_delta",
        "secondary": [
            "ece_mean_delta",
            "brier_mean_delta",
            "subgroup_gap_delta",
            "sensitivity_abs_delta",
        ],
    },
    thresholds={
        "calibration_regression_ece": 0.02,
        "calibration_regression_brier": 0.02,
        "subgroup_gap_regression": 0.03,
        "metric_regression": -0.01,
    },
    weights={
        "metric_mean_delta": 0.4,
        "ece_mean_delta": 0.2,
        "brier_mean_delta": 0.15,
        "subgroup_gap_delta": 0.2,
        "sensitivity_abs_delta": 0.05,
    },
)

RISK_AVERSE = DecisionProfile(
    name="risk_averse",
    description="Risk-averse evaluation with heavy penalties for calibration and subgroup regressions",
    aggregation={
        "mode": "worst_case",
        "scenario_scope": "all",
    },
    objectives={
        "primary": "metric_mean_delta",
        "secondary": [
            "ece_mean_delta",
            "brier_mean_delta",
            "subgroup_gap_delta",
            "sensitivity_abs_delta",
        ],
    },
    thresholds={
        "calibration_regression_ece": 0.015,  # Stricter
        "calibration_regression_brier": 0.015,  # Stricter
        "subgroup_gap_regression": 0.02,  # Stricter
        "metric_regression": -0.005,  # Stricter
    },
    weights={
        "metric_mean_delta": 0.25,  # Lower weight on metric gain
        "ece_mean_delta": 0.25,  # Higher weight on calibration
        "brier_mean_delta": 0.2,
        "subgroup_gap_delta": 0.25,  # Higher weight on subgroup
        "sensitivity_abs_delta": 0.05,
    },
)

PERFORMANCE_FIRST = DecisionProfile(
    name="performance_first",
    description="Performance-first evaluation emphasizing mean metric improvement with relaxed calibration constraints",
    aggregation={
        "mode": "mean",
        "scenario_scope": "all",
    },
    objectives={
        "primary": "metric_mean_delta",
        "secondary": [
            "metric_inflation_delta",
            "sensitivity_abs_delta",
        ],
    },
    thresholds={
        "calibration_regression_ece": 0.03,  # More lenient
        "calibration_regression_brier": 0.03,  # More lenient
        "subgroup_gap_regression": 0.05,  # More lenient
        "metric_regression": -0.02,  # More lenient
    },
    weights={
        "metric_mean_delta": 0.6,  # Higher weight on metric
        "ece_mean_delta": 0.1,
        "brier_mean_delta": 0.1,
        "subgroup_gap_delta": 0.1,
        "metric_inflation_delta": 0.1,
    },
)

PROFILES: dict[str, DecisionProfile] = {
    "balanced": BALANCED,
    "risk_averse": RISK_AVERSE,
    "performance_first": PERFORMANCE_FIRST,
}


def get_profile(name: str) -> DecisionProfile:
    """
    Get a built-in profile by name.
    
    Raises:
        ValueError: If profile name is not found.
    """
    if name not in PROFILES:
        raise ValueError(
            f"Profile '{name}' not found. Available profiles: {sorted(PROFILES.keys())}"
        )
    return PROFILES[name]

