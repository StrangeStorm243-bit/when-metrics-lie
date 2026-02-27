"""Tests for Phase 2.7.2 decision component extraction and aggregation."""

import pytest

from metrics_lie.decision import DecisionComponents, extract_components
from metrics_lie.profiles import BALANCED, PERFORMANCE_FIRST


def _make_fake_compare_report() -> dict:
    """Create a fake compare report with 3 scenarios."""
    return {
        "run_a": "RUN_A",
        "run_b": "RUN_B",
        "metric_name": "auc",
        "baseline_delta": {"mean": 0.01},
        "scenario_deltas": {
            "label_noise": {
                "metric_mean_delta": -0.02,  # Worse
                "ece_mean_delta": 0.01,
                "brier_mean_delta": 0.005,
                "subgroup_gap_delta": 0.02,
                "sensitivity_abs_delta": 0.015,
            },
            "score_noise": {
                "metric_mean_delta": -0.01,  # Slightly worse
                "ece_mean_delta": 0.02,  # Worst ECE
                "brier_mean_delta": 0.01,  # Worst Brier
                "subgroup_gap_delta": None,  # Missing
                "sensitivity_abs_delta": 0.01,
            },
            "class_imbalance": {
                "metric_mean_delta": 0.005,  # Best (slight improvement)
                "ece_mean_delta": 0.005,
                "brier_mean_delta": 0.002,
                "subgroup_gap_delta": 0.03,  # Worst subgroup gap
                "sensitivity_abs_delta": 0.02,  # Worst sensitivity
            },
        },
        "metric_gaming_delta": None,
        "regressions": {},
        "risk_flags": [],
        "decision": {},
    }


def test_worst_case_all_scenarios():
    """Test worst_case aggregation with all scenarios and per-component directions."""
    report = _make_fake_compare_report()
    profile = BALANCED  # Uses worst_case mode

    result = extract_components(report, profile)

    assert isinstance(result, DecisionComponents)
    assert result.profile_name == "balanced"
    assert result.aggregation["mode"] == "worst_case"
    assert len(result.meta["used_scenarios"]) == 3

    # metric_mean_delta: should be min (worst) = -0.02
    assert result.components["metric_mean_delta"] == pytest.approx(-0.02)

    # ece_mean_delta: should be max (worst) = 0.02
    assert result.components["ece_mean_delta"] == pytest.approx(0.02)

    # brier_mean_delta: should be max (worst) = 0.01
    assert result.components["brier_mean_delta"] == pytest.approx(0.01)

    # subgroup_gap_delta: should be max (worst) = 0.03 (nulls ignored)
    assert result.components["subgroup_gap_delta"] == pytest.approx(0.03)

    # sensitivity_abs_delta: should be max (worst) = 0.02
    assert result.components["sensitivity_abs_delta"] == pytest.approx(0.02)

    # metric_inflation_delta: should be None (not in report)
    assert result.components["metric_inflation_delta"] is None


def test_mean_aggregation():
    """Test mean aggregation mode."""
    report = _make_fake_compare_report()
    profile = PERFORMANCE_FIRST  # Uses mean mode

    result = extract_components(report, profile)

    assert result.aggregation["mode"] == "mean"

    # metric_mean_delta: mean of [-0.02, -0.01, 0.005] = -0.00833...
    assert result.components["metric_mean_delta"] == pytest.approx(-0.00833, abs=0.0001)

    # ece_mean_delta: mean of [0.01, 0.02, 0.005] = 0.01167...
    assert result.components["ece_mean_delta"] == pytest.approx(0.01167, abs=0.0001)

    # subgroup_gap_delta: mean of [0.02, None, 0.03] = mean of [0.02, 0.03] = 0.025
    assert result.components["subgroup_gap_delta"] == pytest.approx(0.025)


def test_subset_scenarios():
    """Test subset mode uses only specified scenarios."""
    report = _make_fake_compare_report()

    # Create a profile with subset
    from metrics_lie.profiles.schema import DecisionProfile

    profile = DecisionProfile(
        name="subset_test",
        aggregation={
            "mode": "worst_case",
            "scenario_scope": "subset",
            "scenario_subset": ["label_noise", "score_noise"],
        },
        objectives={"primary": "metric_mean_delta", "secondary": []},
    )

    result = extract_components(report, profile)

    assert len(result.meta["used_scenarios"]) == 2
    assert set(result.meta["used_scenarios"]) == {"label_noise", "score_noise"}

    # metric_mean_delta: min of [-0.02, -0.01] = -0.02
    assert result.components["metric_mean_delta"] == pytest.approx(-0.02)

    # ece_mean_delta: max of [0.01, 0.02] = 0.02
    assert result.components["ece_mean_delta"] == pytest.approx(0.02)


def test_subset_none_available_raises_error():
    """Test that subset with no available scenarios raises ValueError."""
    report = _make_fake_compare_report()

    from metrics_lie.profiles.schema import DecisionProfile

    profile = DecisionProfile(
        name="invalid_subset",
        aggregation={
            "mode": "worst_case",
            "scenario_scope": "subset",
            "scenario_subset": ["nonexistent_scenario"],
        },
        objectives={"primary": "metric_mean_delta", "secondary": []},
    )

    with pytest.raises(ValueError, match="none are available"):
        extract_components(report, profile)


def test_metric_gaming_delta_present():
    """Test that metric_inflation_delta is extracted when metric_gaming_delta is present."""
    report = _make_fake_compare_report()
    report["metric_gaming_delta"] = {
        "delta_baseline": 0.01,
        "delta_optimized": 0.02,
        "delta_inflation": 0.015,
        "downstream": {},
    }

    profile = BALANCED
    result = extract_components(report, profile)

    assert result.components["metric_inflation_delta"] == pytest.approx(0.015)



def test_percentile_aggregation():
    """Test percentile aggregation mode."""
    report = _make_fake_compare_report()

    from metrics_lie.profiles.schema import DecisionProfile

    profile = DecisionProfile(
        name="percentile_test",
        aggregation={
            "mode": "percentile",
            "percentile": 0.05,
            "scenario_scope": "all",
        },
        objectives={"primary": "metric_mean_delta", "secondary": []},
    )

    result = extract_components(report, profile)

    assert result.aggregation["mode"] == "percentile"
    assert result.meta["percentile"] == 0.05

    # metric_mean_delta: lower percentile (0.05) of [-0.02, -0.01, 0.005]
    # Should be close to -0.02 (worst case)
    assert result.components["metric_mean_delta"] < -0.01

    # ece_mean_delta: upper percentile (0.95) of [0.01, 0.02, 0.005]
    # Should be close to 0.02 (worst case)
    assert result.components["ece_mean_delta"] > 0.01


def test_percentile_missing_raises_error():
    """Test that percentile mode without percentile value raises error (caught by profile validation)."""

    from metrics_lie.profiles.schema import DecisionProfile
    from pydantic import ValidationError

    # Profile validation catches missing percentile
    with pytest.raises(ValidationError, match="percentile is required"):
        DecisionProfile(
            name="invalid_percentile",
            aggregation={
                "mode": "percentile",
                "scenario_scope": "all",
                # Missing percentile
            },
            objectives={"primary": "metric_mean_delta", "secondary": []},
        )


def test_empty_scenarios_raises_error():
    """Test that empty scenario_deltas raises ValueError."""
    report = {
        "scenario_deltas": {},
        "baseline_delta": {},
    }

    profile = BALANCED

    with pytest.raises(ValueError, match="No scenarios available"):
        extract_components(report, profile)


def test_meta_includes_per_scenario_data():
    """Test that meta includes per_scenario breakdown."""
    report = _make_fake_compare_report()
    profile = BALANCED

    result = extract_components(report, profile)

    assert "per_scenario" in result.meta
    assert "label_noise" in result.meta["per_scenario"]
    assert "metric_mean_delta" in result.meta["per_scenario"]["label_noise"]
    assert result.meta["per_scenario"]["label_noise"][
        "metric_mean_delta"
    ] == pytest.approx(-0.02)


def test_scenario_scope_echo():
    """Test that scenario_scope echo includes correct information."""
    report = _make_fake_compare_report()
    profile = BALANCED

    result = extract_components(report, profile)

    assert result.scenario_scope["scope"] == "all"
    assert len(result.scenario_scope["scenario_ids"]) == 3


def test_scenario_scope_echo_subset():
    """Test scenario_scope echo for subset mode."""
    report = _make_fake_compare_report()

    from metrics_lie.profiles.schema import DecisionProfile

    profile = DecisionProfile(
        name="subset_echo_test",
        aggregation={
            "mode": "worst_case",
            "scenario_scope": "subset",
            "scenario_subset": ["label_noise", "class_imbalance"],
        },
        objectives={"primary": "metric_mean_delta", "secondary": []},
    )

    result = extract_components(report, profile)

    assert result.scenario_scope["scope"] == "subset"
    assert "requested_subset" in result.scenario_scope
    assert set(result.scenario_scope["requested_subset"]) == {
        "label_noise",
        "class_imbalance",
    }


def test_all_components_present():
    """Test that all expected components are present in output."""
    report = _make_fake_compare_report()
    profile = BALANCED

    result = extract_components(report, profile)

    expected_components = [
        "metric_mean_delta",
        "ece_mean_delta",
        "brier_mean_delta",
        "subgroup_gap_delta",
        "sensitivity_abs_delta",
        "metric_inflation_delta",
    ]

    for comp in expected_components:
        assert comp in result.components
