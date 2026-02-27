"""Tests for Phase 2.7.3 decision scorecard and weighted scoring."""

import pytest

from metrics_lie.decision import extract_components, build_scorecard
from metrics_lie.profiles import BALANCED, PERFORMANCE_FIRST, RISK_AVERSE


def _make_fake_compare_report() -> dict:
    """Create a fake compare report with 2 scenarios with deltas."""
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
        },
        "metric_gaming_delta": None,
        "regressions": {},
        "risk_flags": [],
        "decision": {},
    }


def test_build_scorecard_returns_total_score_and_contributions():
    """Test that build_scorecard returns total_score and contributions."""
    report = _make_fake_compare_report()
    profile = BALANCED

    comps = extract_components(report, profile)
    scorecard = build_scorecard(comps, profile)

    assert isinstance(scorecard.total_score, float)
    assert isinstance(scorecard.contributions, dict)
    assert len(scorecard.contributions) > 0

    # Verify total_score equals sum of contributions
    calculated_total = sum(scorecard.contributions.values())
    assert scorecard.total_score == pytest.approx(calculated_total)


def test_ignored_components_includes_none_values():
    """Test that ignored_components includes components that are None."""
    report = _make_fake_compare_report()
    # Use PERFORMANCE_FIRST which has metric_inflation_delta in weights
    profile = PERFORMANCE_FIRST

    comps = extract_components(report, profile)
    scorecard = build_scorecard(comps, profile)

    # metric_inflation_delta should be None (not in report)
    assert comps.components["metric_inflation_delta"] is None

    # Check that it's in ignored_components
    ignored_names = [ic["component"] for ic in scorecard.ignored_components]
    assert "metric_inflation_delta" in ignored_names

    # Find the specific entry
    inflation_ignored = next(
        ic
        for ic in scorecard.ignored_components
        if ic["component"] == "metric_inflation_delta"
    )
    assert inflation_ignored["reason"] == "value_is_none"


def test_top_contributors_length_leq_3():
    """Test that top_contributors length <= 3."""
    report = _make_fake_compare_report()
    profile = BALANCED

    comps = extract_components(report, profile)
    scorecard = build_scorecard(comps, profile)

    top_contributors = scorecard.meta.get("top_contributors", {})
    assert len(top_contributors) <= 3



def test_performance_first_different_from_balanced():
    """Test that performance_first profile produces different scores."""
    report = _make_fake_compare_report()

    profile_perf = PERFORMANCE_FIRST
    comps_perf = extract_components(report, profile_perf)
    scorecard_perf = build_scorecard(comps_perf, profile_perf)

    profile_balanced = BALANCED
    comps_balanced = extract_components(report, profile_balanced)
    scorecard_balanced = build_scorecard(comps_balanced, profile_balanced)

    # Performance_first uses mean aggregation vs worst_case, so components will differ
    # Performance_first also has higher weight on metric_mean_delta (0.6 vs 0.4)
    assert scorecard_perf.total_score != scorecard_balanced.total_score


def test_meta_includes_weights_and_component_values():
    """Test that meta includes weights and component values."""
    report = _make_fake_compare_report()
    profile = BALANCED

    comps = extract_components(report, profile)
    scorecard = build_scorecard(comps, profile)

    assert "weights" in scorecard.meta
    assert scorecard.meta["weights"] == profile.weights

    assert "component_values" in scorecard.meta
    # component_values should only include non-None values
    for comp_name, value in scorecard.meta["component_values"].items():
        assert value is not None
        assert comp_name in comps.components


def test_meta_includes_aggregation_info():
    """Test that meta includes aggregation and scenario_scope info."""
    report = _make_fake_compare_report()
    profile = BALANCED

    comps = extract_components(report, profile)
    scorecard = build_scorecard(comps, profile)

    assert "aggregation" in scorecard.meta
    assert scorecard.meta["aggregation"] == comps.aggregation

    assert "scenario_scope" in scorecard.meta
    assert scorecard.meta["scenario_scope"] == comps.scenario_scope

    assert "used_scenarios" in scorecard.meta
    assert len(scorecard.meta["used_scenarios"]) == 2


def test_used_components_list():
    """Test that used_components list contains only components that were scored."""
    report = _make_fake_compare_report()
    profile = BALANCED

    comps = extract_components(report, profile)
    scorecard = build_scorecard(comps, profile)

    # All used_components should have contributions
    for comp_name in scorecard.used_components:
        assert comp_name in scorecard.contributions
        assert comp_name in profile.weights
        assert comps.components.get(comp_name) is not None


def test_risk_averse_penalizes_subgroup_gap_more():
    """Test that risk_averse profile penalizes subgroup_gap_delta more than balanced."""
    report = _make_fake_compare_report()

    profile_balanced = BALANCED
    comps_balanced = extract_components(report, profile_balanced)
    scorecard_balanced = build_scorecard(comps_balanced, profile_balanced)

    profile_risk = RISK_AVERSE
    comps_risk = extract_components(report, profile_risk)
    scorecard_risk = build_scorecard(comps_risk, profile_risk)

    # Both should have subgroup_gap_delta (it's not None in the report)
    if "subgroup_gap_delta" in scorecard_balanced.contributions:
        balanced_weight = profile_balanced.weights["subgroup_gap_delta"]  # 0.2
        risk_weight = profile_risk.weights["subgroup_gap_delta"]  # 0.25

        # Risk_averse has higher weight
        assert risk_weight > balanced_weight

        # Since subgroup_gap_delta is positive (worse), the contribution should be more negative
        # Actually wait, let me check: subgroup_gap_delta = 0.02 (positive means worse)
        # Weight * value: balanced = 0.2 * 0.02 = 0.004, risk = 0.25 * 0.02 = 0.005
        # Both are positive contributions, but risk is larger
        balanced_contrib = scorecard_balanced.contributions["subgroup_gap_delta"]
        risk_contrib = scorecard_risk.contributions["subgroup_gap_delta"]
        assert abs(risk_contrib) > abs(balanced_contrib)
