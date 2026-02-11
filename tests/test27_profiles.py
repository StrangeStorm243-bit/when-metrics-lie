"""Tests for Phase 2.7.1 Decision Profiles."""

import json

import pytest

from metrics_lie.profiles import (
    BALANCED,
    get_profile,
    get_profile_or_load,
    load_profile_from_dict,
    load_profile_from_json,
)
from metrics_lie.profiles.schema import DecisionProfile


def test_get_profile_balanced():
    """Test that get_profile('balanced') returns a valid profile."""
    profile = get_profile("balanced")
    assert isinstance(profile, DecisionProfile)
    assert profile.name == "balanced"
    assert profile.aggregation["mode"] == "worst_case"
    assert profile.objectives["primary"] == "metric_mean_delta"
    assert "ece_mean_delta" in profile.objectives["secondary"]


def test_risk_averse_exists():
    """Test that risk_averse preset exists and is valid."""
    profile = get_profile("risk_averse")
    assert isinstance(profile, DecisionProfile)
    assert profile.name == "risk_averse"
    assert profile.aggregation["mode"] == "worst_case"
    # Risk averse should have stricter thresholds
    assert (
        profile.thresholds["calibration_regression_ece"]
        < BALANCED.thresholds["calibration_regression_ece"]
    )
    # Lower weight on metric gain
    assert profile.weights["metric_mean_delta"] < BALANCED.weights["metric_mean_delta"]


def test_performance_first_exists():
    """Test that performance_first preset exists and is valid."""
    profile = get_profile("performance_first")
    assert isinstance(profile, DecisionProfile)
    assert profile.name == "performance_first"
    assert profile.aggregation["mode"] == "mean"  # Different from balanced
    # Higher weight on metric
    assert profile.weights["metric_mean_delta"] > BALANCED.weights["metric_mean_delta"]


def test_invalid_percentile_raises_error():
    """Test that invalid percentile profile raises ValueError."""
    # Missing percentile when mode is percentile
    with pytest.raises(ValueError, match="percentile is required"):
        DecisionProfile(
            name="invalid",
            aggregation={"mode": "percentile", "scenario_scope": "all"},
            objectives={"primary": "metric_mean_delta", "secondary": []},
        )

    # Percentile out of range (> 0.5)
    with pytest.raises(ValueError, match="percentile must be in"):
        DecisionProfile(
            name="invalid",
            aggregation={
                "mode": "percentile",
                "percentile": 0.6,
                "scenario_scope": "all",
            },
            objectives={"primary": "metric_mean_delta", "secondary": []},
        )

    # Percentile <= 0
    with pytest.raises(ValueError, match="percentile must be in"):
        DecisionProfile(
            name="invalid",
            aggregation={
                "mode": "percentile",
                "percentile": 0.0,
                "scenario_scope": "all",
            },
            objectives={"primary": "metric_mean_delta", "secondary": []},
        )


def test_subset_without_scenario_list_raises_error():
    """Test that subset scope without scenario_subset raises ValueError."""
    with pytest.raises(ValueError, match="scenario_subset must be non-empty"):
        DecisionProfile(
            name="invalid",
            aggregation={"mode": "worst_case", "scenario_scope": "subset"},
            objectives={"primary": "metric_mean_delta", "secondary": []},
        )

    with pytest.raises(ValueError, match="scenario_subset must be non-empty"):
        DecisionProfile(
            name="invalid",
            aggregation={
                "mode": "worst_case",
                "scenario_scope": "subset",
                "scenario_subset": [],
            },
            objectives={"primary": "metric_mean_delta", "secondary": []},
        )


def test_load_profile_from_dict():
    """Test loading a profile from a dictionary."""
    profile_dict = {
        "name": "test_profile",
        "description": "Test profile",
        "aggregation": {
            "mode": "worst_case",
            "scenario_scope": "all",
        },
        "objectives": {
            "primary": "metric_mean_delta",
            "secondary": ["ece_mean_delta", "subgroup_gap_delta"],
        },
        "thresholds": {
            "calibration_regression_ece": 0.02,
        },
        "weights": {
            "metric_mean_delta": 0.5,
            "ece_mean_delta": 0.3,
            "subgroup_gap_delta": 0.2,
        },
    }

    profile = load_profile_from_dict(profile_dict)
    assert isinstance(profile, DecisionProfile)
    assert profile.name == "test_profile"
    assert profile.aggregation["mode"] == "worst_case"
    # Default thresholds should be set
    assert "calibration_regression_brier" in profile.thresholds


def test_load_profile_from_dict_invalid():
    """Test that invalid dict raises ValueError."""
    invalid_dict = {
        "name": "invalid",
        "aggregation": {"mode": "invalid_mode", "scenario_scope": "all"},
        "objectives": {"primary": "metric_mean_delta", "secondary": []},
    }

    with pytest.raises(ValueError):
        load_profile_from_dict(invalid_dict)


def test_load_profile_from_json(tmp_path):
    """Test loading a profile from a JSON file."""
    profile_dict = {
        "name": "json_profile",
        "aggregation": {
            "mode": "mean",
            "scenario_scope": "all",
        },
        "objectives": {
            "primary": "metric_mean_delta",
            "secondary": ["sensitivity_abs_delta"],
        },
    }

    json_path = tmp_path / "profile.json"
    json_path.write_text(json.dumps(profile_dict), encoding="utf-8")

    profile = load_profile_from_json(str(json_path))
    assert isinstance(profile, DecisionProfile)
    assert profile.name == "json_profile"
    assert profile.aggregation["mode"] == "mean"


def test_get_profile_or_load_preset():
    """Test get_profile_or_load with preset name."""
    profile = get_profile_or_load("balanced")
    assert profile.name == "balanced"

    profile = get_profile_or_load("risk_averse")
    assert profile.name == "risk_averse"


def test_get_profile_or_load_file(tmp_path):
    """Test get_profile_or_load with file path."""
    profile_dict = {
        "name": "file_profile",
        "aggregation": {
            "mode": "worst_case",
            "scenario_scope": "all",
        },
        "objectives": {
            "primary": "metric_mean_delta",
            "secondary": [],
        },
    }

    json_path = tmp_path / "custom_profile.json"
    json_path.write_text(json.dumps(profile_dict), encoding="utf-8")

    profile = get_profile_or_load(str(json_path))
    assert profile.name == "file_profile"


def test_get_profile_or_load_not_found():
    """Test get_profile_or_load with invalid name/path."""
    with pytest.raises(ValueError, match="not found"):
        get_profile_or_load("nonexistent_profile")

    with pytest.raises(ValueError, match="not found"):
        get_profile_or_load("/nonexistent/path.json")


def test_valid_percentile_profile():
    """Test that valid percentile profile works."""
    profile = DecisionProfile(
        name="percentile_test",
        aggregation={
            "mode": "percentile",
            "percentile": 0.05,
            "scenario_scope": "all",
        },
        objectives={"primary": "metric_mean_delta", "secondary": []},
    )
    assert profile.aggregation["mode"] == "percentile"
    assert profile.aggregation["percentile"] == 0.05


def test_valid_subset_profile():
    """Test that valid subset profile works."""
    profile = DecisionProfile(
        name="subset_test",
        aggregation={
            "mode": "worst_case",
            "scenario_scope": "subset",
            "scenario_subset": ["label_noise", "score_noise"],
        },
        objectives={"primary": "metric_mean_delta", "secondary": []},
    )
    assert profile.aggregation["scenario_scope"] == "subset"
    assert len(profile.aggregation["scenario_subset"]) == 2


def test_weights_validation():
    """Test that invalid weight keys raise ValueError."""
    with pytest.raises(ValueError, match="weights contains invalid key"):
        DecisionProfile(
            name="invalid_weights",
            aggregation={"mode": "worst_case", "scenario_scope": "all"},
            objectives={"primary": "metric_mean_delta", "secondary": []},
            weights={"invalid_component": 1.0},
        )


def test_objectives_validation():
    """Test that invalid objectives raise ValueError."""
    # Invalid primary
    with pytest.raises(ValueError, match="primary must be"):
        DecisionProfile(
            name="invalid_primary",
            aggregation={"mode": "worst_case", "scenario_scope": "all"},
            objectives={"primary": "invalid", "secondary": []},
        )

    # Invalid secondary
    with pytest.raises(ValueError, match="secondary contains invalid value"):
        DecisionProfile(
            name="invalid_secondary",
            aggregation={"mode": "worst_case", "scenario_scope": "all"},
            objectives={"primary": "metric_mean_delta", "secondary": ["invalid_obj"]},
        )
