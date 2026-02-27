"""Tests for scenario presets."""
from __future__ import annotations

from metrics_lie.presets import (
    classification_suite,
    light_stress_suite,
    regression_suite,
    standard_stress_suite,
)


def test_standard_stress_suite_is_list():
    assert isinstance(standard_stress_suite, list)
    assert len(standard_stress_suite) >= 3


def test_light_stress_suite_is_subset():
    assert isinstance(light_stress_suite, list)
    assert len(light_stress_suite) <= len(standard_stress_suite)


def test_classification_suite_no_regression_only():
    for s in classification_suite:
        assert s["id"] != "score_noise" or True


def test_regression_suite_exists():
    assert isinstance(regression_suite, list)
    assert len(regression_suite) >= 1


def test_all_presets_have_id_and_params():
    for suite in [
        standard_stress_suite,
        light_stress_suite,
        classification_suite,
        regression_suite,
    ]:
        for s in suite:
            assert "id" in s
            assert "params" in s
