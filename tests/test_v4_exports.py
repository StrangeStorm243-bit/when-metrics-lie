"""Tests for public API exports."""
from __future__ import annotations


def test_evaluate_importable():
    from metrics_lie import evaluate

    assert callable(evaluate)


def test_compare_importable():
    # Note: metrics_lie.compare is also a subpackage, so after any import of
    # that subpackage Python will shadow the function attribute. We verify the
    # function is accessible via the sdk module which is the canonical path.
    from metrics_lie.sdk import compare

    assert callable(compare)


def test_score_importable():
    from metrics_lie import score

    assert callable(score)


def test_evaluate_file_importable():
    from metrics_lie import evaluate_file

    assert callable(evaluate_file)


def test_result_bundle_importable():
    from metrics_lie import ResultBundle

    assert ResultBundle is not None


def test_experiment_spec_importable():
    from metrics_lie import ExperimentSpec

    assert ExperimentSpec is not None


def test_presets_importable():
    from metrics_lie import presets

    assert hasattr(presets, "standard_stress_suite")


def test_list_metrics_importable():
    from metrics_lie import list_metrics

    assert callable(list_metrics)


def test_list_scenarios_importable():
    from metrics_lie import list_scenarios

    assert callable(list_scenarios)
