"""Tests for public API exports."""
from __future__ import annotations


def test_evaluate_importable():
    from metrics_lie import evaluate

    assert callable(evaluate)


def test_compare_importable():
    from metrics_lie import compare

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
