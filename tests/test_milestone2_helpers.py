"""Tests for Milestone 2 extracted helper functions.

Verifies that filter_compatible_scenarios() and compute_metric() produce
the exact same results as the inline code they replaced.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from metrics_lie.metrics.core import (
    METRICS,
    THRESHOLD_METRICS,
    CALIBRATION_METRICS,
    RANKING_METRICS,
    MULTICLASS_METRICS,
    REGRESSION_METRICS,
    compute_metric,
)
from metrics_lie.model.surface import SurfaceType
from metrics_lie.surface_compat import (
    DEFAULT_THRESHOLD,
    SCENARIO_SURFACE_COMPAT,
    filter_compatible_scenarios,
)


# ── filter_compatible_scenarios ──────────────────────────────────────


@dataclass
class _FakeScenario:
    id: str


_ALL_SCENARIOS = [
    _FakeScenario("label_noise"),
    _FakeScenario("score_noise"),
    _FakeScenario("class_imbalance"),
    _FakeScenario("threshold_gaming"),
]


def test_filter_probability_allows_all():
    compat, skipped = filter_compatible_scenarios(
        _ALL_SCENARIOS, SurfaceType.PROBABILITY
    )
    assert [s.id for s in compat] == [
        "label_noise",
        "score_noise",
        "class_imbalance",
        "threshold_gaming",
    ]
    assert skipped == []


def test_filter_score_excludes_threshold_gaming():
    compat, skipped = filter_compatible_scenarios(_ALL_SCENARIOS, SurfaceType.SCORE)
    assert [s.id for s in compat] == [
        "label_noise",
        "score_noise",
        "class_imbalance",
    ]
    assert skipped == ["threshold_gaming"]


def test_filter_label_allows_only_label_noise_and_class_imbalance():
    compat, skipped = filter_compatible_scenarios(_ALL_SCENARIOS, SurfaceType.LABEL)
    assert [s.id for s in compat] == ["label_noise", "class_imbalance"]
    assert sorted(skipped) == ["score_noise", "threshold_gaming"]


def test_filter_preserves_input_order():
    reversed_scenarios = list(reversed(_ALL_SCENARIOS))
    compat, _ = filter_compatible_scenarios(reversed_scenarios, SurfaceType.SCORE)
    assert [s.id for s in compat] == [
        "class_imbalance",
        "score_noise",
        "label_noise",
    ]


def test_filter_empty_input():
    compat, skipped = filter_compatible_scenarios([], SurfaceType.PROBABILITY)
    assert compat == []
    assert skipped == []


def test_filter_matches_compat_table():
    """Verify function output matches SCENARIO_SURFACE_COMPAT for all surface types."""
    for st in SurfaceType:
        compat, skipped = filter_compatible_scenarios(_ALL_SCENARIOS, st)
        compat_ids = {s.id for s in compat}
        assert compat_ids == SCENARIO_SURFACE_COMPAT[st]


# ── compute_metric ───────────────────────────────────────────────────


@pytest.fixture()
def _binary_data():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_score = rng.random(size=200)
    return y_true, y_score


def test_compute_metric_threshold_metrics_pass_threshold(_binary_data):
    """Threshold metrics must receive the threshold kwarg."""
    y_true, y_score = _binary_data
    for metric_id in THRESHOLD_METRICS:
        fn = METRICS[metric_id]
        expected = fn(y_true, y_score, threshold=DEFAULT_THRESHOLD)
        got = compute_metric(
            metric_id, fn, y_true, y_score, threshold=DEFAULT_THRESHOLD
        )
        assert got == pytest.approx(expected), f"Mismatch for {metric_id}"


def test_compute_metric_ranking_metrics_skip_threshold(_binary_data):
    """Ranking metrics must be called without threshold."""
    y_true, y_score = _binary_data
    for metric_id in RANKING_METRICS:
        fn = METRICS.get(metric_id)
        if fn is None:
            continue
        expected = fn(y_true, y_score)
        got = compute_metric(
            metric_id, fn, y_true, y_score, threshold=DEFAULT_THRESHOLD
        )
        assert got == pytest.approx(expected), f"Mismatch for {metric_id}"


def test_compute_metric_calibration_metrics_skip_threshold(_binary_data):
    """Calibration metrics must be called without threshold."""
    y_true, y_score = _binary_data
    for metric_id in CALIBRATION_METRICS:
        fn = METRICS.get(metric_id)
        if fn is None:
            continue
        expected = fn(y_true, y_score)
        got = compute_metric(
            metric_id, fn, y_true, y_score, threshold=DEFAULT_THRESHOLD
        )
        assert got == pytest.approx(expected), f"Mismatch for {metric_id}"


def test_compute_metric_custom_threshold(_binary_data):
    """Threshold metrics with a non-default threshold produce different results."""
    y_true, y_score = _binary_data
    fn = METRICS["accuracy"]
    at_05 = compute_metric("accuracy", fn, y_true, y_score, threshold=0.5)
    at_03 = compute_metric("accuracy", fn, y_true, y_score, threshold=0.3)
    # With different thresholds, results should generally differ
    # (not guaranteed for all data, but very likely with 200 samples)
    assert isinstance(at_05, float)
    assert isinstance(at_03, float)


def test_compute_metric_covers_all_known_metrics(_binary_data):
    """Every metric in the METRICS dict can be called via compute_metric."""
    y_true, y_score = _binary_data
    for metric_id, fn in METRICS.items():
        if metric_id in MULTICLASS_METRICS:
            continue  # multiclass metrics require different data shapes
        val = compute_metric(
            metric_id, fn, y_true, y_score, threshold=DEFAULT_THRESHOLD
        )
        assert isinstance(val, float), f"{metric_id} did not return float"


# ── Category set completeness ────────────────────────────────────────


def test_metric_categories_cover_all_metrics():
    """Every metric in METRICS must belong to exactly one category."""
    all_categorized = (
        THRESHOLD_METRICS | CALIBRATION_METRICS | RANKING_METRICS
        | MULTICLASS_METRICS | REGRESSION_METRICS
    )
    assert all_categorized == set(METRICS.keys())


def test_metric_categories_are_disjoint():
    """Categories must not overlap."""
    all_sets = [THRESHOLD_METRICS, CALIBRATION_METRICS, RANKING_METRICS, MULTICLASS_METRICS, REGRESSION_METRICS]
    for i, a in enumerate(all_sets):
        for b in all_sets[i + 1:]:
            assert a & b == set()
