"""Tests for generalized class_imbalance scenario."""
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.scenarios.class_imbalance import ClassImbalanceScenario
from metrics_lie.scenarios.base import ScenarioContext


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_binary_unchanged(rng):
    """Binary behavior is preserved."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.15, 0.25, 0.8, 0.9, 0.7, 0.85, 0.75])
    ctx = ScenarioContext(task="binary_classification")
    scenario = ClassImbalanceScenario(target_pos_rate=0.2)
    y_out, s_out = scenario.apply(y_true, y_score, rng, ctx)
    assert len(y_out) < len(y_true)
    assert set(np.unique(y_out)).issubset({0, 1})


def test_multiclass_reduces_majority_class(rng):
    """Multiclass: subsample the largest class to shift distribution."""
    # Class 0 is the majority with 50 samples, classes 1 and 2 have 10 each.
    y_true = np.array([0] * 50 + [1] * 10 + [2] * 10)
    y_score = np.arange(70, dtype=float)
    ctx = ScenarioContext(task="multiclass_classification")
    scenario = ClassImbalanceScenario(target_pos_rate=0.1)
    y_out, s_out = scenario.apply(y_true, y_score, rng, ctx)
    # The largest class (0) should be subsampled, reducing overall size.
    assert len(y_out) < len(y_true)
    # Class 0 count must be reduced; classes 1 and 2 remain untouched.
    class_0_out = int((y_out == 0).sum())
    class_1_out = int((y_out == 1).sum())
    class_2_out = int((y_out == 2).sum())
    assert class_0_out < 50, "Largest class (0) must be subsampled"
    assert class_1_out == 10, "Non-largest classes must remain untouched"
    assert class_2_out == 10, "Non-largest classes must remain untouched"
    assert set(np.unique(y_out)).issubset({0, 1, 2})


def test_multiclass_single_class_noop(rng):
    """Multiclass with a single class returns data unchanged."""
    y_true = np.array([2, 2, 2, 2, 2])
    y_score = np.arange(5, dtype=float)
    ctx = ScenarioContext(task="multiclass_classification")
    scenario = ClassImbalanceScenario(target_pos_rate=0.2)
    y_out, s_out = scenario.apply(y_true, y_score, rng, ctx)
    assert np.array_equal(y_out, y_true)
    assert np.array_equal(s_out, y_score)


def test_regression_returns_unchanged(rng):
    """Regression: class_imbalance is a no-op, returns data unchanged."""
    y_true = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    y_score = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    ctx = ScenarioContext(task="regression")
    scenario = ClassImbalanceScenario(target_pos_rate=0.2)
    y_out, s_out = scenario.apply(y_true, y_score, rng, ctx)
    assert np.array_equal(y_out, y_true)
    assert np.array_equal(s_out, y_score)


def test_regression_returns_unchanged_large(rng):
    """Regression with many samples is still a no-op."""
    y_true = np.linspace(0, 10, 100)
    y_score = y_true + rng.normal(0, 0.1, size=100)
    ctx = ScenarioContext(task="regression")
    scenario = ClassImbalanceScenario(target_pos_rate=0.2)
    y_out, s_out = scenario.apply(y_true, y_score, rng, ctx)
    assert np.array_equal(y_out, y_true)
    assert np.array_equal(s_out, y_score)
