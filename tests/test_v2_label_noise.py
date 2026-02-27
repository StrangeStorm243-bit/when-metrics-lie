"""Tests for generalized label_noise scenario."""
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.scenarios.label_noise import LabelNoiseScenario
from metrics_lie.scenarios.base import ScenarioContext


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_binary_unchanged(rng):
    """Binary behavior is preserved."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    ctx = ScenarioContext(task="binary_classification")
    scenario = LabelNoiseScenario(p=0.3)
    y_out, s_out = scenario.apply(y_true, y_score, rng, ctx)
    assert set(np.unique(y_out)).issubset({0, 1})
    assert np.array_equal(s_out, y_score)


def test_multiclass_flips_to_different_class(rng):
    """Multiclass flips should produce different valid class labels."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_score = np.zeros(10)  # dummy
    ctx = ScenarioContext(task="multiclass_classification")
    scenario = LabelNoiseScenario(p=0.5)
    y_out, _ = scenario.apply(y_true, y_score, rng, ctx)
    assert set(np.unique(y_out)).issubset({0, 1, 2})
    assert not np.array_equal(y_out, y_true)
    changed = y_out != y_true
    assert np.all(y_out[changed] != y_true[changed])


def test_regression_adds_noise(rng):
    """Regression label noise adds Gaussian noise to targets."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_score = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    ctx = ScenarioContext(task="regression")
    scenario = LabelNoiseScenario(p=0.1)
    y_out, s_out = scenario.apply(y_true, y_score, rng, ctx)
    assert not np.allclose(y_out, y_true)
    assert np.array_equal(s_out, y_score)
