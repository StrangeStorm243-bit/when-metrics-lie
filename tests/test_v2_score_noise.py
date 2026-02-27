"""Tests for generalized score_noise scenario."""
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.scenarios.score_noise import ScoreNoiseScenario
from metrics_lie.scenarios.base import ScenarioContext


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_binary_unchanged(rng):
    """Binary probability behavior preserved."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.3, 0.7, 0.9])
    ctx = ScenarioContext(task="binary_classification", surface_type="probability")
    scenario = ScoreNoiseScenario(sigma=0.05)
    _, s_out = scenario.apply(y_true, y_score, rng, ctx)
    assert s_out.ndim == 1
    assert np.all(s_out >= 0.0) and np.all(s_out <= 1.0)


def test_multiclass_2d_preserved(rng):
    """Multiclass probability matrix stays 2D with valid row sums."""
    y_true = np.array([0, 1, 2])
    y_proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]])
    ctx = ScenarioContext(task="multiclass_classification", surface_type="probability")
    scenario = ScoreNoiseScenario(sigma=0.05)
    _, s_out = scenario.apply(y_true, y_proba, rng, ctx)
    assert s_out.ndim == 2
    assert s_out.shape == (3, 3)
    np.testing.assert_allclose(s_out.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(s_out >= 0.0)


def test_regression_no_clip(rng):
    """Regression scores should not be clipped to [0, 1]."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_score = np.array([105.0, 195.0, 310.0])
    ctx = ScenarioContext(task="regression", surface_type="continuous")
    scenario = ScoreNoiseScenario(sigma=5.0)
    _, s_out = scenario.apply(y_true, y_score, rng, ctx)
    # Values should NOT be clipped to [0, 1]
    assert s_out.max() > 1.0 or s_out.min() < 0.0
