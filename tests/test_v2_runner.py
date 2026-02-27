"""Tests for task-type-aware runner diagnostics."""
from __future__ import annotations

import numpy as np

import metrics_lie.scenarios.label_noise  # noqa: F401

from metrics_lie.metrics.core import METRICS
from metrics_lie.runner import RunConfig, run_scenarios
from metrics_lie.scenarios.base import ScenarioContext


def test_regression_runner_no_brier_ece():
    """Regression runner should not compute brier/ece diagnostics."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_score = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    ctx = ScenarioContext(task="regression", surface_type="continuous")
    results = run_scenarios(
        y_true=y_true,
        y_score=y_score,
        metric_name="mae",
        metric_fn=METRICS["mae"],
        scenario_specs=[{"id": "label_noise", "params": {"p": 0.1}}],
        cfg=RunConfig(n_trials=3, seed=42),
        ctx=ctx,
    )
    assert len(results) == 1
    assert "brier" not in results[0].diagnostics
    assert "ece" not in results[0].diagnostics


def test_multiclass_runner_no_gaming():
    """Multiclass runner should not compute accuracy gaming diagnostics."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    ctx = ScenarioContext(task="multiclass_classification", surface_type="label")
    results = run_scenarios(
        y_true=y_true,
        y_score=y_pred,
        metric_name="macro_f1",
        metric_fn=METRICS["macro_f1"],
        scenario_specs=[{"id": "label_noise", "params": {"p": 0.1}}],
        cfg=RunConfig(n_trials=3, seed=42),
        ctx=ctx,
    )
    assert len(results) == 1
    assert "metric_inflation" not in results[0].diagnostics


def test_scenario_context_has_n_classes():
    ctx = ScenarioContext(task="multiclass_classification", n_classes=3)
    assert ctx.n_classes == 3

    ctx_default = ScenarioContext()
    assert ctx_default.n_classes is None
