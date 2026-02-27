"""Tests for scenario-task-type compatibility filtering."""
from __future__ import annotations

from metrics_lie.surface_compat import SCENARIO_TASK_COMPAT, filter_compatible_scenarios_by_task


def test_compat_table_exists():
    assert isinstance(SCENARIO_TASK_COMPAT, dict)


def test_regression_excludes_threshold_gaming():
    assert "threshold_gaming" not in SCENARIO_TASK_COMPAT["regression"]


def test_regression_excludes_class_imbalance():
    assert "class_imbalance" not in SCENARIO_TASK_COMPAT["regression"]


def test_regression_includes_label_noise():
    assert "label_noise" in SCENARIO_TASK_COMPAT["regression"]


def test_regression_includes_score_noise():
    assert "score_noise" in SCENARIO_TASK_COMPAT["regression"]


def test_binary_includes_all_four():
    compat = SCENARIO_TASK_COMPAT["binary_classification"]
    assert {"label_noise", "score_noise", "class_imbalance", "threshold_gaming"} == compat


def test_multiclass_excludes_threshold_gaming():
    assert "threshold_gaming" not in SCENARIO_TASK_COMPAT["multiclass_classification"]


def test_filter_function():
    class FakeScenario:
        def __init__(self, sid):
            self.id = sid
    scenarios = [FakeScenario("label_noise"), FakeScenario("threshold_gaming"),
                 FakeScenario("score_noise")]
    kept, skipped = filter_compatible_scenarios_by_task(scenarios, "regression")
    assert [s.id for s in kept] == ["label_noise", "score_noise"]
    assert skipped == ["threshold_gaming"]
