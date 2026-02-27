"""Tests for notebook display functions."""
from __future__ import annotations

from metrics_lie.display import format_summary, format_comparison


def test_format_summary_returns_string():
    from metrics_lie.schema import MetricSummary, ResultBundle

    bundle = ResultBundle(
        run_id="TEST123",
        experiment_name="test",
        metric_name="auc",
        task_type="binary_classification",
        baseline=MetricSummary(
            mean=0.85, std=0.0, q05=0.85, q50=0.85, q95=0.85, n=1
        ),
        scenarios=[],
        applicable_metrics=["auc"],
        metric_results={},
        scenario_results_by_metric={},
        analysis_artifacts={},
        notes={},
    )
    text = format_summary(bundle)
    assert "TEST123" in text
    assert "auc" in text
    assert "0.85" in text


def test_format_summary_no_baseline():
    from metrics_lie.schema import ResultBundle

    bundle = ResultBundle(
        run_id="TEST456",
        experiment_name="test2",
        metric_name="f1",
        task_type="binary_classification",
        scenarios=[],
        applicable_metrics=[],
        metric_results={},
        scenario_results_by_metric={},
        analysis_artifacts={},
        notes={},
    )
    text = format_summary(bundle)
    assert "TEST456" in text
    assert "no baseline" in text


def test_format_comparison_returns_string():
    report = {
        "metric_name": "auc",
        "baseline_delta": {"mean": -0.05, "a": 0.90, "b": 0.85},
        "decision": {
            "winner": "A",
            "confidence": "high",
            "reasoning": "better AUC",
        },
    }
    text = format_comparison(report)
    assert "auc" in text
    assert "A" in text


def test_format_comparison_with_flags():
    report = {
        "metric_name": "f1",
        "baseline_delta": {"mean": 0.01, "a": 0.80, "b": 0.81},
        "decision": {
            "winner": "B",
            "confidence": "low",
            "reasoning": "marginal",
        },
        "risk_flags": ["calibration_drift", "subgroup_gap"],
    }
    text = format_comparison(report)
    assert "calibration_drift" in text
    assert "subgroup_gap" in text


def test_display_without_ipython(capsys):
    from metrics_lie.schema import MetricSummary, ResultBundle
    from metrics_lie.display import display

    bundle = ResultBundle(
        run_id="DISP001",
        experiment_name="display_test",
        metric_name="auc",
        task_type="binary_classification",
        baseline=MetricSummary(
            mean=0.90, std=0.01, q05=0.88, q50=0.90, q95=0.92, n=10
        ),
        scenarios=[],
        applicable_metrics=["auc"],
        metric_results={},
        scenario_results_by_metric={},
        analysis_artifacts={},
        notes={},
    )
    display(bundle)
    captured = capsys.readouterr()
    assert "DISP001" in captured.out
    assert "0.90" in captured.out


def test_to_html():
    from metrics_lie.schema import MetricSummary, ResultBundle
    from metrics_lie.display import _to_html

    bundle = ResultBundle(
        run_id="HTML001",
        experiment_name="html_test",
        metric_name="auc",
        task_type="binary_classification",
        baseline=MetricSummary(
            mean=0.85, std=0.0, q05=0.85, q50=0.85, q95=0.85, n=1
        ),
        scenarios=[],
        applicable_metrics=["auc"],
        metric_results={},
        scenario_results_by_metric={},
        analysis_artifacts={},
        notes={},
    )
    html = _to_html(bundle)
    assert "<pre" in html
    assert "HTML001" in html
