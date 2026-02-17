"""Tests for the extracted bundle_to_result_summary transformation.

Verifies that the web API ResultSummary is correctly built from a core
engine ResultBundle with known values.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from metrics_lie.schema import MetricSummary, ResultBundle, ScenarioResult

from web.backend.app.bundle_transform import bundle_to_result_summary


def _make_metric_summary(mean: float = 0.85) -> MetricSummary:
    return MetricSummary(mean=mean, std=0.02, q05=0.81, q50=0.85, q95=0.89, n=200)


def _make_bundle(
    *,
    baseline_mean: float = 0.90,
    scenarios: list[ScenarioResult] | None = None,
    baseline_diagnostics: dict | None = None,
    analysis_artifacts: dict | None = None,
    applicable_metrics: list[str] | None = None,
) -> ResultBundle:
    baseline = _make_metric_summary(baseline_mean)
    if scenarios is None:
        scenarios = [
            ScenarioResult(
                scenario_id="label_noise",
                params={"p": 0.1},
                metric=_make_metric_summary(0.85),
            ),
            ScenarioResult(
                scenario_id="score_noise",
                params={"sigma": 0.05},
                metric=_make_metric_summary(0.88),
            ),
        ]
    notes: dict = {"phase": "test"}
    if baseline_diagnostics is not None:
        notes["baseline_diagnostics"] = baseline_diagnostics
    return ResultBundle(
        run_id="TESTRUN01",
        experiment_name="test_experiment",
        metric_name="auc",
        baseline=baseline,
        scenarios=scenarios,
        notes=notes,
        analysis_artifacts=analysis_artifacts or {},
        applicable_metrics=applicable_metrics or ["auc"],
        created_at="2025-01-15T12:00:00+00:00",
    )


def test_headline_score_from_baseline():
    bundle = _make_bundle(baseline_mean=0.90)
    result = bundle_to_result_summary(bundle, "exp_1", "run_1")
    assert result.headline_score == pytest.approx(0.90)


def test_scenario_results_count():
    bundle = _make_bundle()
    result = bundle_to_result_summary(bundle, "exp_1", "run_1")
    assert len(result.scenario_results) == 2


def test_scenario_delta_calculation():
    bundle = _make_bundle(baseline_mean=0.90)
    result = bundle_to_result_summary(bundle, "exp_1", "run_1")
    # label_noise scenario mean = 0.85, baseline = 0.90, delta = -0.05
    label_noise = result.scenario_results[0]
    assert label_noise.scenario_id == "label_noise"
    assert label_noise.delta == pytest.approx(-0.05)
    assert label_noise.score == pytest.approx(0.85)


def test_scenario_name_formatting():
    bundle = _make_bundle()
    result = bundle_to_result_summary(bundle, "exp_1", "run_1")
    assert result.scenario_results[0].scenario_name == "Label Noise"
    assert result.scenario_results[1].scenario_name == "Score Noise"


def test_component_scores_brier_and_ece():
    bundle = _make_bundle(baseline_diagnostics={"brier": 0.12, "ece": 0.08})
    result = bundle_to_result_summary(bundle, "exp_1", "run_1")
    names = [c.name for c in result.component_scores]
    assert "brier_score" in names
    assert "ece_score" in names
    brier = next(c for c in result.component_scores if c.name == "brier_score")
    assert brier.score == pytest.approx(0.12)


def test_component_scores_empty_without_diagnostics():
    bundle = _make_bundle()
    result = bundle_to_result_summary(bundle, "exp_1", "run_1")
    assert result.component_scores == []


def test_high_ece_flag():
    bundle = _make_bundle(baseline_diagnostics={"ece": 0.15})
    result = bundle_to_result_summary(bundle, "exp_1", "run_1")
    assert len(result.flags) == 1
    assert result.flags[0].code == "high_ece"
    assert result.flags[0].severity == "warn"


def test_no_flag_when_ece_is_low():
    bundle = _make_bundle(baseline_diagnostics={"ece": 0.05})
    result = bundle_to_result_summary(bundle, "exp_1", "run_1")
    assert len(result.flags) == 0


def test_dashboard_summary_extracted():
    dashboard = {"primary_metric": "auc", "overview": {"auc": {"mean": 0.9}}}
    bundle = _make_bundle(analysis_artifacts={"dashboard_summary": dashboard})
    result = bundle_to_result_summary(bundle, "exp_1", "run_1")
    assert result.dashboard_summary == dashboard


def test_dashboard_summary_none_without_artifacts():
    bundle = _make_bundle()
    result = bundle_to_result_summary(bundle, "exp_1", "run_1")
    assert result.dashboard_summary is None


def test_applicable_metrics_passed_through():
    bundle = _make_bundle(applicable_metrics=["auc", "f1", "accuracy"])
    result = bundle_to_result_summary(bundle, "exp_1", "run_1")
    assert result.applicable_metrics == ["auc", "f1", "accuracy"]


def test_ids_passed_through():
    bundle = _make_bundle()
    result = bundle_to_result_summary(bundle, "exp_42", "run_99")
    assert result.experiment_id == "exp_42"
    assert result.run_id == "run_99"


def test_no_baseline_returns_zero_headline():
    bundle = ResultBundle(
        run_id="TESTRUN01",
        experiment_name="test",
        metric_name="auc",
        baseline=None,
        scenarios=[],
        created_at="2025-01-15T12:00:00+00:00",
    )
    result = bundle_to_result_summary(bundle, "exp_1", "run_1")
    assert result.headline_score == 0.0
