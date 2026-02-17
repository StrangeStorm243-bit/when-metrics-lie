"""Transform engine ResultBundle to web API ResultSummary."""
from __future__ import annotations

from datetime import datetime

from metrics_lie.schema import ResultBundle

from .contracts import (
    ComponentScore,
    FindingFlag,
    ResultSummary,
    ScenarioResult as ContractScenarioResult,
)


def bundle_to_result_summary(
    bundle: ResultBundle, experiment_id: str, run_id: str
) -> ResultSummary:
    """Convert a core engine ResultBundle to the web API ResultSummary contract."""
    # Extract headline score from baseline
    headline_score = bundle.baseline.mean if bundle.baseline else 0.0

    # Convert scenario results
    scenario_results = []
    for sr in bundle.scenarios:
        # Calculate delta: scenario mean - baseline mean
        delta = sr.metric.mean - headline_score if bundle.baseline else 0.0
        scenario_results.append(
            ContractScenarioResult(
                scenario_id=sr.scenario_id,
                scenario_name=sr.scenario_id.replace("_", " ").title(),
                delta=delta,
                score=sr.metric.mean,
                severity=None,
                notes=None,
            )
        )

    # Extract component scores from diagnostics
    component_scores = []
    if bundle.baseline:
        baseline_diag = bundle.notes.get("baseline_diagnostics", {})
        if "brier" in baseline_diag:
            component_scores.append(
                ComponentScore(
                    name="brier_score",
                    score=baseline_diag["brier"],
                    weight=None,
                    notes="Baseline Brier score",
                )
            )
        if "ece" in baseline_diag:
            component_scores.append(
                ComponentScore(
                    name="ece_score",
                    score=baseline_diag["ece"],
                    weight=None,
                    notes="Baseline ECE",
                )
            )

    # Extract flags from diagnostics
    flags = []
    if bundle.baseline:
        baseline_diag = bundle.notes.get("baseline_diagnostics", {})
        if baseline_diag.get("ece", 0) > 0.1:
            flags.append(
                FindingFlag(
                    code="high_ece",
                    title="High Expected Calibration Error",
                    detail=f"ECE is {baseline_diag.get('ece', 0):.4f}, indicating poor calibration",
                    severity="warn",
                )
            )

    # Phase 8: Extract dashboard_summary from analysis_artifacts if present
    dashboard_summary = None
    if bundle.analysis_artifacts and "dashboard_summary" in bundle.analysis_artifacts:
        dashboard_summary = bundle.analysis_artifacts["dashboard_summary"]

    return ResultSummary(
        experiment_id=experiment_id,
        run_id=run_id,
        headline_score=headline_score,
        weighted_score=None,
        component_scores=component_scores,
        scenario_results=scenario_results,
        flags=flags,
        prediction_surface=bundle.prediction_surface,
        applicable_metrics=bundle.applicable_metrics,
        analysis_artifacts=bundle.analysis_artifacts,
        dashboard_summary=dashboard_summary,
        generated_at=datetime.fromisoformat(bundle.created_at.replace("Z", "+00:00")),
    )
