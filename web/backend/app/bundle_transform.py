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
    task_type = getattr(bundle, "task_type", "binary_classification")
    headline_score = bundle.baseline.mean if bundle.baseline else 0.0

    # Convert scenario results
    scenario_results = []
    for sr in bundle.scenarios:
        delta = sr.metric.mean - headline_score if bundle.baseline else 0.0
        scenario_results.append(
            ContractScenarioResult(
                scenario_id=sr.scenario_id,
                scenario_name=sr.scenario_id.replace("_", " ").title(),
                delta=delta,
                score=sr.metric.mean,
                severity=_classify_severity(delta, task_type),
                notes=None,
            )
        )

    # Extract component scores (task-aware)
    component_scores = _extract_component_scores(bundle, task_type)

    # Extract flags (task-aware)
    flags = _extract_flags(bundle, task_type)

    # Extract task-specific fields from notes
    task_specific = bundle.notes.get("task_specific", {})

    # Dashboard summary from analysis artifacts
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
        task_type=task_type,
        confusion_matrix=task_specific.get("confusion_matrix"),
        class_names=task_specific.get("class_names"),
        per_class_metrics=task_specific.get("per_class_metrics"),
        residual_stats=task_specific.get("residual_stats"),
        ranking_metrics=task_specific.get("ranking_metrics"),
        generated_at=datetime.fromisoformat(bundle.created_at.replace("Z", "+00:00")),
    )


def _classify_severity(delta: float, task_type: str) -> str | None:
    """Classify scenario severity based on delta magnitude."""
    abs_delta = abs(delta)
    if abs_delta >= 0.1:
        return "high"
    elif abs_delta >= 0.05:
        return "med"
    elif abs_delta >= 0.01:
        return "low"
    return None


def _extract_component_scores(bundle: ResultBundle, task_type: str) -> list[ComponentScore]:
    """Extract component scores from baseline diagnostics."""
    scores = []
    if not bundle.baseline:
        return scores

    baseline_diag = bundle.notes.get("baseline_diagnostics", {})

    # Binary calibration
    if "brier" in baseline_diag:
        scores.append(ComponentScore(
            name="brier_score", score=baseline_diag["brier"],
            weight=None, notes="Baseline Brier score",
        ))
    if "ece" in baseline_diag:
        scores.append(ComponentScore(
            name="ece_score", score=baseline_diag["ece"],
            weight=None, notes="Baseline ECE",
        ))

    # Multiclass calibration
    if "multiclass_brier" in baseline_diag:
        scores.append(ComponentScore(
            name="multiclass_brier", score=baseline_diag["multiclass_brier"],
            weight=None, notes="Multiclass Brier score",
        ))
    if "multiclass_ece" in baseline_diag:
        scores.append(ComponentScore(
            name="multiclass_ece", score=baseline_diag["multiclass_ece"],
            weight=None, notes="Multiclass ECE",
        ))

    # Task-specific summary scores from notes
    task_specific = bundle.notes.get("task_specific", {})
    residual_stats = task_specific.get("residual_stats", {})
    if residual_stats:
        scores.append(ComponentScore(
            name="mae", score=residual_stats.get("mae", 0.0),
            weight=None, notes="Mean Absolute Error",
        ))
        scores.append(ComponentScore(
            name="rmse", score=residual_stats.get("rmse", 0.0),
            weight=None, notes="Root Mean Squared Error",
        ))

    return scores


def _extract_flags(bundle: ResultBundle, task_type: str) -> list[FindingFlag]:
    """Extract finding flags from diagnostics."""
    flags = []
    if not bundle.baseline:
        return flags

    baseline_diag = bundle.notes.get("baseline_diagnostics", {})

    # Binary ECE warning
    if baseline_diag.get("ece", 0) > 0.1:
        flags.append(FindingFlag(
            code="high_ece",
            title="High Expected Calibration Error",
            detail=f"ECE is {baseline_diag.get('ece', 0):.4f}, indicating poor calibration",
            severity="warn",
        ))

    # Multiclass ECE warning
    if baseline_diag.get("multiclass_ece", 0) > 0.1:
        flags.append(FindingFlag(
            code="high_multiclass_ece",
            title="High Multiclass Calibration Error",
            detail=f"Multiclass ECE is {baseline_diag.get('multiclass_ece', 0):.4f}",
            severity="warn",
        ))

    # Regression high-residual warning
    task_specific = bundle.notes.get("task_specific", {})
    residual_stats = task_specific.get("residual_stats", {})
    if residual_stats:
        std = residual_stats.get("std", 0)
        max_abs = max(abs(residual_stats.get("min", 0)), abs(residual_stats.get("max", 0)))
        if max_abs > 3 * std and std > 0:
            flags.append(FindingFlag(
                code="high_residual_outliers",
                title="Large Residual Outliers",
                detail=f"Max residual ({max_abs:.4f}) exceeds 3x std ({std:.4f})",
                severity="warn",
            ))

    return flags
