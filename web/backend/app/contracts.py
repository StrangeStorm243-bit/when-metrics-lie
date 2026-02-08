"""Pydantic contracts for Spectra web API.

These contracts define the stable interface between the frontend and backend.
They will be used in Phase 3.2+ when endpoints are implemented.
"""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ExperimentCreateRequest(BaseModel):
    """Request to create a new experiment."""
    name: str = Field(..., description="Human-readable experiment name")
    metric_id: str = Field(..., description="Metric identifier from presets")
    stress_suite_id: str = Field(..., description="Stress suite identifier from presets")
    notes: Optional[str] = Field(None, description="Optional notes")
    config: dict = Field(default_factory=dict, description="Additional configuration (reserved for future use)")


class ExperimentSummary(BaseModel):
    """Summary of an experiment."""
    id: str = Field(..., description="Experiment ID")
    name: str = Field(..., description="Experiment name")
    metric_id: str = Field(..., description="Metric identifier")
    stress_suite_id: str = Field(..., description="Stress suite identifier")
    status: Literal["created", "running", "completed", "failed"] = Field(..., description="Current status")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_run_at: Optional[datetime] = Field(None, description="Timestamp of last run")
    error_message: Optional[str] = Field(None, description="Error message if status is failed")


class RunRequest(BaseModel):
    """Request to run an experiment."""
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class RunResponse(BaseModel):
    """Response from run request."""
    run_id: str = Field(..., description="Run ID")
    status: Literal["queued", "running", "completed", "failed"] = Field(..., description="Run status")


class ComponentScore(BaseModel):
    """Score for a single decision component."""
    name: str = Field(..., description="Component name")
    score: float = Field(..., description="Component score")
    weight: Optional[float] = Field(None, description="Weight applied to this component")
    notes: Optional[str] = Field(None, description="Optional notes")


class ScenarioResult(BaseModel):
    """Result for a single scenario."""
    scenario_id: str = Field(..., description="Scenario identifier")
    scenario_name: str = Field(..., description="Human-readable scenario name")
    delta: float = Field(..., description="Delta value for this scenario")
    score: float = Field(..., description="Score for this scenario")
    severity: Optional[Literal["low", "med", "high"]] = Field(None, description="Severity level")
    notes: Optional[str] = Field(None, description="Optional notes")


class FindingFlag(BaseModel):
    """A finding or flag from evaluation."""
    code: str = Field(..., description="Flag code identifier")
    title: str = Field(..., description="Short title")
    detail: str = Field(..., description="Detailed description")
    severity: Literal["info", "warn", "critical"] = Field(..., description="Severity level")


class ResultSummary(BaseModel):
    """Summary of evaluation results."""
    experiment_id: str = Field(..., description="Experiment ID")
    run_id: str = Field(..., description="Run ID")
    headline_score: float = Field(..., description="Primary headline score")
    weighted_score: Optional[float] = Field(None, description="Weighted score if available")
    component_scores: list[ComponentScore] = Field(default_factory=list, description="Per-component scores")
    scenario_results: list[ScenarioResult] = Field(default_factory=list, description="Per-scenario results")
    flags: list[FindingFlag] = Field(default_factory=list, description="Findings and flags")
    prediction_surface: Optional[dict] = Field(
        None, description="Standardized prediction surface summary (Phase 5)"
    )
    applicable_metrics: list[str] = Field(
        default_factory=list, description="Applicable metrics resolved for this run (Phase 5)"
    )
    analysis_artifacts: Optional[dict] = Field(
        None, description="Phase 5 analysis artifacts (threshold sweep, sensitivity, etc.)"
    )
    generated_at: datetime = Field(..., description="Timestamp when results were generated")


class RunAnalysisResponse(BaseModel):
    """Phase 5 analysis artifacts for a run."""
    run_id: str = Field(..., description="Run ID")
    analysis_artifacts: dict = Field(default_factory=dict, description="Threshold sweep, sensitivity, disagreement, failure modes")

