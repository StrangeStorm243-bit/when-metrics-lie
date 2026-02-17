from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field


class AnalysisArtifacts(TypedDict, total=False):
    """Known keys produced by the analysis phase in execution.py."""

    threshold_sweep: dict
    sensitivity: dict
    metric_disagreements: list
    failure_modes: dict
    dashboard_summary: dict


class MetricSummary(BaseModel):
    """
    Summary statistics of a metric distribution.
    """

    mean: float
    std: float
    q05: float
    q50: float
    q95: float
    n: int


class Artifact(BaseModel):
    """
    A generated file (plot, table, etc.) stored alongside the run results.
    """

    kind: str = Field(..., description="e.g. 'plot', 'table', 'json'")
    path: str = Field(..., description="Relative path within the run directory.")
    meta: Dict[str, Any] = Field(default_factory=dict)


class ScenarioResult(BaseModel):
    scenario_id: str
    params: Dict[str, Any] = Field(default_factory=dict)

    metric: MetricSummary
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[Artifact] = Field(default_factory=list)


class ResultBundle(BaseModel):
    """
    Single authoritative output JSON for one run.
    """

    schema_version: str = "0.1"
    run_id: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    experiment_name: str
    metric_name: str

    baseline: Optional[MetricSummary] = None
    scenarios: List[ScenarioResult] = Field(default_factory=list)
    prediction_surface: Optional[Dict[str, Any]] = None
    applicable_metrics: List[str] = Field(default_factory=list)
    metric_results: Dict[str, MetricSummary] = Field(default_factory=dict)
    scenario_results_by_metric: Dict[str, List[ScenarioResult]] = Field(
        default_factory=dict
    )
    analysis_artifacts: Dict[str, Any] = Field(default_factory=dict)

    notes: Dict[str, Any] = Field(default_factory=dict)

    def to_pretty_json(self) -> str:
        return self.model_dump_json(indent=2)
