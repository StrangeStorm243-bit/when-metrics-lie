"""Experiments API router."""
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from ..contracts import ExperimentCreateRequest, ExperimentSummary, RunRequest, RunResponse
from ..engine_bridge import run_experiment
from ..persistence import load_experiment, list_experiments, save_experiment, save_result
from ..storage import METRIC_PRESETS, STRESS_SUITE_PRESETS

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("", response_model=ExperimentSummary, status_code=status.HTTP_201_CREATED)
async def create_experiment(create_req: ExperimentCreateRequest) -> ExperimentSummary:
    """Create a new experiment."""
    # Validate metric_id
    metric_ids = {p["id"] for p in METRIC_PRESETS}
    if create_req.metric_id not in metric_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid metric_id: {create_req.metric_id}. Available: {sorted(metric_ids)}",
        )

    # Validate stress_suite_id
    suite_ids = {p["id"] for p in STRESS_SUITE_PRESETS}
    if create_req.stress_suite_id not in suite_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid stress_suite_id: {create_req.stress_suite_id}. Available: {sorted(suite_ids)}",
        )

    # Create experiment
    experiment_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    summary = ExperimentSummary(
        id=experiment_id,
        name=create_req.name,
        metric_id=create_req.metric_id,
        stress_suite_id=create_req.stress_suite_id,
        status="created",
        created_at=now,
        last_run_at=None,
    )

    # Persist
    save_experiment(experiment_id, create_req, summary)

    return summary


@router.get("", response_model=list[ExperimentSummary])
async def list_experiments_endpoint() -> list[ExperimentSummary]:
    """List all experiments."""
    return list_experiments()


@router.post("/{experiment_id}/run", response_model=RunResponse)
async def run_experiment_endpoint(experiment_id: str, run_req: RunRequest) -> RunResponse:
    """Run an experiment."""
    # Load experiment
    try:
        create_req, summary = load_experiment(experiment_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    # Update status to running
    summary.status = "running"
    summary.last_run_at = datetime.now(timezone.utc)
    save_experiment(experiment_id, create_req, summary)

    # Create run
    run_id = str(uuid.uuid4())

    try:
        # Run experiment
        result = run_experiment(create_req, experiment_id, run_id, seed=run_req.seed)

        # Save result
        save_result(experiment_id, run_id, result)

        # Update status to completed
        summary.status = "completed"
        save_experiment(experiment_id, create_req, summary)

        return RunResponse(run_id=run_id, status="completed")

    except Exception as e:
        # Update status to failed
        summary.status = "failed"
        save_experiment(experiment_id, create_req, summary)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Experiment run failed: {str(e)}",
        )

