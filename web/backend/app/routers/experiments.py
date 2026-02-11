"""Experiments API router."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import get_current_user
from ..contracts import (
    ExperimentCreateRequest,
    ExperimentSummary,
    ResultSummary,
    RunAnalysisResponse,
    RunRequest,
    RunResponse,
)
from ..engine_bridge import run_experiment
from ..persistence import (
    load_experiment,
    load_result_for_run,
    list_experiments,
    list_runs,
    save_experiment,
    save_result,
)
from ..storage import METRIC_PRESETS, STRESS_SUITE_PRESETS

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("", response_model=ExperimentSummary, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    create_req: ExperimentCreateRequest,
    owner_id: str = Depends(get_current_user),
) -> ExperimentSummary:
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
        error_message=None,
    )

    # Persist
    save_experiment(experiment_id, create_req, summary, owner_id=owner_id)

    return summary


@router.get("", response_model=list[ExperimentSummary])
async def list_experiments_endpoint(
    owner_id: str = Depends(get_current_user),
) -> list[ExperimentSummary]:
    """List all experiments for the current user."""
    return list_experiments(owner_id=owner_id)


@router.get("/{experiment_id}", response_model=ExperimentSummary)
async def get_experiment(
    experiment_id: str,
    owner_id: str = Depends(get_current_user),
) -> ExperimentSummary:
    """Get a specific experiment by ID."""
    try:
        _, summary = load_experiment(experiment_id, owner_id=owner_id)
        return summary
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )


@router.post("/{experiment_id}/run", response_model=RunResponse)
async def run_experiment_endpoint(
    experiment_id: str,
    run_req: RunRequest,
    owner_id: str = Depends(get_current_user),
) -> RunResponse:
    """Run an experiment."""
    # Load experiment
    try:
        create_req, summary = load_experiment(experiment_id, owner_id=owner_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    # Check if experiment is already running
    if summary.status == "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Experiment already running",
        )

    # Update status to running
    summary.status = "running"
    summary.last_run_at = datetime.now(timezone.utc)
    summary.error_message = None  # Clear any previous error
    save_experiment(experiment_id, create_req, summary, owner_id=owner_id)

    # Create run
    run_id = str(uuid.uuid4())

    try:
        # Run experiment
        result = run_experiment(create_req, experiment_id, run_id, seed=run_req.seed)

        # Save result
        save_result(experiment_id, run_id, result, owner_id=owner_id)

        # Update status to completed
        summary.status = "completed"
        save_experiment(experiment_id, create_req, summary, owner_id=owner_id)

        return RunResponse(run_id=run_id, status="completed")

    except ValueError as e:
        # Dataset path or CSV reading errors - return 400 with helpful message
        error_msg = str(e)
        summary.status = "failed"
        summary.error_message = error_msg
        save_experiment(experiment_id, create_req, summary, owner_id=owner_id)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        )
    except Exception as e:
        # Update status to failed with error message
        error_msg = str(e)
        summary.status = "failed"
        summary.error_message = error_msg
        save_experiment(experiment_id, create_req, summary, owner_id=owner_id)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Experiment run failed: {error_msg}",
        )


@router.get("/{experiment_id}/runs")
async def list_runs_endpoint(
    experiment_id: str,
    owner_id: str = Depends(get_current_user),
) -> list[dict]:
    """List all runs for an experiment."""
    # Verify experiment exists
    try:
        load_experiment(experiment_id, owner_id=owner_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    return list_runs(experiment_id, owner_id=owner_id)


@router.get("/{experiment_id}/runs/{run_id}/results", response_model=ResultSummary)
async def get_run_results(
    experiment_id: str,
    run_id: str,
    owner_id: str = Depends(get_current_user),
) -> ResultSummary:
    """Get results for a specific run."""
    # Verify experiment exists
    try:
        load_experiment(experiment_id, owner_id=owner_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    try:
        return load_result_for_run(experiment_id, run_id, owner_id=owner_id)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get(
    "/{experiment_id}/runs/{run_id}/analysis", response_model=RunAnalysisResponse
)
async def get_run_analysis(
    experiment_id: str,
    run_id: str,
    owner_id: str = Depends(get_current_user),
) -> RunAnalysisResponse:
    """Get Phase 5 analysis artifacts for a specific run."""
    try:
        load_experiment(experiment_id, owner_id=owner_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    try:
        result = load_result_for_run(experiment_id, run_id, owner_id=owner_id)
        return RunAnalysisResponse(
            run_id=run_id, analysis_artifacts=result.analysis_artifacts or {}
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
