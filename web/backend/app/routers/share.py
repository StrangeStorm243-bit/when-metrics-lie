"""Share API: create share links and serve public run data by token."""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..auth import get_current_user
from ..contracts import (
    ShareCreateRequest,
    ShareCreateResponse,
    SharedRunResponse,
)
from ..persistence import (
    load_experiment,
    load_result_for_shared_run,
    save_share_token,
    validate_share_token,
)

router = APIRouter(prefix="/share", tags=["share"])


def _generate_share_token() -> str:
    """Generate a cryptographically strong share token."""
    return secrets.token_urlsafe(32)


@router.post("/create", response_model=ShareCreateResponse)
async def create_share_link(
    req: ShareCreateRequest,
    owner_id: str = Depends(get_current_user),
) -> ShareCreateResponse:
    """Generate a share token for a run. Requires auth (only owner can share)."""
    try:
        load_experiment(req.experiment_id, owner_id=owner_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {req.experiment_id} not found",
        )

    try:
        from ..persistence import load_result_for_run

        load_result_for_run(req.experiment_id, req.run_id, owner_id=owner_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {req.run_id} not found",
        )

    share_token = _generate_share_token()
    save_share_token(
        req.experiment_id, req.run_id, share_token, owner_id=owner_id
    )
    return ShareCreateResponse(share_token=share_token)


@router.get("/{experiment_id}/{run_id}", response_model=SharedRunResponse)
async def get_shared_run(
    experiment_id: str,
    run_id: str,
    token: str = Query(..., description="Share token"),
) -> SharedRunResponse:
    """Public endpoint -- no auth required. Validates share_token and returns run data."""
    owner_id = validate_share_token(experiment_id, run_id, token)
    if not owner_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired share link",
        )

    try:
        result = load_result_for_shared_run(experiment_id, run_id, owner_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    return SharedRunResponse(
        result=result,
        analysis_artifacts=result.analysis_artifacts or {},
    )
