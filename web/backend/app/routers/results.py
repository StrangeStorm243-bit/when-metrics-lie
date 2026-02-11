"""Results API router."""

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import get_current_user
from ..contracts import ResultSummary
from ..persistence import load_latest_result

router = APIRouter(prefix="/experiments", tags=["results"])


@router.get("/{experiment_id}/results", response_model=ResultSummary)
async def get_results(
    experiment_id: str,
    owner_id: str = Depends(get_current_user),
) -> ResultSummary:
    """Get the latest results for an experiment."""
    result = load_latest_result(experiment_id, owner_id=owner_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No results found for experiment {experiment_id}",
        )

    return result
