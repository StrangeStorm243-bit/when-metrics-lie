"""Compare API: compare two runs via core engine compare_bundles."""

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import get_current_user
from ..contracts import CompareRequest, CompareResponse
from ..persistence import load_bundle

router = APIRouter(tags=["compare"])


@router.post("/compare", response_model=CompareResponse)
async def compare_runs_endpoint(
    req: CompareRequest,
    owner_id: str = Depends(get_current_user),
) -> CompareResponse:
    """Compare two runs using the core engine's pure compare_bundles function."""
    bundle_a = load_bundle(
        req.run_a.experiment_id, req.run_a.run_id, owner_id=owner_id
    )
    bundle_b = load_bundle(
        req.run_b.experiment_id, req.run_b.run_id, owner_id=owner_id
    )
    if not bundle_a or not bundle_b:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bundle not found -- run may predate bundle persistence",
        )

    from metrics_lie.compare.compare import compare_bundles

    result = compare_bundles(bundle_a, bundle_b)

    return CompareResponse(
        run_a=result.get("run_a", req.run_a.run_id),
        run_b=result.get("run_b", req.run_b.run_id),
        metric_name=result.get("metric_name", ""),
        baseline_delta=result.get("baseline_delta", {}),
        scenario_deltas=result.get("scenario_deltas", {}),
        regressions=result.get("regressions", {}),
        risk_flags=result.get("risk_flags", []),
        decision=result.get("decision", {}),
    )
