"""Presets API router."""

from fastapi import APIRouter

from ..storage import METRIC_PRESETS, STRESS_SUITE_PRESETS

router = APIRouter(prefix="/presets", tags=["presets"])


@router.get("/metrics")
async def get_metric_presets():
    """Get available metric presets."""
    return METRIC_PRESETS


@router.get("/stress-suites")
async def get_stress_suite_presets():
    """Get available stress suite presets."""
    return STRESS_SUITE_PRESETS

