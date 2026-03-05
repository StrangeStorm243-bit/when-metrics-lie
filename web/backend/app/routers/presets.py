"""Presets API router."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query

from metrics_lie.surface_compat import SCENARIO_TASK_COMPAT

from ..storage import METRIC_PRESETS, STRESS_SUITE_PRESETS


def _find_repo_root() -> Path:
    """Find repository root by walking up until pyproject.toml is found."""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current.resolve()
        current = current.parent
    raise RuntimeError("Could not find repository root (pyproject.toml not found)")


router = APIRouter(prefix="/presets", tags=["presets"])


@router.get("/metrics")
async def get_metric_presets(
    task_type: str | None = Query(None, description="Filter by task type"),
):
    """Get available metric presets, optionally filtered by task type."""
    if task_type is None:
        return METRIC_PRESETS
    return [p for p in METRIC_PRESETS if task_type in p.get("task_types", [])]


@router.get("/stress-suites")
async def get_stress_suite_presets(
    task_type: str | None = Query(None, description="Filter by task type"),
):
    """Get available stress suite presets, optionally filtered by task type."""
    if task_type is None:
        return STRESS_SUITE_PRESETS

    allowed_scenarios = SCENARIO_TASK_COMPAT.get(task_type, set())
    filtered: list[dict] = []
    for suite in STRESS_SUITE_PRESETS:
        suite_scenarios = suite.get("scenarios", [])
        compatible = [s for s in suite_scenarios if s in allowed_scenarios]
        if compatible:
            filtered.append({**suite, "scenarios": compatible})
    return filtered


@router.get("/datasets")
async def get_dataset_presets():
    """Get available dataset presets from data/*.csv files."""
    repo_root = _find_repo_root()
    data_dir = repo_root / "data"

    if not data_dir.exists() or not data_dir.is_dir():
        return []

    csv_files = sorted(data_dir.glob("*.csv"))
    datasets = []

    for csv_file in csv_files:
        # id: filename without extension
        file_id = csv_file.stem
        # name: human-friendly (replace underscores with spaces, title case)
        name = file_id.replace("_", " ").replace("-", " ").title()
        # path: relative to repo root
        path = csv_file.relative_to(repo_root).as_posix()

        datasets.append(
            {
                "id": file_id,
                "name": name,
                "path": path,
            }
        )

    return datasets
