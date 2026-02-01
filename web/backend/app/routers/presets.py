"""Presets API router."""

from pathlib import Path

from fastapi import APIRouter

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
async def get_metric_presets():
    """Get available metric presets."""
    return METRIC_PRESETS


@router.get("/stress-suites")
async def get_stress_suite_presets():
    """Get available stress suite presets."""
    return STRESS_SUITE_PRESETS


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
        
        datasets.append({
            "id": file_id,
            "name": name,
            "path": path,
        })
    
    return datasets

