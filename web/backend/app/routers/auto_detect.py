"""Auto-detect endpoint: analyze model + dataset to recommend configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import get_current_user
from ..contracts import AutoDetectRequest, AutoDetectResponse

router = APIRouter(tags=["auto-detect"])

# Default recommendations per task type: (metric, stress_suite)
_DEFAULTS: dict[str, tuple[str, str]] = {
    "binary_classification": ("auc", "balanced"),
    "multiclass_classification": ("weighted_f1", "balanced"),
    "regression": ("rmse", "performance_first"),
    "ranking": ("auc", "performance_first"),
}


def _repo_root() -> Path:
    """Repository root (where pyproject.toml lives)."""
    p = Path(__file__).resolve()
    for _ in range(5):
        p = p.parent
        if (p / "pyproject.toml").exists():
            return p
    return Path(__file__).resolve().parent.parent.parent.parent


def _load_dataset_meta(dataset_id: str, owner_id: str) -> dict:
    """Load dataset metadata from local storage."""
    root = _repo_root()
    meta_path = root / ".spectra_ui" / "datasets" / owner_id / f"{dataset_id}.meta.json"
    if not meta_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found",
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_model_meta(model_id: str, owner_id: str) -> dict | None:
    """Load model metadata from local storage, or None if not found."""
    root = _repo_root()
    meta_path = root / ".spectra_ui" / "models" / owner_id / f"{model_id}.meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _infer_task_type_from_dataset(ds_meta: dict) -> str:
    """Infer task type by inspecting y_true column values in the CSV."""
    y_true_col = ds_meta.get("detected_y_true_col")
    if not y_true_col:
        return "binary_classification"

    # Load CSV to inspect y_true values
    owner_id = ds_meta.get("owner_id", "anon")
    dataset_id = ds_meta["dataset_id"]
    root = _repo_root()
    csv_path = root / ".spectra_ui" / "datasets" / owner_id / f"{dataset_id}.csv"
    if not csv_path.exists():
        return "binary_classification"

    try:
        df = pd.read_csv(csv_path, nrows=1000)
    except Exception:
        return "binary_classification"

    if y_true_col not in df.columns:
        return "binary_classification"

    unique_vals = df[y_true_col].dropna().unique()
    n_unique = len(unique_vals)

    # Check for binary: exactly 2 unique values that are 0 and 1
    if n_unique == 2:
        val_set = set(int(v) if float(v) == int(float(v)) else v for v in unique_vals)
        if val_set == {0, 1}:
            return "binary_classification"

    # Check for multiclass: 3-20 unique integer-like values
    if 3 <= n_unique <= 20:
        try:
            all_int = all(float(v) == int(float(v)) for v in unique_vals)
            if all_int:
                return "multiclass_classification"
        except (ValueError, TypeError):
            pass

    # More than 20 unique or non-integer values -> regression
    if n_unique > 20:
        return "regression"

    return "binary_classification"


@router.post("/auto-detect", response_model=AutoDetectResponse)
async def auto_detect(
    request: AutoDetectRequest,
    owner_id: str = Depends(get_current_user),
) -> AutoDetectResponse:
    """Analyze uploaded model + dataset to recommend experiment configuration."""
    ds_meta = _load_dataset_meta(request.dataset_id, owner_id)

    model_meta: dict | None = None
    if request.model_id:
        model_meta = _load_model_meta(request.model_id, owner_id)

    # Determine task type
    if model_meta and model_meta.get("task_type"):
        task_type = model_meta["task_type"]
        confidence = "high"
    else:
        task_type = _infer_task_type_from_dataset(ds_meta)
        confidence = "medium"

    metric, suite = _DEFAULTS.get(task_type, ("auc", "balanced"))

    return AutoDetectResponse(
        task_type=task_type,
        y_true_col=ds_meta.get("detected_y_true_col"),
        y_score_col=ds_meta.get("detected_y_score_col"),
        feature_cols=ds_meta.get("detected_feature_cols", []),
        recommended_metric=metric,
        recommended_stress_suite=suite,
        n_rows=ds_meta.get("n_rows", 0),
        model_class=model_meta["model_class"] if model_meta else None,
        confidence=confidence,
    )
