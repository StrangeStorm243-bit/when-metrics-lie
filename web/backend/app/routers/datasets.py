"""Datasets API router: upload CSV datasets with auto-detected column roles."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, status, UploadFile

from ..auth import get_current_user
from ..config import get_settings
from ..contracts import DatasetUploadResponse
from ..storage_backend import get_storage_backend

router = APIRouter(prefix="/datasets", tags=["datasets"])

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

Y_TRUE_PATTERNS = [
    "y_true", "label", "target", "ground_truth", "true_label", "class", "y",
]
Y_SCORE_PATTERNS = [
    "y_score", "prediction", "pred", "probability", "prob", "score", "y_pred",
    "predicted",
]
NON_FEATURE_PATTERNS = [
    "id", "index", "row", "unnamed", "group", "subgroup",
]


def _detect_columns(
    columns: list[str],
) -> tuple[str | None, str | None, list[str]]:
    """Auto-detect y_true, y_score, and feature columns from column names.

    Returns (y_true_col, y_score_col, feature_cols).
    """
    lower_map = {c.lower().strip(): c for c in columns}

    y_true_col: str | None = None
    for pattern in Y_TRUE_PATTERNS:
        if pattern in lower_map:
            y_true_col = lower_map[pattern]
            break

    y_score_col: str | None = None
    for pattern in Y_SCORE_PATTERNS:
        if pattern in lower_map:
            y_score_col = lower_map[pattern]
            break

    excluded = {y_true_col, y_score_col}
    feature_cols: list[str] = []
    for col in columns:
        if col in excluded:
            continue
        col_lower = col.lower().strip()
        if any(col_lower.startswith(p) for p in NON_FEATURE_PATTERNS):
            continue
        feature_cols.append(col)

    return y_true_col, y_score_col, feature_cols


def _repo_root() -> Path:
    """Repository root (where pyproject.toml lives)."""
    p = Path(__file__).resolve()
    for _ in range(5):
        p = p.parent
        if (p / "pyproject.toml").exists():
            return p
    return Path(__file__).resolve().parent.parent.parent.parent


def _datasets_dir_local(owner_id: str) -> Path:
    """Local filesystem directory for a user's datasets."""
    root = _repo_root()
    d = root / ".spectra_ui" / "datasets" / owner_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dataset_key(owner_id: str, dataset_id: str, suffix: str) -> str:
    """Storage key for dataset file or meta (hosted)."""
    return f"datasets/{owner_id}/{dataset_id}{suffix}"


@router.post(
    "",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_dataset(
    file: UploadFile = File(...),
    owner_id: str = Depends(get_current_user),
) -> DatasetUploadResponse:
    """Upload a CSV dataset with auto-detected column roles."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File must have a filename",
        )

    ext = PurePosixPath(file.filename).suffix.lower()
    if ext != ".csv":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Only .csv files are accepted, got '{ext}'",
        )

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )

    # Parse CSV
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse CSV: {exc}",
        ) from exc

    columns = list(df.columns.astype(str))
    n_rows = len(df)

    y_true_col, y_score_col, feature_cols = _detect_columns(columns)

    dataset_id = hashlib.sha256(raw).hexdigest()
    settings = get_settings()
    now = datetime.now(timezone.utc).isoformat()

    meta = {
        "dataset_id": dataset_id,
        "original_filename": file.filename,
        "columns": columns,
        "n_rows": n_rows,
        "detected_y_true_col": y_true_col,
        "detected_y_score_col": y_score_col,
        "detected_feature_cols": feature_cols,
        "uploaded_at": now,
        "owner_id": owner_id,
    }

    if settings.is_hosted:
        backend = get_storage_backend()
        file_key = _dataset_key(owner_id, dataset_id, ".csv")
        meta_key = _dataset_key(owner_id, dataset_id, ".meta.json")
        backend.upload(file_key, raw, content_type="text/csv")
        backend.upload(
            meta_key,
            json.dumps(meta, indent=2).encode("utf-8"),
            content_type="application/json",
        )
    else:
        local_dir = _datasets_dir_local(owner_id)
        file_path = local_dir / f"{dataset_id}.csv"
        meta_path = local_dir / f"{dataset_id}.meta.json"
        file_path.write_bytes(raw)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        original_filename=file.filename,
        columns=columns,
        n_rows=n_rows,
        detected_y_true_col=y_true_col,
        detected_y_score_col=y_score_col,
        detected_feature_cols=feature_cols,
    )
