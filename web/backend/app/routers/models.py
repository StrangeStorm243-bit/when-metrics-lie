"""Models API router: upload and list sklearn pickle models."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, status, UploadFile

from ..auth import get_current_user
from ..config import get_settings
from ..contracts import ModelMeta, ModelUploadResponse
from ..storage_backend import get_storage_backend

router = APIRouter(prefix="/models", tags=["models"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


@router.get("/formats")
async def list_supported_formats():
    """List supported model formats and their accepted extensions."""
    return [
        {
            "format_id": "pickle",
            "name": "sklearn Pickle",
            "extensions": [".pkl", ".joblib"],
            "task_types": ["binary_classification", "multiclass_classification", "regression"],
        },
        {
            "format_id": "onnx",
            "name": "ONNX",
            "extensions": [".onnx"],
            "task_types": ["binary_classification", "multiclass_classification", "regression", "ranking"],
        },
        {
            "format_id": "xgboost",
            "name": "XGBoost",
            "extensions": [".ubj", ".xgb"],
            "task_types": ["binary_classification", "multiclass_classification", "regression", "ranking"],
        },
        {
            "format_id": "lightgbm",
            "name": "LightGBM",
            "extensions": [".lgb"],
            "task_types": ["binary_classification", "multiclass_classification", "regression", "ranking"],
        },
        {
            "format_id": "catboost",
            "name": "CatBoost",
            "extensions": [".cbm"],
            "task_types": ["binary_classification", "multiclass_classification", "regression", "ranking"],
        },
    ]


def _repo_root() -> Path:
    """Repository root (where pyproject.toml lives)."""
    p = Path(__file__).resolve()
    for _ in range(5):
        p = p.parent
        if (p / "pyproject.toml").exists():
            return p
    return Path(__file__).resolve().parent.parent.parent.parent


def _models_dir_local(owner_id: str) -> Path:
    """Local filesystem directory for a user's models."""
    root = _repo_root()
    d = root / ".spectra_ui" / "models" / owner_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _model_key(owner_id: str, model_id: str, suffix: str) -> str:
    """Storage key for model file or meta (hosted)."""
    return f"models/{owner_id}/{model_id}{suffix}"


@router.post("", response_model=ModelUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_model(
    file: UploadFile = File(...),
    owner_id: str = Depends(get_current_user),
) -> ModelUploadResponse:
    """Upload and validate a model file (supports pickle, ONNX, boosting formats)."""
    from ..model_validation import validate_model, ACCEPTED_EXTENSIONS
    from pathlib import PurePosixPath

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File must have a filename",
        )

    ext = PurePosixPath(file.filename).suffix.lower()
    if ext not in ACCEPTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported extension: {ext}. Accepted: {', '.join(sorted(ACCEPTED_EXTENSIONS.keys()))}",
        )

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit",
        )

    # Extract task_type from query or default
    # For now, try to auto-detect from validation. Default to binary.
    result = validate_model(raw, ext, "binary_classification")

    # If binary validation fails with n_classes error, retry as multiclass
    if not result.valid and result.error and "Binary classification requires 2 classes" in result.error:
        result = validate_model(raw, ext, "multiclass_classification")

    if not result.valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result.error or "Validation failed",
        )

    model_id = hashlib.sha256(raw).hexdigest()
    settings = get_settings()
    now = datetime.now(timezone.utc).isoformat()

    # Store with original extension (not just .pkl)
    storage_suffix = ext or ".pkl"

    meta = {
        "model_id": model_id,
        "original_filename": file.filename,
        "model_class": result.model_class,
        "task_type": result.task_type,
        "n_classes": result.n_classes,
        "capabilities": result.capabilities,
        "file_size_bytes": len(raw),
        "uploaded_at": now,
        "owner_id": owner_id,
    }

    if settings.is_hosted:
        backend = get_storage_backend()
        file_key = _model_key(owner_id, model_id, storage_suffix)
        meta_key = _model_key(owner_id, model_id, ".meta.json")
        backend.upload(file_key, raw, content_type="application/octet-stream")
        backend.upload(
            meta_key,
            json.dumps(meta, indent=2).encode("utf-8"),
            content_type="application/json",
        )
    else:
        local_dir = _models_dir_local(owner_id)
        file_path = local_dir / f"{model_id}{storage_suffix}"
        meta_path = local_dir / f"{model_id}.meta.json"
        file_path.write_bytes(raw)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return ModelUploadResponse(
        model_id=model_id,
        original_filename=meta["original_filename"],
        model_class=result.model_class,
        task_type=result.task_type,
        n_classes=result.n_classes,
        capabilities=result.capabilities,
        file_size_bytes=len(raw),
    )


@router.get("", response_model=list[ModelMeta])
async def list_models(
    owner_id: str = Depends(get_current_user),
) -> list[ModelMeta]:
    """List uploaded models for the current user."""
    settings = get_settings()
    if settings.is_hosted:
        backend = get_storage_backend()
        prefix = f"models/{owner_id}/"
        keys = backend.list_keys(prefix)
        meta_list = []
        for key in keys:
            if not key.endswith(".meta.json"):
                continue
            try:
                data = backend.download(key)
                meta = json.loads(data.decode("utf-8"))
                meta_list.append(
                    ModelMeta(
                        model_id=meta["model_id"],
                        original_filename=meta["original_filename"],
                        model_class=meta["model_class"],
                        capabilities=meta.get("capabilities", {}),
                        file_size_bytes=meta["file_size_bytes"],
                        uploaded_at=datetime.fromisoformat(meta["uploaded_at"].replace("Z", "+00:00")),
                    )
                )
            except Exception:
                continue
        return sorted(meta_list, key=lambda m: m.uploaded_at, reverse=True)

    local_dir = _models_dir_local(owner_id)
    meta_list = []
    for meta_path in local_dir.glob("*.meta.json"):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            meta_list.append(
                ModelMeta(
                    model_id=data["model_id"],
                    original_filename=data["original_filename"],
                    model_class=data["model_class"],
                    capabilities=data.get("capabilities", {}),
                    file_size_bytes=data["file_size_bytes"],
                    uploaded_at=datetime.fromisoformat(data["uploaded_at"].replace("Z", "+00:00")),
                )
            )
        except Exception:
            continue
    return sorted(meta_list, key=lambda m: m.uploaded_at, reverse=True)


@router.get("/{model_id}", response_model=ModelMeta)
async def get_model(
    model_id: str,
    owner_id: str = Depends(get_current_user),
) -> ModelMeta:
    """Get metadata for a single uploaded model."""
    settings = get_settings()
    if settings.is_hosted:
        backend = get_storage_backend()
        meta_key = _model_key(owner_id, model_id, ".meta.json")
        if not backend.exists(meta_key):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found",
            )
        data = json.loads(backend.download(meta_key).decode("utf-8"))
        return ModelMeta(
            model_id=data["model_id"],
            original_filename=data["original_filename"],
            model_class=data["model_class"],
            capabilities=data.get("capabilities", {}),
            file_size_bytes=data["file_size_bytes"],
            uploaded_at=datetime.fromisoformat(data["uploaded_at"].replace("Z", "+00:00")),
        )

    local_dir = _models_dir_local(owner_id)
    meta_path = local_dir / f"{model_id}.meta.json"
    if not meta_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    return ModelMeta(
        model_id=data["model_id"],
        original_filename=data["original_filename"],
        model_class=data["model_class"],
        capabilities=data.get("capabilities", {}),
        file_size_bytes=data["file_size_bytes"],
        uploaded_at=datetime.fromisoformat(data["uploaded_at"].replace("Z", "+00:00")),
    )
