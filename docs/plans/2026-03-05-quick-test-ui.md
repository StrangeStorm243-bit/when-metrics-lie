# Quick Test UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Simplify the Spectra web UI so users can upload a model + dataset CSV and get automatic stress-testing with one click, without needing to understand metrics, stress suites, or column configuration.

**Architecture:** New `/quick` frontend page with 2-file drag-and-drop. New `POST /api/datasets` backend endpoint for CSV upload + column detection. New `POST /api/auto-detect` endpoint that analyzes uploaded model metadata + CSV headers to recommend task type, columns, metric, and stress suite. Existing `/new` page stays as "Advanced Mode". Home page updated to point to `/quick` as primary entry.

**Tech Stack:** Next.js 14 (App Router), React 18, Tailwind CSS, FastAPI, Pandas (CSV header sniffing), existing Spectra engine bridge

---

## Context

### Current UX Problem
The `/new` page requires users to manually select: task type, experiment name, metric, stress suite, dataset (from presets only), model upload, feature columns, y_true column, y_score column, threshold, and notes. This is 10+ fields that require ML expertise.

### Proposed Quick Test Flow
1. User drops/selects a model file (.pkl, .onnx, etc.)
2. User drops/selects a CSV dataset file
3. Backend auto-detects: task type (from model), CSV columns (y_true, y_score, features), best metric, best stress suite
4. User sees detected config with option to override
5. One click "Run Stress Test" -> redirect to results

### Key Files (Read Before Starting)
- `web/frontend/app/new/page.tsx` — Current advanced form (keep as-is)
- `web/frontend/app/page.tsx` — Home page (update links)
- `web/frontend/lib/api.ts` — API client (add new endpoints)
- `web/backend/app/contracts.py` — Pydantic schemas (add new contracts)
- `web/backend/app/routers/models.py` — Model upload (already auto-detects task type)
- `web/backend/app/routers/experiments.py` — Experiment CRUD + run
- `web/backend/app/engine_bridge.py` — Engine bridge (builds spec_dict)
- `web/backend/app/storage.py` — Metric/stress suite presets
- `web/backend/app/main.py` — FastAPI app (register new router)

---

### Task 1: Dataset Upload Backend Endpoint

**Files:**
- Create: `web/backend/app/routers/datasets.py`
- Modify: `web/backend/app/contracts.py` (add DatasetUploadResponse)
- Modify: `web/backend/app/main.py` (register router)
- Test: `tests/test_web_dataset_upload.py`

**Step 1: Write the failing test**

Create `tests/test_web_dataset_upload.py`:

```python
"""Tests for dataset upload endpoint."""
from __future__ import annotations

import io
import pytest


@pytest.fixture
def csv_bytes():
    """Simple CSV with y_true, y_score, and a feature column."""
    return b"y_true,y_score,age,income\n1,0.9,25,50000\n0,0.1,30,60000\n1,0.7,35,70000\n"


@pytest.fixture
def csv_no_score():
    """CSV with y_true but no y_score — should detect best candidate."""
    return b"label,prediction,feat1,feat2\n1,0.9,1.0,2.0\n0,0.2,3.0,4.0\n"


def test_upload_csv_returns_columns(csv_bytes):
    """Upload a CSV and get back detected columns."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from web.backend.app.main import app

    client = TestClient(app)
    resp = client.post(
        "/datasets",
        files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["dataset_id"]
    assert data["original_filename"] == "test.csv"
    assert "y_true" in data["columns"]
    assert "y_score" in data["columns"]
    assert data["n_rows"] == 3
    assert data["detected_y_true_col"] == "y_true"
    assert data["detected_y_score_col"] == "y_score"
    assert "age" in data["detected_feature_cols"]
    assert "income" in data["detected_feature_cols"]


def test_upload_csv_detects_label_prediction_columns(csv_no_score):
    """Detect label/prediction columns from common naming patterns."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from web.backend.app.main import app

    client = TestClient(app)
    resp = client.post(
        "/datasets",
        files={"file": ("data.csv", io.BytesIO(csv_no_score), "text/csv")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["detected_y_true_col"] == "label"
    assert data["detected_y_score_col"] == "prediction"


def test_upload_rejects_non_csv():
    """Reject non-CSV files."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from web.backend.app.main import app

    client = TestClient(app)
    resp = client.post(
        "/datasets",
        files={"file": ("test.json", io.BytesIO(b'{"a":1}'), "application/json")},
    )
    assert resp.status_code == 422
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_web_dataset_upload.py -v`
Expected: FAIL (404 — no /datasets endpoint)

**Step 3: Add DatasetUploadResponse contract**

In `web/backend/app/contracts.py`, add after `SupportedFormat`:

```python
class DatasetUploadResponse(BaseModel):
    """Response after successful dataset CSV upload."""

    dataset_id: str = Field(..., description="Content-addressable dataset ID (SHA256)")
    original_filename: str = Field(..., description="Original file name")
    columns: list[str] = Field(..., description="All column names in the CSV")
    n_rows: int = Field(..., description="Number of data rows")
    detected_y_true_col: Optional[str] = Field(None, description="Auto-detected ground truth column")
    detected_y_score_col: Optional[str] = Field(None, description="Auto-detected score/prediction column")
    detected_feature_cols: list[str] = Field(default_factory=list, description="Auto-detected feature columns")
```

**Step 4: Create the datasets router**

Create `web/backend/app/routers/datasets.py`:

```python
"""Datasets API router: upload CSV datasets with column auto-detection."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, status, UploadFile

from ..auth import get_current_user
from ..config import get_settings
from ..contracts import DatasetUploadResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

# Column name patterns for auto-detection (ordered by priority)
Y_TRUE_PATTERNS = ["y_true", "label", "target", "ground_truth", "true_label", "class", "y"]
Y_SCORE_PATTERNS = ["y_score", "prediction", "pred", "probability", "prob", "score", "y_pred", "predicted"]


def _detect_columns(columns: list[str]) -> tuple[str | None, str | None, list[str]]:
    """Auto-detect y_true, y_score, and feature columns from CSV headers.

    Returns:
        (y_true_col, y_score_col, feature_cols)
    """
    cols_lower = {c.lower(): c for c in columns}

    y_true_col = None
    for pattern in Y_TRUE_PATTERNS:
        if pattern in cols_lower:
            y_true_col = cols_lower[pattern]
            break

    y_score_col = None
    for pattern in Y_SCORE_PATTERNS:
        if pattern in cols_lower:
            y_score_col = cols_lower[pattern]
            break

    # Feature cols = everything except y_true, y_score, and common non-feature cols
    exclude = {y_true_col, y_score_col, None}
    non_feature_patterns = {"id", "index", "row", "unnamed", "group", "subgroup"}
    feature_cols = [
        c for c in columns
        if c not in exclude and c.lower() not in non_feature_patterns
    ]

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


@router.post("", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    owner_id: str = Depends(get_current_user),
) -> DatasetUploadResponse:
    """Upload a CSV dataset file and auto-detect columns."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File must have a filename",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only CSV files are accepted",
        )

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit",
        )

    # Parse CSV to get columns and row count
    import io
    try:
        df = pd.read_csv(io.BytesIO(raw), nrows=0)  # just headers
        columns = list(df.columns)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse CSV: {e}",
        )

    # Count rows (without loading all data into memory for large files)
    try:
        df_count = pd.read_csv(io.BytesIO(raw))
        n_rows = len(df_count)
    except Exception:
        n_rows = 0

    # Auto-detect columns
    y_true_col, y_score_col, feature_cols = _detect_columns(columns)

    # Content-addressable ID
    dataset_id = hashlib.sha256(raw).hexdigest()

    # Store locally
    local_dir = _datasets_dir_local(owner_id)
    file_path = local_dir / f"{dataset_id}.csv"
    meta_path = local_dir / f"{dataset_id}.meta.json"
    file_path.write_bytes(raw)

    meta = {
        "dataset_id": dataset_id,
        "original_filename": file.filename,
        "columns": columns,
        "n_rows": n_rows,
        "detected_y_true_col": y_true_col,
        "detected_y_score_col": y_score_col,
        "detected_feature_cols": feature_cols,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "owner_id": owner_id,
    }
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
```

**Step 5: Register the router in main.py**

In `web/backend/app/main.py`, add:

```python
from .routers import datasets
app.include_router(datasets.router)
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/test_web_dataset_upload.py -v`
Expected: PASS (3/3)

**Step 7: Commit**

```bash
git add web/backend/app/routers/datasets.py web/backend/app/contracts.py web/backend/app/main.py tests/test_web_dataset_upload.py
git commit -m "feat: dataset upload endpoint with CSV column auto-detection"
```

---

### Task 2: Auto-Detect Backend Endpoint

**Files:**
- Create: `web/backend/app/routers/auto_detect.py`
- Modify: `web/backend/app/contracts.py` (add AutoDetectRequest/Response)
- Modify: `web/backend/app/main.py` (register router)
- Test: `tests/test_web_auto_detect.py`

**Step 1: Write the failing test**

Create `tests/test_web_auto_detect.py`:

```python
"""Tests for auto-detect endpoint."""
from __future__ import annotations

import io
import pytest


def _upload_fixtures(client):
    """Upload a model and dataset, return their IDs."""
    # Upload model (use the demo model from data/)
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    model_path = repo_root / "data" / "demo_model.pkl"
    if not model_path.exists():
        pytest.skip("demo_model.pkl not found")

    with open(model_path, "rb") as f:
        model_resp = client.post("/models", files={"file": ("model.pkl", f, "application/octet-stream")})
    assert model_resp.status_code == 201
    model_id = model_resp.json()["model_id"]

    # Upload CSV dataset
    csv_data = b"y_true,y_score,feature1,feature2\n1,0.9,1.0,2.0\n0,0.1,3.0,4.0\n1,0.7,5.0,6.0\n"
    ds_resp = client.post("/datasets", files={"file": ("test.csv", io.BytesIO(csv_data), "text/csv")})
    assert ds_resp.status_code == 201
    dataset_id = ds_resp.json()["dataset_id"]

    return model_id, dataset_id


def test_auto_detect_returns_config():
    """Auto-detect returns recommended configuration."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from web.backend.app.main import app

    client = TestClient(app)
    model_id, dataset_id = _upload_fixtures(client)

    resp = client.post("/auto-detect", json={"model_id": model_id, "dataset_id": dataset_id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_type"] in ["binary_classification", "multiclass_classification", "regression", "ranking"]
    assert data["y_true_col"]
    assert data["y_score_col"]
    assert data["recommended_metric"]
    assert data["recommended_stress_suite"]
    assert isinstance(data["feature_cols"], list)


def test_auto_detect_without_model():
    """Auto-detect with only a dataset still returns column detection."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from web.backend.app.main import app

    client = TestClient(app)
    csv_data = b"y_true,y_score,f1\n1,0.9,1.0\n0,0.1,2.0\n"
    ds_resp = client.post("/datasets", files={"file": ("test.csv", io.BytesIO(csv_data), "text/csv")})
    dataset_id = ds_resp.json()["dataset_id"]

    resp = client.post("/auto-detect", json={"dataset_id": dataset_id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["y_true_col"] == "y_true"
    assert data["y_score_col"] == "y_score"
    assert data["task_type"] == "binary_classification"  # inferred from y_true values
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_web_auto_detect.py -v`
Expected: FAIL (404 — no /auto-detect endpoint)

**Step 3: Add AutoDetect contracts**

In `web/backend/app/contracts.py`, add:

```python
class AutoDetectRequest(BaseModel):
    """Request to auto-detect experiment configuration."""

    model_id: Optional[str] = Field(None, description="Uploaded model ID (from /models)")
    dataset_id: str = Field(..., description="Uploaded dataset ID (from /datasets)")


class AutoDetectResponse(BaseModel):
    """Auto-detected experiment configuration."""

    task_type: str = Field(..., description="Detected task type")
    y_true_col: Optional[str] = Field(None, description="Detected ground truth column")
    y_score_col: Optional[str] = Field(None, description="Detected score/prediction column")
    feature_cols: list[str] = Field(default_factory=list, description="Detected feature columns")
    recommended_metric: str = Field(..., description="Recommended primary metric")
    recommended_stress_suite: str = Field(..., description="Recommended stress suite")
    n_rows: int = Field(0, description="Dataset row count")
    model_class: Optional[str] = Field(None, description="Model class name if model provided")
    confidence: str = Field("high", description="Detection confidence: high, medium, low")
```

**Step 4: Create the auto-detect router**

Create `web/backend/app/routers/auto_detect.py`:

```python
"""Auto-detect API router: analyze model + dataset and recommend configuration."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import get_current_user
from ..contracts import AutoDetectRequest, AutoDetectResponse

router = APIRouter(tags=["auto-detect"])

# Default metric recommendations per task type
DEFAULT_METRICS = {
    "binary_classification": "auc",
    "multiclass_classification": "weighted_f1",
    "regression": "rmse",
    "ranking": "auc",
}

DEFAULT_SUITES = {
    "binary_classification": "balanced",
    "multiclass_classification": "balanced",
    "regression": "performance_first",
    "ranking": "performance_first",
}


def _repo_root() -> Path:
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
    """Load model metadata from local storage. Returns None if not found."""
    root = _repo_root()
    meta_path = root / ".spectra_ui" / "models" / owner_id / f"{model_id}.meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _infer_task_type_from_dataset(ds_meta: dict) -> str:
    """Infer task type from dataset column patterns and values."""
    y_true_col = ds_meta.get("detected_y_true_col")
    if not y_true_col:
        return "binary_classification"  # safe default

    # Load a sample of the data to check unique values
    import io
    import pandas as pd

    root = _repo_root()
    csv_path = root / ".spectra_ui" / "datasets" / ds_meta["owner_id"] / f"{ds_meta['dataset_id']}.csv"
    if not csv_path.exists():
        return "binary_classification"

    try:
        df = pd.read_csv(csv_path, nrows=1000)
        if y_true_col not in df.columns:
            return "binary_classification"

        unique_vals = df[y_true_col].dropna().unique()
        n_unique = len(unique_vals)

        # If all values are 0/1, it's binary
        if n_unique == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
            return "binary_classification"
        # If small number of distinct integers, multiclass
        if n_unique <= 20 and all(float(v).is_integer() for v in unique_vals if pd.notna(v)):
            if n_unique > 2:
                return "multiclass_classification"
            return "binary_classification"
        # Otherwise regression
        return "regression"
    except Exception:
        return "binary_classification"


@router.post("/auto-detect", response_model=AutoDetectResponse)
async def auto_detect(
    req: AutoDetectRequest,
    owner_id: str = Depends(get_current_user),
) -> AutoDetectResponse:
    """Analyze uploaded model + dataset and recommend experiment configuration."""
    # Load dataset metadata
    ds_meta = _load_dataset_meta(req.dataset_id, owner_id)

    # Load model metadata (optional)
    model_meta = None
    if req.model_id:
        model_meta = _load_model_meta(req.model_id, owner_id)
        if model_meta is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {req.model_id} not found",
            )

    # Determine task type: prefer model metadata, fall back to dataset inference
    if model_meta and model_meta.get("task_type"):
        task_type = model_meta["task_type"]
        confidence = "high"
    else:
        task_type = _infer_task_type_from_dataset(ds_meta)
        confidence = "medium"

    # Column detection comes from dataset upload
    y_true_col = ds_meta.get("detected_y_true_col")
    y_score_col = ds_meta.get("detected_y_score_col")
    feature_cols = ds_meta.get("detected_feature_cols", [])

    # Recommend metric and stress suite
    recommended_metric = DEFAULT_METRICS.get(task_type, "auc")
    recommended_suite = DEFAULT_SUITES.get(task_type, "balanced")

    return AutoDetectResponse(
        task_type=task_type,
        y_true_col=y_true_col,
        y_score_col=y_score_col,
        feature_cols=feature_cols,
        recommended_metric=recommended_metric,
        recommended_stress_suite=recommended_suite,
        n_rows=ds_meta.get("n_rows", 0),
        model_class=model_meta.get("model_class") if model_meta else None,
        confidence=confidence,
    )
```

**Step 5: Register the router in main.py**

In `web/backend/app/main.py`, add:

```python
from .routers import auto_detect
app.include_router(auto_detect.router)
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/test_web_auto_detect.py -v`
Expected: PASS (2/2)

**Step 7: Commit**

```bash
git add web/backend/app/routers/auto_detect.py web/backend/app/contracts.py web/backend/app/main.py tests/test_web_auto_detect.py
git commit -m "feat: auto-detect endpoint for model + dataset configuration"
```

---

### Task 3: Frontend API Client Extensions

**Files:**
- Modify: `web/frontend/lib/api.ts` (add new types and functions)
- No test file (TypeScript types verified by compilation)

**Step 1: Add TypeScript types and API functions**

In `web/frontend/lib/api.ts`, add after the `getModelFormats` function:

```typescript
// ---------------------------------------------------------------------------
// Dataset upload
// ---------------------------------------------------------------------------

export interface DatasetUploadResponse {
  dataset_id: string;
  original_filename: string;
  columns: string[];
  n_rows: number;
  detected_y_true_col: string | null;
  detected_y_score_col: string | null;
  detected_feature_cols: string[];
}

/**
 * Upload a CSV dataset file.
 */
export async function uploadDataset(file: File): Promise<DatasetUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  let authToken: string | null = null;
  if (_tokenProvider) {
    try {
      authToken = await _tokenProvider();
    } catch {
      // ignore
    }
  }

  const headers: Record<string, string> = {};
  if (authToken) {
    headers["Authorization"] = `Bearer ${authToken}`;
  }

  const url = `${API_BASE}/datasets`;
  const response = await fetch(url, {
    method: "POST",
    headers,
    body: formData,
  });

  if (!response.ok) {
    let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
    try {
      const errorData = await response.json();
      if (errorData.detail) {
        errorMessage = typeof errorData.detail === "string" ? errorData.detail : JSON.stringify(errorData.detail);
      }
    } catch {
      // ignore
    }
    throw new ApiError(errorMessage, response.status);
  }

  return response.json();
}

// ---------------------------------------------------------------------------
// Auto-detect
// ---------------------------------------------------------------------------

export interface AutoDetectRequest {
  model_id?: string | null;
  dataset_id: string;
}

export interface AutoDetectResponse {
  task_type: string;
  y_true_col: string | null;
  y_score_col: string | null;
  feature_cols: string[];
  recommended_metric: string;
  recommended_stress_suite: string;
  n_rows: number;
  model_class: string | null;
  confidence: "high" | "medium" | "low";
}

/**
 * Auto-detect experiment configuration from uploaded model + dataset.
 */
export async function autoDetect(req: AutoDetectRequest): Promise<AutoDetectResponse> {
  return apiFetch<AutoDetectResponse>("/auto-detect", {
    method: "POST",
    body: JSON.stringify(req),
  });
}
```

**Step 2: Verify TypeScript compiles**

Run: `cd web/frontend && npx tsc --noEmit --pretty 2>&1 | head -20`
Expected: No errors related to api.ts

**Step 3: Commit**

```bash
git add web/frontend/lib/api.ts
git commit -m "feat: frontend API client for dataset upload and auto-detect"
```

---

### Task 4: Quick Test Frontend Page

**Files:**
- Create: `web/frontend/app/quick/page.tsx`
- No test file (UI tested manually + by backend integration tests)

**Step 1: Create the Quick Test page**

Create `web/frontend/app/quick/page.tsx`:

```tsx
"use client";

import { useRouter } from "next/navigation";
import { useState, useCallback } from "react";
import {
  uploadModel,
  uploadDataset,
  autoDetect,
  createExperiment,
  runExperiment,
  ApiError,
  type ModelUploadResponse,
  type DatasetUploadResponse,
  type AutoDetectResponse,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import Link from "next/link";

type Step = "upload" | "review" | "running";

export default function QuickTestPage() {
  const router = useRouter();
  const [step, setStep] = useState<Step>("upload");

  // Upload state
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [modelMeta, setModelMeta] = useState<ModelUploadResponse | null>(null);
  const [datasetMeta, setDatasetMeta] = useState<DatasetUploadResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Detection state
  const [detection, setDetection] = useState<AutoDetectResponse | null>(null);

  // Run state
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  const handleModelSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setModelFile(file);
  }, []);

  const handleDatasetSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setDatasetFile(file);
  }, []);

  async function handleUploadAndDetect() {
    if (!datasetFile) {
      setUploadError("Please select a dataset CSV file");
      return;
    }
    setUploading(true);
    setUploadError(null);

    try {
      // Upload files in parallel
      const [dsResult, modelResult] = await Promise.all([
        uploadDataset(datasetFile),
        modelFile ? uploadModel(modelFile) : Promise.resolve(null),
      ]);

      setDatasetMeta(dsResult);
      if (modelResult) setModelMeta(modelResult);

      // Auto-detect configuration
      const detectResult = await autoDetect({
        model_id: modelResult?.model_id ?? undefined,
        dataset_id: dsResult.dataset_id,
      });

      setDetection(detectResult);
      setStep("review");
    } catch (err) {
      setUploadError(err instanceof ApiError ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function handleRunStressTest() {
    if (!detection || !datasetMeta) return;
    setRunning(true);
    setRunError(null);

    try {
      const config: Record<string, unknown> = {
        task_type: detection.task_type,
        y_true_col: detection.y_true_col || "y_true",
        y_score_col: detection.y_score_col || "y_score",
      };

      // Use uploaded dataset
      const repoRoot = ""; // Backend resolves relative to .spectra_ui
      config.dataset_id = datasetMeta.dataset_id;

      if (modelMeta) {
        config.model_id = modelMeta.model_id;
        config.feature_cols = detection.feature_cols.length > 0
          ? detection.feature_cols
          : [detection.y_score_col || "y_score"];
      }

      const experiment = await createExperiment({
        name: `Quick Test - ${datasetMeta.original_filename}`,
        metric_id: detection.recommended_metric,
        stress_suite_id: detection.recommended_stress_suite,
        config,
      });

      await runExperiment(experiment.id);
      router.push(`/experiments/${experiment.id}`);
    } catch (err) {
      setRunError(err instanceof ApiError ? err.message : "Run failed");
      setRunning(false);
    }
  }

  return (
    <div className="container max-w-2xl py-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold mb-1">Quick Stress Test</h1>
          <p className="text-sm text-muted-foreground">
            Upload your model and dataset — Spectra handles the rest.
          </p>
        </div>
        <Link href="/new" className="text-sm text-muted-foreground hover:text-foreground underline">
          Advanced Mode
        </Link>
      </div>

      {/* Step 1: Upload */}
      {step === "upload" && (
        <Card>
          <CardHeader>
            <CardTitle>Upload Files</CardTitle>
            <CardDescription>
              Drop your model file and dataset CSV. Spectra will auto-detect everything.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {uploadError && (
              <div className="p-4 bg-destructive/10 text-destructive border border-destructive/20 rounded-md text-sm">
                {uploadError}
              </div>
            )}

            <div className="space-y-2">
              <Label>Dataset CSV <span className="text-destructive">*</span></Label>
              <Input
                type="file"
                accept=".csv"
                onChange={handleDatasetSelect}
                disabled={uploading}
              />
              {datasetFile && (
                <p className="text-sm text-muted-foreground">Selected: {datasetFile.name}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label>Model File (optional)</Label>
              <p className="text-xs text-muted-foreground">
                Supports: sklearn (.pkl, .joblib), ONNX (.onnx), XGBoost (.ubj, .xgb), LightGBM (.lgb), CatBoost (.cbm)
              </p>
              <Input
                type="file"
                accept=".pkl,.joblib,.onnx,.ubj,.xgb,.lgb,.cbm"
                onChange={handleModelSelect}
                disabled={uploading}
              />
              {modelFile && (
                <p className="text-sm text-muted-foreground">Selected: {modelFile.name}</p>
              )}
            </div>

            <Button
              onClick={handleUploadAndDetect}
              disabled={uploading || !datasetFile}
              className="w-full"
            >
              {uploading ? "Uploading & Analyzing..." : "Upload & Analyze"}
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Step 2: Review detected config */}
      {step === "review" && detection && (
        <Card>
          <CardHeader>
            <CardTitle>Detected Configuration</CardTitle>
            <CardDescription>
              Review what Spectra detected. Click Run to start the stress test.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {runError && (
              <div className="p-4 bg-destructive/10 text-destructive border border-destructive/20 rounded-md text-sm">
                {runError}
              </div>
            )}

            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium text-muted-foreground">Task Type</span>
                <p>{detection.task_type.replace(/_/g, " ")}</p>
              </div>
              <div>
                <span className="font-medium text-muted-foreground">Confidence</span>
                <p className={detection.confidence === "high" ? "text-green-600" : "text-yellow-600"}>
                  {detection.confidence}
                </p>
              </div>
              <div>
                <span className="font-medium text-muted-foreground">Ground Truth Column</span>
                <p className="font-mono">{detection.y_true_col || "—"}</p>
              </div>
              <div>
                <span className="font-medium text-muted-foreground">Score Column</span>
                <p className="font-mono">{detection.y_score_col || "—"}</p>
              </div>
              <div>
                <span className="font-medium text-muted-foreground">Metric</span>
                <p>{detection.recommended_metric}</p>
              </div>
              <div>
                <span className="font-medium text-muted-foreground">Stress Suite</span>
                <p>{detection.recommended_stress_suite}</p>
              </div>
              {detection.model_class && (
                <div>
                  <span className="font-medium text-muted-foreground">Model</span>
                  <p>{detection.model_class}</p>
                </div>
              )}
              <div>
                <span className="font-medium text-muted-foreground">Dataset Rows</span>
                <p>{detection.n_rows.toLocaleString()}</p>
              </div>
            </div>

            {detection.feature_cols.length > 0 && (
              <div className="text-sm">
                <span className="font-medium text-muted-foreground">Features</span>
                <p className="font-mono text-xs mt-1">
                  {detection.feature_cols.join(", ")}
                </p>
              </div>
            )}

            <div className="flex gap-2 pt-2">
              <Button
                variant="outline"
                onClick={() => setStep("upload")}
                disabled={running}
              >
                Back
              </Button>
              <Button
                onClick={handleRunStressTest}
                disabled={running}
                className="flex-1"
              >
                {running ? "Running Stress Test..." : "Run Stress Test"}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
```

**Step 2: Verify page compiles**

Run: `cd web/frontend && npx tsc --noEmit --pretty 2>&1 | head -20`
Expected: No errors

**Step 3: Commit**

```bash
git add web/frontend/app/quick/page.tsx
git commit -m "feat: Quick Test page with 2-file upload and auto-detection"
```

---

### Task 5: Wire Dataset Path in Engine Bridge

The engine bridge currently resolves datasets from `data/` directory or config paths. We need it to also resolve uploaded datasets from `.spectra_ui/datasets/`.

**Files:**
- Modify: `web/backend/app/engine_bridge.py` (add dataset_id resolution)
- Test: `tests/test_web_dataset_upload.py` (add integration test)

**Step 1: Write the failing test**

Add to `tests/test_web_dataset_upload.py`:

```python
def test_engine_bridge_resolves_dataset_id():
    """Engine bridge should resolve dataset_id to the uploaded CSV path."""
    pytest.importorskip("fastapi")
    from unittest.mock import patch
    from web.backend.app.engine_bridge import _get_dataset_path
    from web.backend.app.contracts import ExperimentCreateRequest

    # Create a fake uploaded dataset
    from pathlib import Path
    from web.backend.app.routers.datasets import _repo_root, _datasets_dir_local
    local_dir = _datasets_dir_local("anonymous")
    fake_id = "abc123"
    csv_path = local_dir / f"{fake_id}.csv"
    csv_path.write_text("y_true,y_score\n1,0.9\n0,0.1\n")

    try:
        req = ExperimentCreateRequest(
            name="test",
            metric_id="auc",
            stress_suite_id="balanced",
            config={"dataset_id": fake_id},
        )
        result = _get_dataset_path(req, owner_id="anonymous")
        assert result == csv_path.resolve()
    finally:
        csv_path.unlink(missing_ok=True)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_web_dataset_upload.py::test_engine_bridge_resolves_dataset_id -v`
Expected: FAIL (TypeError — _get_dataset_path doesn't accept owner_id)

**Step 3: Update engine_bridge.py to resolve dataset_id**

Modify `_get_dataset_path` in `web/backend/app/engine_bridge.py` to accept an `owner_id` parameter and check for `dataset_id` in config before `dataset_path`:

```python
def _get_dataset_path(create_req: ExperimentCreateRequest, owner_id: str = "anonymous") -> Path:
    """
    Determine dataset path using config or candidate fallbacks.
    """
    repo_root = _find_repo_root()
    searched_locations = []

    # Check dataset_id first (uploaded datasets)
    if "dataset_id" in create_req.config:
        dataset_id = create_req.config["dataset_id"]
        ds_path = repo_root / ".spectra_ui" / "datasets" / owner_id / f"{dataset_id}.csv"
        searched_locations.append(f"config['dataset_id']: {ds_path}")
        if ds_path.exists() and ds_path.is_file():
            return ds_path.resolve()
        raise ValueError(
            f"Uploaded dataset {dataset_id} not found at {ds_path}"
        )

    # ... rest of existing logic unchanged ...
```

Also update `_get_default_dataset` and `run_experiment` to pass `owner_id` through:

In `_get_default_dataset`, change signature to accept `owner_id`:
```python
def _get_default_dataset(create_req: ExperimentCreateRequest, task_type: str = "binary_classification", owner_id: str = "anonymous") -> dict:
    dataset_path = _get_dataset_path(create_req, owner_id=owner_id)
    # ... rest unchanged
```

In `run_experiment`, pass `owner_id` to `_get_default_dataset`:
```python
dataset_dict = _get_default_dataset(create_req, task_type, owner_id=owner_id)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_web_dataset_upload.py -v`
Expected: PASS (all tests)

**Step 5: Run existing tests to verify no regressions**

Run: `pytest tests/ -x -q`
Expected: All existing tests still pass

**Step 6: Commit**

```bash
git add web/backend/app/engine_bridge.py tests/test_web_dataset_upload.py
git commit -m "feat: engine bridge resolves uploaded dataset_id from .spectra_ui"
```

---

### Task 6: Update Home Page Navigation

**Files:**
- Modify: `web/frontend/app/page.tsx` (update primary CTA to `/quick`)

**Step 1: Update the home page**

In `web/frontend/app/page.tsx`, change the primary "New Experiment" link to point to `/quick`, and add a secondary "Advanced" link:

Replace the existing header div (lines 29-49) with:

```tsx
<div
  style={{
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "2rem",
  }}
>
  <h1>Experiments</h1>
  <div style={{ display: "flex", gap: "0.5rem" }}>
    <Link
      href="/quick"
      style={{
        padding: "0.5rem 1rem",
        backgroundColor: "#0070f3",
        color: "white",
        borderRadius: "4px",
        textDecoration: "none",
      }}
    >
      Quick Test
    </Link>
    <Link
      href="/new"
      style={{
        padding: "0.5rem 1rem",
        backgroundColor: "transparent",
        color: "#0070f3",
        borderRadius: "4px",
        textDecoration: "none",
        border: "1px solid #0070f3",
      }}
    >
      Advanced
    </Link>
  </div>
</div>
```

Also update the empty-state text (around line 74):

```tsx
No experiments yet.{" "}
<Link href="/quick" style={{ color: "#0070f3" }}>
  Run a Quick Test
</Link>{" "}
to get started.
```

**Step 2: Verify page compiles**

Run: `cd web/frontend && npx tsc --noEmit --pretty 2>&1 | head -20`
Expected: No errors

**Step 3: Commit**

```bash
git add web/frontend/app/page.tsx
git commit -m "feat: update home page with Quick Test as primary CTA"
```

---

### Task 7: End-to-End Integration Test

**Files:**
- Create: `tests/test_web_quick_test_e2e.py`

**Step 1: Write the integration test**

Create `tests/test_web_quick_test_e2e.py`:

```python
"""End-to-end integration test for Quick Test flow."""
from __future__ import annotations

import io
from pathlib import Path

import pytest


@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from web.backend.app.main import app
    return TestClient(app)


@pytest.fixture
def demo_model_bytes():
    """Load demo model if available."""
    repo_root = Path(__file__).resolve().parent.parent
    model_path = repo_root / "data" / "demo_model.pkl"
    if not model_path.exists():
        pytest.skip("demo_model.pkl not found")
    return model_path.read_bytes()


@pytest.fixture
def demo_csv_bytes():
    """CSV compatible with the demo model."""
    repo_root = Path(__file__).resolve().parent.parent
    # Use the first CSV in data/ that has y_true and y_score
    data_dir = repo_root / "data"
    for csv_file in sorted(data_dir.glob("*.csv")):
        content = csv_file.read_text()
        if "y_true" in content and "y_score" in content:
            return csv_file.read_bytes()
    pytest.skip("No suitable demo CSV found")


def test_quick_test_full_flow(client, demo_csv_bytes):
    """Full Quick Test flow: upload dataset -> auto-detect -> create+run experiment."""
    # Step 1: Upload dataset
    ds_resp = client.post(
        "/datasets",
        files={"file": ("demo.csv", io.BytesIO(demo_csv_bytes), "text/csv")},
    )
    assert ds_resp.status_code == 201
    ds_data = ds_resp.json()
    dataset_id = ds_data["dataset_id"]
    assert ds_data["detected_y_true_col"] is not None

    # Step 2: Auto-detect
    detect_resp = client.post(
        "/auto-detect",
        json={"dataset_id": dataset_id},
    )
    assert detect_resp.status_code == 200
    detect_data = detect_resp.json()
    assert detect_data["task_type"]
    assert detect_data["recommended_metric"]
    assert detect_data["recommended_stress_suite"]

    # Step 3: Create experiment with detected config
    config = {
        "task_type": detect_data["task_type"],
        "y_true_col": detect_data["y_true_col"],
        "y_score_col": detect_data["y_score_col"],
        "dataset_id": dataset_id,
    }
    exp_resp = client.post(
        "/experiments",
        json={
            "name": "Quick Test - demo.csv",
            "metric_id": detect_data["recommended_metric"],
            "stress_suite_id": detect_data["recommended_stress_suite"],
            "config": config,
        },
    )
    assert exp_resp.status_code == 201
    experiment_id = exp_resp.json()["id"]

    # Step 4: Run experiment
    run_resp = client.post(
        f"/experiments/{experiment_id}/run",
        json={},
    )
    assert run_resp.status_code == 200
    assert run_resp.json()["status"] == "completed"


def test_auto_detect_column_patterns(client):
    """Verify auto-detect handles various column naming patterns."""
    # Test with 'label' and 'prediction' columns
    csv = b"label,prediction,age\n1,0.9,25\n0,0.1,30\n1,0.7,35\n"
    ds_resp = client.post(
        "/datasets",
        files={"file": ("data.csv", io.BytesIO(csv), "text/csv")},
    )
    assert ds_resp.status_code == 201
    ds = ds_resp.json()

    detect_resp = client.post(
        "/auto-detect",
        json={"dataset_id": ds["dataset_id"]},
    )
    assert detect_resp.status_code == 200
    data = detect_resp.json()
    assert data["y_true_col"] == "label"
    assert data["y_score_col"] == "prediction"
    assert "age" in data["feature_cols"]
```

**Step 2: Run integration test**

Run: `pytest tests/test_web_quick_test_e2e.py -v`
Expected: PASS (2/2) — the full flow works end-to-end

**Step 3: Run full test suite to verify no regressions**

Run: `pytest tests/ -x -q`
Expected: All tests pass

**Step 4: Commit**

```bash
git add tests/test_web_quick_test_e2e.py
git commit -m "test: end-to-end integration tests for Quick Test flow"
```

---

## Summary

| Task | Description | Files Created/Modified |
|------|-------------|----------------------|
| 1 | Dataset upload endpoint | `routers/datasets.py`, `contracts.py`, `main.py` |
| 2 | Auto-detect endpoint | `routers/auto_detect.py`, `contracts.py`, `main.py` |
| 3 | Frontend API client | `lib/api.ts` |
| 4 | Quick Test page | `app/quick/page.tsx` |
| 5 | Engine bridge dataset_id | `engine_bridge.py` |
| 6 | Home page navigation | `app/page.tsx` |
| 7 | E2E integration tests | `test_web_quick_test_e2e.py` |

**Total: 4 new files, 5 modified files, 3 test files**
