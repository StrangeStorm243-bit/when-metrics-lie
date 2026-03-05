# Spectra v1.0 Completion Plan — All Gaps

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close every gap identified in the v1.0 verification audit — security, adapters, metrics, scenarios, diagnostics, SDK, web, open-source infra — bringing the codebase to full DESIGN_PLAN.md compliance.

**Architecture:** 8 independent workstreams that can run in parallel worktrees. Each stream touches non-overlapping files. Streams are labeled T1–T8 for terminal assignment. Dependencies between streams are called out explicitly.

**Tech Stack:** Python 3.11+, sklearn, onnxruntime, torch, tensorflow, transformers, fairlearn, evidently, pluggy, typer, mkdocs-material, GitHub Actions

---

## Dependency Graph

```
T1 (Security)          — standalone, no deps
T2 (Adapters)          — standalone, no deps
T3 (Metrics)           — standalone, no deps
T4 (Scenarios)         — standalone, no deps
T5 (Diagnostics)       — standalone, no deps
T6 (SDK/CLI polish)    — after T3 metrics land (for list_metrics)
T7 (Web enhancements)  — after T3 metrics land (for dynamic lists)
T8 (Open-source infra) — standalone, no deps
```

Streams T1–T5 and T8 have ZERO cross-dependencies and can start simultaneously.
T6 and T7 can start immediately on non-metric tasks; metric-dependent parts wait for T3.

---

## T1: Security Pipeline

### Task T1.1: Model Security Scanner

**Files:**
- Create: `src/metrics_lie/model/security.py`
- Test: `tests/test_v1_security.py`
- Modify: `pyproject.toml` (add `security` optional group)

**Step 1: Add security dependency group to pyproject.toml**

Add after the `drift` group in `pyproject.toml`:

```toml
security = [
  "picklescan>=0.0.14",
]
```

Also add `"metrics_lie[security]"` to the `all` group.

**Step 2: Write failing tests**

```python
# tests/test_v1_security.py
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import pytest


def test_scan_safe_pickle_returns_no_issues():
    from metrics_lie.model.security import scan_model_file

    # Create a safe pickle (just a dict)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump({"weights": [1, 2, 3]}, f)
        path = f.name

    result = scan_model_file(path)
    assert result.is_safe is True
    assert result.issues == []
    Path(path).unlink()


def test_scan_onnx_returns_safe():
    from metrics_lie.model.security import scan_model_file

    # ONNX files are inherently safe (no code execution)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(b"\x08\x06")  # minimal proto bytes
        path = f.name

    result = scan_model_file(path)
    assert result.is_safe is True
    Path(path).unlink()


def test_scan_unknown_extension_returns_safe():
    from metrics_lie.model.security import scan_model_file

    with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as f:
        f.write(b"catboost")
        path = f.name

    result = scan_model_file(path)
    assert result.is_safe is True
    Path(path).unlink()


def test_scan_result_dataclass_fields():
    from metrics_lie.model.security import ScanResult

    r = ScanResult(is_safe=True, issues=[], format_detected="pickle")
    assert r.is_safe is True
    assert r.format_detected == "pickle"


def test_require_trust_pickle_flag():
    """Without trust_pickle=True, pickle loading should raise."""
    from metrics_lie.model.security import check_trust_policy

    with pytest.raises(ValueError, match="trust_pickle"):
        check_trust_policy("model.pkl", trust_pickle=False)


def test_trust_pickle_flag_allows():
    from metrics_lie.model.security import check_trust_policy

    # Should not raise
    check_trust_policy("model.pkl", trust_pickle=True)


def test_safe_formats_skip_trust_check():
    from metrics_lie.model.security import check_trust_policy

    # ONNX, boosting formats don't need trust_pickle
    check_trust_policy("model.onnx", trust_pickle=False)
    check_trust_policy("model.cbm", trust_pickle=False)
    check_trust_policy("model.ubj", trust_pickle=False)
```

**Step 3: Run tests to verify they fail**

Run: `pytest tests/test_v1_security.py -v`
Expected: FAIL (module not found)

**Step 4: Implement security module**

```python
# src/metrics_lie/model/security.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PICKLE_EXTENSIONS = {".pkl", ".pickle", ".joblib"}
SAFE_EXTENSIONS = {".onnx", ".ubj", ".xgb", ".lgb", ".cbm", ".safetensors"}


@dataclass(frozen=True)
class ScanResult:
    """Result of a model file security scan."""

    is_safe: bool
    issues: list[str] = field(default_factory=list)
    format_detected: str = "unknown"


def scan_model_file(path: str) -> ScanResult:
    """Scan a model file for security issues.

    For pickle files, uses picklescan if available. ONNX and boosting
    formats are considered inherently safe (no arbitrary code execution).
    """
    ext = Path(path).suffix.lower()

    if ext in SAFE_EXTENSIONS:
        return ScanResult(is_safe=True, issues=[], format_detected=ext.lstrip("."))

    if ext in PICKLE_EXTENSIONS:
        return _scan_pickle(path)

    # Unknown format — assume safe (no code execution path known)
    return ScanResult(is_safe=True, issues=[], format_detected="unknown")


def _scan_pickle(path: str) -> ScanResult:
    """Scan a pickle file using picklescan if available."""
    try:
        from picklescan.scanner import scan_file_path

        result = scan_file_path(path)
        issues = []
        if result.infected_count > 0:
            for scan in result.scans:
                if scan.issues:
                    for issue in scan.issues:
                        issues.append(str(issue))
        return ScanResult(
            is_safe=result.infected_count == 0,
            issues=issues,
            format_detected="pickle",
        )
    except ImportError:
        # picklescan not installed — warn but allow
        return ScanResult(
            is_safe=True,
            issues=["picklescan not installed; pickle not scanned"],
            format_detected="pickle",
        )


def check_trust_policy(path: str, *, trust_pickle: bool = False) -> None:
    """Enforce trust policy for model loading.

    Pickle files require explicit opt-in via trust_pickle=True.
    Safe formats (ONNX, boosting native) always pass.
    """
    ext = Path(path).suffix.lower()
    if ext in SAFE_EXTENSIONS:
        return
    if ext in PICKLE_EXTENSIONS and not trust_pickle:
        raise ValueError(
            f"Loading pickle files requires trust_pickle=True. "
            f"File: {path}. Pickle files can execute arbitrary code. "
            f"Use --trust-pickle on CLI or trust_pickle=True in SDK."
        )
```

**Step 5: Run tests**

Run: `pytest tests/test_v1_security.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/metrics_lie/model/security.py tests/test_v1_security.py pyproject.toml
git commit -m "feat: add model security scanner with pickle trust policy"
```

---

### Task T1.2: Integrate Security into Execution Pipeline

**Files:**
- Modify: `src/metrics_lie/execution.py:149-211` (adapter loading block)
- Modify: `src/metrics_lie/spec.py:40-63` (add trust_pickle field)
- Modify: `src/metrics_lie/cli_app.py:36-57` (add --trust-pickle flag)
- Modify: `src/metrics_lie/sdk.py:23-87` (add trust_pickle param)
- Test: `tests/test_v1_security_integration.py`

**Step 1: Add trust_pickle to ModelSourceSpec**

In `src/metrics_lie/spec.py`, add field to `ModelSourceSpec`:

```python
trust_pickle: bool = Field(
    default=False,
    description="Allow loading pickle files (security risk). Required for kind=pickle.",
)
```

**Step 2: Add security check in execution.py**

In `execution.py`, after `kind = spec.model_source.kind` (line 155), add:

```python
# Security: enforce trust policy for pickle models
if spec.model_source.path:
    from metrics_lie.model.security import check_trust_policy, scan_model_file
    check_trust_policy(
        spec.model_source.path,
        trust_pickle=spec.model_source.trust_pickle,
    )
    scan_result = scan_model_file(spec.model_source.path)
    if not scan_result.is_safe:
        raise ValueError(
            f"Model file failed security scan: {scan_result.issues}"
        )
```

**Step 3: Add --trust-pickle to CLI**

In `cli_app.py` `run` command, add parameter:

```python
trust_pickle: bool = typer.Option(
    False, "--trust-pickle", help="Allow loading pickle models (security risk)."
),
```

And in the body, set it in spec_dict:

```python
if trust_pickle:
    if "model_source" not in spec_dict:
        spec_dict["model_source"] = {}
    spec_dict["model_source"]["trust_pickle"] = True
```

**Step 4: Add trust_pickle to SDK evaluate()**

In `sdk.py` `evaluate()`, add `trust_pickle: bool = False` parameter. Set it in spec_dict:

```python
if model:
    model_path = Path(model)
    kind = _detect_model_kind(model_path)
    spec_dict["model_source"] = {
        "kind": kind,
        "path": str(model_path),
        "trust_pickle": trust_pickle,
    }
```

**Step 5: Write integration test**

```python
# tests/test_v1_security_integration.py
from __future__ import annotations

import pytest


def test_spec_trust_pickle_defaults_false():
    from metrics_lie.spec import ModelSourceSpec

    spec = ModelSourceSpec(kind="pickle", path="model.pkl")
    assert spec.trust_pickle is False


def test_spec_trust_pickle_can_be_set():
    from metrics_lie.spec import ModelSourceSpec

    spec = ModelSourceSpec(kind="pickle", path="model.pkl", trust_pickle=True)
    assert spec.trust_pickle is True


def test_onnx_spec_does_not_need_trust():
    from metrics_lie.spec import ModelSourceSpec

    spec = ModelSourceSpec(kind="onnx", path="model.onnx")
    assert spec.trust_pickle is False  # not needed for ONNX
```

**Step 6: Run tests**

Run: `pytest tests/test_v1_security.py tests/test_v1_security_integration.py -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: integrate security scanning into execution pipeline with --trust-pickle"
```

---

## T2: Missing Model Adapters (PyTorch, TensorFlow, HuggingFace)

### Task T2.1: PyTorch Adapter

**Files:**
- Create: `src/metrics_lie/model/adapters/pytorch_adapter.py`
- Test: `tests/test_v1_pytorch_adapter.py`
- Modify: `src/metrics_lie/model/default_registry.py` (register pytorch)
- Modify: `pyproject.toml` (add pytorch group)
- Modify: `src/metrics_lie/spec.py:41` (add "pytorch" to kind Literal)
- Modify: `src/metrics_lie/execution.py` (add pytorch elif branch)

**Step 1: Add pytorch dependency group**

In `pyproject.toml` after `boosting`:

```toml
pytorch = [
  "torch>=2.0",
]
```

Add to `all` group.

**Step 2: Write failing test**

```python
# tests/test_v1_pytorch_adapter.py
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.task_types import TaskType


@pytest.fixture
def simple_pytorch_model(tmp_path):
    """Create a minimal TorchScript model for testing."""
    torch = pytest.importorskip("torch")

    class SimpleClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)
            self.softmax = torch.nn.Softmax(dim=1)

        def forward(self, x):
            return self.softmax(self.linear(x))

    model = SimpleClassifier()
    scripted = torch.jit.script(model)
    path = str(tmp_path / "model.pt")
    scripted.save(path)
    return path


def test_pytorch_adapter_predict(simple_pytorch_model):
    from metrics_lie.model.adapters.pytorch_adapter import PyTorchAdapter

    adapter = PyTorchAdapter(
        path=simple_pytorch_model,
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    X = np.random.randn(10, 4).astype(np.float32)
    surface = adapter.predict(X)
    assert surface.n_samples == 10


def test_pytorch_adapter_predict_proba(simple_pytorch_model):
    from metrics_lie.model.adapters.pytorch_adapter import PyTorchAdapter

    adapter = PyTorchAdapter(
        path=simple_pytorch_model,
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    X = np.random.randn(10, 4).astype(np.float32)
    surface = adapter.predict_proba(X)
    assert surface is not None
    assert surface.n_samples == 10
    assert np.all(surface.values >= 0) and np.all(surface.values <= 1)


def test_pytorch_adapter_metadata(simple_pytorch_model):
    from metrics_lie.model.adapters.pytorch_adapter import PyTorchAdapter

    adapter = PyTorchAdapter(path=simple_pytorch_model)
    meta = adapter.metadata
    assert meta.model_format == "pytorch"
    assert "predict" in meta.capabilities


def test_pytorch_adapter_get_all_surfaces(simple_pytorch_model):
    from metrics_lie.model.adapters.pytorch_adapter import PyTorchAdapter
    from metrics_lie.model.surface import SurfaceType

    adapter = PyTorchAdapter(path=simple_pytorch_model)
    X = np.random.randn(5, 4).astype(np.float32)
    surfaces = adapter.get_all_surfaces(X)
    assert SurfaceType.LABEL in surfaces
```

**Step 3: Run tests to verify failure**

Run: `pytest tests/test_v1_pytorch_adapter.py -v`
Expected: FAIL (import error)

**Step 4: Implement PyTorchAdapter**

```python
# src/metrics_lie/model/adapters/pytorch_adapter.py
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from metrics_lie.model.metadata import ModelMetadata
from metrics_lie.model.surface import (
    CalibrationState,
    PredictionSurface,
    SurfaceType,
    validate_surface,
)
from metrics_lie.task_types import TaskType


class PyTorchAdapter:
    """Model adapter for PyTorch TorchScript models."""

    def __init__(
        self,
        *,
        path: str,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        threshold: float = 0.5,
        positive_label: int = 1,
    ) -> None:
        import torch

        self._path = path
        self._task_type = task_type
        self._threshold = threshold
        self._positive_label = positive_label
        self._model = torch.jit.load(path, map_location="cpu")
        self._model.eval()
        self._model_hash = self._compute_hash(path)

    @staticmethod
    def _compute_hash(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return f"sha256:{h.hexdigest()}"

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_class="TorchScriptModel",
            model_module="torch",
            model_format="pytorch",
            model_hash=self._model_hash,
            capabilities={"predict", "predict_proba"},
        )

    def _inference(self, X: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            tensor = torch.from_numpy(np.asarray(X, dtype=np.float32))
            output = self._model(tensor)
            return output.numpy()

    def predict(self, X: np.ndarray) -> PredictionSurface:
        raw = self._inference(X)
        if raw.ndim == 2:
            labels = np.argmax(raw, axis=1)
        else:
            labels = (raw.flatten() >= self._threshold).astype(int)
        arr = validate_surface(
            surface_type=SurfaceType.LABEL,
            values=labels,
            expected_n_samples=X.shape[0],
            threshold=None,
            enforce_binary=self._task_type == TaskType.BINARY_CLASSIFICATION,
        )
        return PredictionSurface(
            surface_type=SurfaceType.LABEL,
            values=arr.astype(int),
            dtype=arr.dtype,
            n_samples=int(arr.shape[0]),
            class_names=None,
            positive_label=self._positive_label,
            threshold=None,
            calibration_state=CalibrationState.UNKNOWN,
            model_hash=self._model_hash,
            is_deterministic=True,
        )

    def predict_proba(self, X: np.ndarray) -> PredictionSurface | None:
        raw = self._inference(X)
        if raw.ndim == 2 and raw.shape[1] > 1:
            if self._task_type == TaskType.BINARY_CLASSIFICATION:
                proba = raw[:, 1]
            else:
                proba = raw
        elif raw.ndim == 1:
            vals = raw.astype(float)
            if np.all((vals >= 0) & (vals <= 1)):
                proba = vals
            else:
                return None
        else:
            return None
        arr = validate_surface(
            surface_type=SurfaceType.PROBABILITY,
            values=proba,
            expected_n_samples=X.shape[0],
            threshold=self._threshold,
        )
        return PredictionSurface(
            surface_type=SurfaceType.PROBABILITY,
            values=arr.astype(float),
            dtype=arr.dtype,
            n_samples=int(arr.shape[0]),
            class_names=None,
            positive_label=self._positive_label,
            threshold=self._threshold,
            calibration_state=CalibrationState.UNKNOWN,
            model_hash=self._model_hash,
            is_deterministic=True,
        )

    def predict_raw(self, X: np.ndarray) -> dict[str, Any]:
        return {"output": self._inference(X).tolist()}

    def get_all_surfaces(self, X: np.ndarray) -> dict[SurfaceType, PredictionSurface]:
        surfaces: dict[SurfaceType, PredictionSurface] = {}
        surfaces[SurfaceType.LABEL] = self.predict(X)
        proba = self.predict_proba(X)
        if proba is not None:
            surfaces[SurfaceType.PROBABILITY] = proba
        return surfaces
```

**Step 5: Register in default_registry.py**

Add factory and registration:

```python
def _pytorch_factory(**kwargs):
    from metrics_lie.model.adapters.pytorch_adapter import PyTorchAdapter
    return PyTorchAdapter(**kwargs)

# Inside get_default_registry(), after CatBoost block:
try:
    import torch  # noqa: F401
    reg.register("pytorch", factory=_pytorch_factory, extensions={".pt", ".pth"})
except ImportError:
    pass
```

**Step 6: Add "pytorch" to spec.py ModelSourceSpec.kind Literal**

Change line 41:
```python
kind: Literal["pickle", "import", "onnx", "xgboost", "lightgbm", "catboost", "http", "pytorch"] = Field(
```

**Step 7: Add pytorch elif in execution.py**

After the boosting elif block (~line 209), before `else: raise`:

```python
elif kind == "pytorch":
    from metrics_lie.model.adapters.pytorch_adapter import PyTorchAdapter

    if not spec.model_source.path:
        raise ValueError("model_source.path is required for kind=pytorch")
    adapter = PyTorchAdapter(
        path=spec.model_source.path,
        task_type=TaskType(spec.task),
        threshold=spec.model_source.threshold or DEFAULT_THRESHOLD,
        positive_label=spec.model_source.positive_label or 1,
    )
```

**Step 8: Run tests**

Run: `pytest tests/test_v1_pytorch_adapter.py -v`
Expected: ALL PASS (or skip if torch not installed)

**Step 9: Commit**

```bash
git add -A
git commit -m "feat: add PyTorch TorchScript adapter with registry integration"
```

---

### Task T2.2: TensorFlow Adapter

**Files:**
- Create: `src/metrics_lie/model/adapters/tensorflow_adapter.py`
- Test: `tests/test_v1_tensorflow_adapter.py`
- Modify: `src/metrics_lie/model/default_registry.py`
- Modify: `pyproject.toml`
- Modify: `src/metrics_lie/spec.py:41`
- Modify: `src/metrics_lie/execution.py`

Follow identical pattern to T2.1 but for TensorFlow:

- Load via `tf.keras.models.load_model(path)` or `tf.saved_model.load(path)`
- Extensions: `.keras`, `.h5`
- Kind: `"tensorflow"`
- pyproject.toml group: `tensorflow = ["tensorflow>=2.15"]`
- Inference: `model.predict(X)` returns numpy

**Test:**
```python
@pytest.fixture
def simple_tf_model(tmp_path):
    tf = pytest.importorskip("tensorflow")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, activation="softmax", input_shape=(4,))
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    path = str(tmp_path / "model.keras")
    model.save(path)
    return path
```

**Implementation key:** `self._model = tf.keras.models.load_model(path)`, `raw = self._model.predict(X, verbose=0)`.

**Step: Commit**

```bash
git commit -m "feat: add TensorFlow/Keras adapter with registry integration"
```

---

### Task T2.3: HuggingFace Adapter

**Files:**
- Create: `src/metrics_lie/model/adapters/huggingface_adapter.py`
- Test: `tests/test_v1_huggingface_adapter.py`
- Modify: `src/metrics_lie/model/default_registry.py`
- Modify: `pyproject.toml`

**pyproject.toml:**
```toml
huggingface = [
  "transformers>=4.35",
  "safetensors",
]
```

**Implementation key:** Load via `transformers.pipeline(task, model=path)`. The HF adapter wraps a pipeline, calls `pipeline(inputs)`, and converts output to PredictionSurface.

**Note:** This adapter is text-aware. It does NOT go through numpy features like tabular adapters. Input is a list of strings or a DataFrame with text column. For v1.0, support `text-classification` pipeline only.

**Step: Commit**

```bash
git commit -m "feat: add HuggingFace pipeline adapter for text classification"
```

---

### Task T2.4: Add TEXT_CLASSIFICATION and TEXT_GENERATION to TaskType

**Files:**
- Modify: `src/metrics_lie/task_types.py` (add 2 enum members)
- Modify: `src/metrics_lie/spec.py:7-13` (add to TaskType Literal)
- Test: `tests/test_v1_task_types.py` (add assertions)

**In task_types.py, add:**
```python
TEXT_CLASSIFICATION = "text_classification"
TEXT_GENERATION = "text_generation"
```

**In spec.py, update Literal:**
```python
TaskType = Literal[
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "regression",
    "ranking",
    "text_classification",
    "text_generation",
]
```

**Test additions:**
```python
def test_text_classification_exists():
    from metrics_lie.task_types import TaskType
    assert TaskType.TEXT_CLASSIFICATION == "text_classification"
    assert TaskType.TEXT_CLASSIFICATION.is_classification

def test_text_generation_exists():
    from metrics_lie.task_types import TaskType
    assert TaskType.TEXT_GENERATION == "text_generation"
    assert not TaskType.TEXT_GENERATION.is_classification
```

**Commit:**
```bash
git commit -m "feat: add TEXT_CLASSIFICATION and TEXT_GENERATION task types"
```

---

## T3: Missing Metrics

### Task T3.1: Additional Regression Metrics

**Files:**
- Modify: `src/metrics_lie/metrics/core.py` (add MAPE, explained_variance, adjusted_r2)
- Modify: `src/metrics_lie/metrics/registry.py` (add MetricRequirements)
- Test: `tests/test_v2_regression_metrics.py` (add test cases)

**Step 1: Write failing tests**

```python
# Add to tests/test_v2_regression_metrics.py
import numpy as np

def test_metric_mape():
    from metrics_lie.metrics.core import metric_mape
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    result = metric_mape(y_true, y_pred)
    assert 0.0 < result < 0.2  # ~5% MAPE

def test_metric_explained_variance():
    from metrics_lie.metrics.core import metric_explained_variance
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1])
    result = metric_explained_variance(y_true, y_pred)
    assert result > 0.9

def test_metric_mape_in_metrics_dict():
    from metrics_lie.metrics.core import METRICS
    assert "mape" in METRICS
    assert "explained_variance" in METRICS
```

**Step 2: Implement**

```python
# Add to src/metrics_lie/metrics/core.py
from sklearn.metrics import (
    mean_absolute_percentage_error,
    explained_variance_score,
)

def metric_mape(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean absolute percentage error."""
    return float(mean_absolute_percentage_error(y_true, y_score))

def metric_explained_variance(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Explained variance score."""
    return float(explained_variance_score(y_true, y_score))
```

Add to `METRICS` dict and to `REGRESSION_METRICS` set.

Add `MetricRequirement` entries in `registry.py` for each new metric (task_types=`frozenset({"regression"})`).

**Step 3: Run tests, commit**

```bash
git commit -m "feat: add MAPE and explained_variance regression metrics"
```

---

### Task T3.2: Ranking Metrics

**Files:**
- Modify: `src/metrics_lie/metrics/core.py` (add ndcg, mrr, map_at_k)
- Modify: `src/metrics_lie/metrics/registry.py`
- Create: `tests/test_v2_ranking_metrics.py`

**Step 1: Write failing tests**

```python
# tests/test_v2_ranking_metrics.py
from __future__ import annotations

import numpy as np
import pytest


def test_metric_ndcg_at_k():
    from metrics_lie.metrics.core import metric_ndcg

    y_true = np.array([3, 2, 1, 0, 0])  # relevance scores
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])  # predicted scores
    result = metric_ndcg(y_true, y_score)
    assert 0.0 <= result <= 1.0


def test_metric_ndcg_perfect_ranking():
    from metrics_lie.metrics.core import metric_ndcg

    y_true = np.array([3, 2, 1, 0])
    y_score = np.array([3.0, 2.0, 1.0, 0.0])
    result = metric_ndcg(y_true, y_score)
    assert result == pytest.approx(1.0, abs=1e-6)


def test_ranking_metrics_in_registry():
    from metrics_lie.metrics.registry import METRIC_REQUIREMENTS

    ranking_ids = {r.metric_id for r in METRIC_REQUIREMENTS if r.task_types and "ranking" in r.task_types}
    assert "ndcg" in ranking_ids
```

**Step 2: Implement**

```python
# Add to metrics/core.py
from sklearn.metrics import ndcg_score

def metric_ndcg(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """NDCG (Normalized Discounted Cumulative Gain)."""
    return float(ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1)))
```

Add to METRICS dict. Add RANKING_METRICS set. Add MetricRequirement with `task_types=frozenset({"ranking"})`.

Add `SurfaceType.SCORE` to ranking metric requirements (ranking uses relevance scores).

**Step 3: Commit**

```bash
git commit -m "feat: add NDCG ranking metric with registry"
```

---

### Task T3.3: NLP Metrics (HF Evaluate Bridge)

**Files:**
- Create: `src/metrics_lie/metrics/nlp.py`
- Modify: `src/metrics_lie/metrics/core.py` (register NLP metrics)
- Modify: `src/metrics_lie/metrics/registry.py`
- Modify: `pyproject.toml` (add `metrics` group)
- Test: `tests/test_v2_nlp_metrics.py`

**pyproject.toml:**
```toml
metrics = [
  "evaluate>=0.4",
  "rouge-score>=0.1",
]
```

**Step 1: Write failing test**

```python
# tests/test_v2_nlp_metrics.py
from __future__ import annotations

import pytest


def test_rouge_l_metric():
    evaluate = pytest.importorskip("evaluate")
    from metrics_lie.metrics.nlp import metric_rouge_l

    y_true = ["the cat sat on the mat", "hello world"]
    y_pred = ["the cat sat on a mat", "hello there world"]
    result = metric_rouge_l(y_true, y_pred)
    assert 0.0 <= result <= 1.0


def test_nlp_metrics_registered():
    from metrics_lie.metrics.core import METRICS

    # Only if evaluate is installed
    try:
        import evaluate  # noqa: F401
        assert "rouge_l" in METRICS
    except ImportError:
        pass
```

**Step 2: Implement**

```python
# src/metrics_lie/metrics/nlp.py
from __future__ import annotations

from typing import Sequence


def metric_rouge_l(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """ROUGE-L score via HuggingFace evaluate."""
    import evaluate

    rouge = evaluate.load("rouge")
    result = rouge.compute(predictions=list(y_pred), references=list(y_true))
    return float(result["rougeL"])


def metric_bleu(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """BLEU score via HuggingFace evaluate."""
    import evaluate

    bleu = evaluate.load("bleu")
    result = bleu.compute(
        predictions=list(y_pred),
        references=[[ref] for ref in y_true],
    )
    return float(result["bleu"])
```

Register conditionally in `core.py`:
```python
try:
    from metrics_lie.metrics.nlp import metric_rouge_l, metric_bleu
    METRICS["rouge_l"] = metric_rouge_l
    METRICS["bleu"] = metric_bleu
    NLP_METRICS = {"rouge_l", "bleu"}
except ImportError:
    NLP_METRICS = set()
```

**Step 3: Commit**

```bash
git commit -m "feat: add ROUGE-L and BLEU NLP metrics via HF Evaluate bridge"
```

---

## T4: New Scenarios

### Task T4.1: Missing Features Scenario

**Files:**
- Create: `src/metrics_lie/scenarios/missing_features.py`
- Modify: `src/metrics_lie/surface_compat.py` (add to SCENARIO_TASK_COMPAT)
- Test: `tests/test_v2_scenario_missing_features.py`

**Step 1: Write failing test**

```python
# tests/test_v2_scenario_missing_features.py
from __future__ import annotations

import numpy as np

from metrics_lie.scenarios.base import ScenarioContext


def test_missing_features_drops_columns():
    from metrics_lie.scenarios.missing_features import MissingFeaturesScenario

    scenario = MissingFeaturesScenario(drop_rate=0.3)
    rng = np.random.default_rng(42)
    y_true = np.array([0, 1, 1, 0, 1])
    # y_score is a 2D feature matrix (5 samples, 10 features)
    y_score = np.random.randn(5, 10)
    ctx = ScenarioContext(task="binary_classification")

    y_t, y_s = scenario.apply(y_true, y_score, rng, ctx)
    # Some values should be NaN (dropped)
    assert np.isnan(y_s).any()
    assert y_t is y_true or np.array_equal(y_t, y_true)


def test_missing_features_describe():
    from metrics_lie.scenarios.missing_features import MissingFeaturesScenario

    s = MissingFeaturesScenario(drop_rate=0.2)
    d = s.describe()
    assert d["id"] == "missing_features"
    assert d["drop_rate"] == 0.2


def test_missing_features_registered():
    from metrics_lie.scenarios.registry import SCENARIO_REGISTRY

    assert "missing_features" in SCENARIO_REGISTRY
```

**Step 2: Implement**

```python
# src/metrics_lie/scenarios/missing_features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from metrics_lie.scenarios.base import ScenarioContext
from metrics_lie.scenarios.registry import register_scenario


@dataclass(frozen=True)
class MissingFeaturesScenario:
    """Drop random feature values to simulate missing data."""

    id: str = "missing_features"
    drop_rate: float = 0.2

    def apply(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        rng: np.random.Generator,
        ctx: ScenarioContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        if y_score.ndim < 2:
            return y_true, y_score
        result = y_score.astype(float).copy()
        mask = rng.random(size=result.shape) < self.drop_rate
        result[mask] = np.nan
        return y_true, result

    def describe(self) -> Dict[str, Any]:
        return {"id": self.id, "drop_rate": self.drop_rate}


def _factory(params: dict[str, Any]) -> MissingFeaturesScenario:
    return MissingFeaturesScenario(drop_rate=float(params.get("drop_rate", 0.2)))


register_scenario("missing_features", _factory)
```

**Step 3: Register in surface_compat.py**

Add `"missing_features"` to all tabular task types in `SCENARIO_TASK_COMPAT`:
```python
"binary_classification": {..., "missing_features"},
"multiclass_classification": {..., "missing_features"},
"regression": {..., "missing_features"},
```

**Step 4: Import in scenarios/__init__.py to trigger registration**

Add: `from metrics_lie.scenarios import missing_features  # noqa: F401`

**Step 5: Run tests, commit**

```bash
git commit -m "feat: add missing_features scenario for tabular stress testing"
```

---

### Task T4.2: Feature Corruption Scenario

Same pattern as T4.1. Replace values with noise/outliers:

```python
@dataclass(frozen=True)
class FeatureCorruptionScenario:
    id: str = "feature_corruption"
    corruption_rate: float = 0.1
    noise_scale: float = 3.0

    def apply(self, y_true, y_score, rng, ctx):
        if y_score.ndim < 2:
            return y_true, y_score
        result = y_score.astype(float).copy()
        mask = rng.random(size=result.shape) < self.corruption_rate
        noise = rng.normal(scale=self.noise_scale, size=result.shape)
        result[mask] += noise[mask]
        return y_true, result
```

**Commit:** `git commit -m "feat: add feature_corruption scenario"`

---

### Task T4.3: Covariate Shift Scenario

Reweight features to simulate distribution shift (built-in, no external deps):

```python
@dataclass(frozen=True)
class CovariateShiftScenario:
    id: str = "covariate_shift"
    shift_scale: float = 1.0

    def apply(self, y_true, y_score, rng, ctx):
        if y_score.ndim < 2:
            return y_true, y_score
        result = y_score.astype(float).copy()
        shifts = rng.normal(scale=self.shift_scale, size=result.shape[1])
        result += shifts[np.newaxis, :]
        return y_true, result
```

**Commit:** `git commit -m "feat: add covariate_shift scenario"`

---

### Task T4.4 through T4.8: Remaining Scenarios

Follow the same pattern for each. All are independent files:

| Task | Scenario | Key Logic | Deps |
|------|----------|-----------|------|
| T4.4 | `typo_injection` | Character-level perturbation on text columns | Built-in |
| T4.5 | `synonym_replacement` | WordNet-based swaps (nltk) | Built-in (basic) |
| T4.6 | `demographic_swap` | Swap protected attribute values | Built-in |
| T4.7 | `temporal_shift` | Shift/offset time-series features | Built-in |
| T4.8 | `label_quality` | Detect/inject realistic label errors | Built-in |

For each: create `src/metrics_lie/scenarios/<name>.py`, test file, register, add to compat table.

Each commit: `git commit -m "feat: add <scenario_name> scenario"`

---

## T5: Diagnostics Gaps

### Task T5.1: Advanced Fairness Metrics

**Files:**
- Modify: `src/metrics_lie/diagnostics/fairness.py`
- Modify: `tests/test_v3_fairness.py`

**Step 1: Write failing test**

```python
def test_equalized_odds_in_report():
    from metrics_lie.diagnostics.fairness import compute_fairness_report
    import numpy as np

    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

    report = compute_fairness_report(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
        metric_fns={"accuracy": lambda yt, yp: float(np.mean(yt == yp))},
    )
    assert "equalized_odds_difference" in report
```

**Step 2: Implement**

Add to `compute_fairness_report()` in `fairness.py`:

```python
# Equalized odds difference
try:
    from fairlearn.metrics import equalized_odds_difference
    result["equalized_odds_difference"] = float(
        equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
    )
except Exception:
    result["equalized_odds_difference"] = None
```

**Step 3: Commit**

```bash
git commit -m "feat: add equalized odds to fairness report"
```

---

### Task T5.2: Multiclass Per-Class Calibration

**Files:**
- Modify: `src/metrics_lie/diagnostics/calibration.py`
- Test: `tests/test_v3_calibration.py`

Add `per_class_ece()` that computes ECE for each class in a one-vs-rest manner:

```python
def per_class_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict[int, float]:
    """Per-class ECE for multiclass problems."""
    n_classes = y_proba.shape[1]
    result = {}
    for c in range(n_classes):
        binary_true = (y_true == c).astype(int)
        binary_proba = y_proba[:, c]
        result[c] = float(expected_calibration_error(binary_true, binary_proba, n_bins=n_bins))
    return result
```

**Commit:** `git commit -m "feat: add per-class ECE for multiclass calibration"`

---

## T6: SDK & CLI Polish

### Task T6.1: Builder Classes (Dataset, Model)

**Files:**
- Create: `src/metrics_lie/builders.py`
- Modify: `src/metrics_lie/__init__.py` (export)
- Test: `tests/test_v4_builders.py`

**Step 1: Write failing test**

```python
# tests/test_v4_builders.py
from __future__ import annotations


def test_dataset_from_csv():
    from metrics_lie.builders import Dataset

    ds = Dataset.from_csv("data.csv", y_true="label", y_score="pred")
    assert ds.path == "data.csv"
    assert ds.y_true_col == "label"


def test_model_from_pickle():
    from metrics_lie.builders import Model

    m = Model.from_pickle("model.pkl")
    assert m.kind == "pickle"
    assert m.path == "model.pkl"


def test_model_from_onnx():
    from metrics_lie.builders import Model

    m = Model.from_onnx("model.onnx")
    assert m.kind == "onnx"


def test_model_from_endpoint():
    from metrics_lie.builders import Model

    m = Model.from_endpoint("http://localhost:8080/v2/models/m/infer")
    assert m.kind == "http"
```

**Step 2: Implement**

```python
# src/metrics_lie/builders.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    """Builder for dataset specs."""

    path: str
    y_true_col: str = "y_true"
    y_score_col: str = "y_score"
    subgroup_col: str | None = None
    feature_cols: list[str] | None = None

    @classmethod
    def from_csv(
        cls,
        path: str,
        *,
        y_true: str = "y_true",
        y_score: str = "y_score",
        subgroup: str | None = None,
        features: list[str] | None = None,
    ) -> Dataset:
        return cls(
            path=path,
            y_true_col=y_true,
            y_score_col=y_score,
            subgroup_col=subgroup,
            feature_cols=features,
        )

    def to_spec_dict(self) -> dict:
        d = {"source": "csv", "path": self.path, "y_true_col": self.y_true_col, "y_score_col": self.y_score_col}
        if self.subgroup_col:
            d["subgroup_col"] = self.subgroup_col
        if self.feature_cols:
            d["feature_cols"] = self.feature_cols
        return d


@dataclass(frozen=True)
class Model:
    """Builder for model source specs."""

    kind: str
    path: str | None = None
    endpoint: str | None = None
    uri: str | None = None
    trust_pickle: bool = False

    @classmethod
    def from_pickle(cls, path: str, *, trust: bool = True) -> Model:
        return cls(kind="pickle", path=path, trust_pickle=trust)

    @classmethod
    def from_onnx(cls, path: str) -> Model:
        return cls(kind="onnx", path=path)

    @classmethod
    def from_pytorch(cls, path: str) -> Model:
        return cls(kind="pytorch", path=path)

    @classmethod
    def from_tensorflow(cls, path: str) -> Model:
        return cls(kind="tensorflow", path=path)

    @classmethod
    def from_endpoint(cls, url: str) -> Model:
        return cls(kind="http", endpoint=url)

    @classmethod
    def from_mlflow(cls, uri: str) -> Model:
        return cls(kind="mlflow", uri=uri)

    def to_spec_dict(self) -> dict:
        d: dict = {"kind": self.kind}
        if self.path:
            d["path"] = self.path
        if self.endpoint:
            d["endpoint"] = self.endpoint
        if self.trust_pickle:
            d["trust_pickle"] = True
        return d
```

**Step 3: Export in __init__.py**

Add `from metrics_lie.builders import Dataset, Model` and add to `__all__`.

**Step 4: Commit**

```bash
git commit -m "feat: add Dataset and Model builder classes for programmatic SDK"
```

---

## T7: Web Enhancements

### Task T7.1: LLM Context Task-Type Enrichment

**Files:**
- Modify: `web/backend/app/routers/llm.py` (add task_type to context bundle)

Add `task_type` to the context bundle built in the LLM router:

```python
# In buildContextBundle or equivalent context construction:
context["task_type"] = result_a.get("task_type", "binary_classification")
```

Update system prompt to include:
```python
f"The model is evaluated as a {task_type} task. "
```

**Commit:** `git commit -m "feat: enrich LLM analyst context with task_type"`

---

### Task T7.2: Frontend Scenario Filtering by Task Type

**Files:**
- Modify: `web/frontend/app/new/page.tsx` (filter stress suites by task type)
- Modify: `web/backend/app/routers/presets.py` (filter scenarios by task)

Backend: Add task_type filtering to scenario presets endpoint (same pattern as metric presets).

Frontend: Pass `taskType` to scenario fetch, filter visible suites.

**Commit:** `git commit -m "feat: filter stress suites by task type in web UI"`

---

## T8: Open Source Infrastructure

### Task T8.1: LICENSE File

**Files:**
- Create: `LICENSE`

Use Apache 2.0 (compatible with all integration targets per DESIGN_PLAN.md):

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/
...
```

**Commit:** `git commit -m "chore: add Apache 2.0 license"`

---

### Task T8.2: GitHub Templates

**Files:**
- Create: `.github/ISSUE_TEMPLATE/bug_report.md`
- Create: `.github/ISSUE_TEMPLATE/feature_request.md`
- Create: `.github/ISSUE_TEMPLATE/new_adapter.md`
- Create: `.github/PULL_REQUEST_TEMPLATE.md`

Standard GitHub templates. Bug report asks for: Spectra version, Python version, steps to reproduce, expected vs actual. Feature request: use case, proposed solution. New adapter: model format, library, example code.

PR template checklist:
```markdown
## Summary

## Checklist
- [ ] Tests pass (`pytest`)
- [ ] Lint passes (`ruff check src tests`)
- [ ] New tests for new functionality
- [ ] Documentation updated (if applicable)
```

**Commit:** `git commit -m "chore: add GitHub issue and PR templates"`

---

### Task T8.3: Pre-commit and Nox

**Files:**
- Create: `.pre-commit-config.yaml`
- Create: `noxfile.py`

**.pre-commit-config.yaml:**
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

**noxfile.py:**
```python
import nox

@nox.session(python=["3.11", "3.12"])
def tests(session):
    session.install("-e", ".[dev]")
    session.run("pytest", "--tb=short")

@nox.session
def lint(session):
    session.install("ruff>=0.3")
    session.run("ruff", "check", "src", "tests")

@nox.session
def docs(session):
    session.install("-e", ".[docs]")
    session.run("mkdocs", "build", "--strict")
```

**Commit:** `git commit -m "chore: add pre-commit config and noxfile"`

---

### Task T8.4: CI/CD Matrix Upgrade

**Files:**
- Modify: `.github/workflows/ci.yml`

Upgrade to Python matrix, coverage, and optional group testing:

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: ruff check src tests
      - run: pytest --tb=short -q

  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[docs]"
      - run: mkdocs build --strict
```

**Commit:** `git commit -m "chore: upgrade CI to Python matrix with docs build"`

---

### Task T8.5: CONTRIBUTING.md at Root

**Files:**
- Create: `CONTRIBUTING.md` (root level, links to docs/contributing.md for details)

```markdown
# Contributing to Spectra

We welcome contributions! See [the full contributing guide](docs/contributing.md) for details on:

- Adding new metrics
- Adding new scenarios
- Adding new model adapters
- Development setup
- Testing conventions

## Quick Start

git clone <repo> && cd when-metrics-lie
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest

## Code Style

We use ruff for linting and formatting. Run `ruff check src tests` before submitting.
```

**Commit:** `git commit -m "chore: add root CONTRIBUTING.md"`

---

### Task T8.6: Version Bump to 0.4.0

**Files:**
- Modify: `pyproject.toml:7` (version)
- Modify: `src/metrics_lie/__init__.py:4` (__version__)

Change `"0.3.0"` to `"0.4.0"` in both files.

**Commit:** `git commit -m "chore: bump version to 0.4.0 for v1.0 milestone"`

---

## Terminal Assignment Summary

| Terminal | Stream | Tasks | Est. Commits |
|----------|--------|-------|--------------|
| **T1** | Security | T1.1, T1.2 | 2 |
| **T2** | Adapters | T2.1, T2.2, T2.3, T2.4 | 4 |
| **T3** | Metrics | T3.1, T3.2, T3.3 | 3 |
| **T4** | Scenarios | T4.1–T4.8 | 8 |
| **T5** | Diagnostics | T5.1, T5.2 | 2 |
| **T6** | SDK polish | T6.1 | 1 |
| **T7** | Web | T7.1, T7.2 | 2 |
| **T8** | OSS infra | T8.1–T8.6 | 6 |

**Total: 28 commits across 8 parallel streams**

---

## Merge Strategy

1. Each terminal works on a feature branch: `v1/<stream-name>` (e.g., `v1/security`, `v1/adapters`)
2. All streams merge to `v1/integration` branch
3. Run full test suite on integration branch
4. Merge to `main` after all tests pass

---

## Post-Merge Verification

After all streams merge:

```bash
# Full test suite
pytest --tb=short -q

# Lint
ruff check src tests

# Docs build
mkdocs build --strict

# Verify new features
python -c "from metrics_lie.model.security import scan_model_file; print('Security: OK')"
python -c "from metrics_lie.metrics.core import METRICS; print(f'Metrics: {len(METRICS)}')"
python -c "from metrics_lie.scenarios.registry import SCENARIO_REGISTRY; print(f'Scenarios: {len(SCENARIO_REGISTRY)}')"
python -c "from metrics_lie.builders import Dataset, Model; print('Builders: OK')"
python -c "from metrics_lie.task_types import TaskType; print(f'TaskTypes: {len(TaskType)}')"
```
