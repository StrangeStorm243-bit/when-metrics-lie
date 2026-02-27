# V1.0 Phase 1: Universal Model Adapter — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the hardcoded binary-classification/sklearn-only engine into a task-agnostic, multi-format model evaluation platform. This is the foundation that unblocks Phases 2-6.

**Architecture:** Introduce a `TaskType` enum, a universal `ModelAdapter` Protocol, an adapter registry that auto-discovers model format, and a security scanning layer. Refactor `ExperimentSpec`, `execution.py`, and `validation.py` to route on `TaskType` instead of assuming binary classification. Existing binary classification behavior stays identical (all current tests pass).

**Tech Stack:** Python 3.11+, Pydantic v2, NumPy, scikit-learn, onnxruntime (optional), xgboost/lightgbm/catboost (optional)

---

## Work Streams

This plan has **3 parallel work streams** that can run in separate terminals/worktrees:

| Stream | Tasks | Dependencies |
|--------|-------|-------------|
| **A: Core Foundation** | Tasks 1-5 | None (start immediately) |
| **B: Adapter Implementations** | Tasks 6-9 | Blocked by Task 3 (protocol) |
| **C: Execution Pipeline Integration** | Tasks 10-14 | Blocked by Tasks 4, 5, 6 |

---

## Task 1: Create TaskType Enum

**Files:**
- Create: `src/metrics_lie/task_types.py`
- Test: `tests/test_v1_task_types.py`

**Step 1: Write the failing test**

```python
# tests/test_v1_task_types.py
from __future__ import annotations

from metrics_lie.task_types import TaskType


def test_task_type_values():
    assert TaskType.BINARY_CLASSIFICATION == "binary_classification"
    assert TaskType.MULTICLASS_CLASSIFICATION == "multiclass_classification"
    assert TaskType.REGRESSION == "regression"
    assert TaskType.RANKING == "ranking"


def test_task_type_from_string():
    assert TaskType("binary_classification") == TaskType.BINARY_CLASSIFICATION
    assert TaskType("regression") == TaskType.REGRESSION


def test_task_type_is_classification():
    assert TaskType.BINARY_CLASSIFICATION.is_classification is True
    assert TaskType.MULTICLASS_CLASSIFICATION.is_classification is True
    assert TaskType.REGRESSION.is_classification is False
    assert TaskType.RANKING.is_classification is False


def test_task_type_all_members():
    names = [t.value for t in TaskType]
    assert len(names) >= 4
    assert "binary_classification" in names
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_v1_task_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'metrics_lie.task_types'`

**Step 3: Write minimal implementation**

```python
# src/metrics_lie/task_types.py
from __future__ import annotations

from enum import Enum


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    RANKING = "ranking"

    @property
    def is_classification(self) -> bool:
        return self in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        )
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_v1_task_types.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/metrics_lie/task_types.py tests/test_v1_task_types.py
git commit -m "feat: add TaskType enum with classification/regression/ranking support"
```

---

## Task 2: Create ModelMetadata Dataclass

**Files:**
- Create: `src/metrics_lie/model/metadata.py`
- Test: `tests/test_v1_model_metadata.py`

**Step 1: Write the failing test**

```python
# tests/test_v1_model_metadata.py
from __future__ import annotations

from metrics_lie.model.metadata import ModelMetadata


def test_model_metadata_creation():
    meta = ModelMetadata(
        model_class="LogisticRegression",
        model_module="sklearn.linear_model",
        model_format="pickle",
        model_hash="sha256:abc123",
        capabilities={"predict", "predict_proba"},
    )
    assert meta.model_class == "LogisticRegression"
    assert meta.model_format == "pickle"
    assert "predict_proba" in meta.capabilities


def test_model_metadata_optional_fields():
    meta = ModelMetadata(
        model_class="CustomModel",
        model_module="custom",
        model_format="onnx",
    )
    assert meta.model_hash is None
    assert meta.capabilities == set()


def test_model_metadata_frozen():
    meta = ModelMetadata(
        model_class="X", model_module="x", model_format="pickle"
    )
    try:
        meta.model_class = "Y"  # type: ignore[misc]
        assert False, "Should be frozen"
    except (AttributeError, TypeError):
        pass
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_v1_model_metadata.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/metrics_lie/model/metadata.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelMetadata:
    model_class: str
    model_module: str
    model_format: str
    model_hash: str | None = None
    capabilities: set[str] = field(default_factory=set)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_v1_model_metadata.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/metrics_lie/model/metadata.py tests/test_v1_model_metadata.py
git commit -m "feat: add ModelMetadata dataclass for universal adapter info"
```

---

## Task 3: Create Universal ModelAdapter Protocol

This replaces the old concrete `ModelAdapter` class with a Protocol. The old class becomes `SklearnAdapter`.

**Files:**
- Create: `src/metrics_lie/model/protocol.py`
- Test: `tests/test_v1_model_protocol.py`

**Step 1: Write the failing test**

```python
# tests/test_v1_model_protocol.py
from __future__ import annotations

from typing import Any

import numpy as np

from metrics_lie.model.protocol import ModelAdapterProtocol
from metrics_lie.model.metadata import ModelMetadata
from metrics_lie.model.surface import PredictionSurface, SurfaceType, CalibrationState
from metrics_lie.task_types import TaskType


class FakeAdapter:
    """A minimal adapter that satisfies the protocol."""

    @property
    def task_type(self) -> TaskType:
        return TaskType.BINARY_CLASSIFICATION

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_class="Fake",
            model_module="test",
            model_format="memory",
        )

    def predict(self, X: np.ndarray) -> PredictionSurface:
        return PredictionSurface(
            surface_type=SurfaceType.LABEL,
            values=np.zeros(X.shape[0], dtype=int),
            dtype=np.dtype("int64"),
            n_samples=X.shape[0],
            class_names=("negative", "positive"),
            positive_label=1,
            threshold=None,
            calibration_state=CalibrationState.UNKNOWN,
            model_hash=None,
            is_deterministic=True,
        )

    def predict_proba(self, X: np.ndarray) -> PredictionSurface | None:
        return None

    def predict_raw(self, X: np.ndarray) -> dict[str, Any]:
        return {"labels": np.zeros(X.shape[0])}

    def get_all_surfaces(self, X: np.ndarray) -> dict[SurfaceType, PredictionSurface]:
        return {SurfaceType.LABEL: self.predict(X)}


def test_fake_adapter_satisfies_protocol():
    adapter: ModelAdapterProtocol = FakeAdapter()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    surface = adapter.predict(X)
    assert surface.surface_type == SurfaceType.LABEL
    assert surface.n_samples == 2
    assert adapter.task_type == TaskType.BINARY_CLASSIFICATION


def test_protocol_metadata():
    adapter: ModelAdapterProtocol = FakeAdapter()
    meta = adapter.metadata
    assert meta.model_class == "Fake"
    assert meta.model_format == "memory"


def test_protocol_predict_raw():
    adapter: ModelAdapterProtocol = FakeAdapter()
    X = np.array([[1.0]])
    raw = adapter.predict_raw(X)
    assert "labels" in raw
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_v1_model_protocol.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'metrics_lie.model.protocol'`

**Step 3: Write minimal implementation**

```python
# src/metrics_lie/model/protocol.py
from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from metrics_lie.model.metadata import ModelMetadata
from metrics_lie.model.surface import PredictionSurface, SurfaceType
from metrics_lie.task_types import TaskType


class ModelAdapterProtocol(Protocol):
    """Universal model adapter interface.

    All model adapters (sklearn, ONNX, PyTorch, etc.) implement this protocol.
    The adapter is responsible for loading the model, detecting capabilities,
    and producing PredictionSurfaces.
    """

    @property
    def task_type(self) -> TaskType: ...

    @property
    def metadata(self) -> ModelMetadata: ...

    def predict(self, X: np.ndarray) -> PredictionSurface: ...

    def predict_proba(self, X: np.ndarray) -> PredictionSurface | None: ...

    def predict_raw(self, X: np.ndarray) -> dict[str, Any]: ...

    def get_all_surfaces(self, X: np.ndarray) -> dict[SurfaceType, PredictionSurface]: ...
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_v1_model_protocol.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/metrics_lie/model/protocol.py tests/test_v1_model_protocol.py
git commit -m "feat: add ModelAdapterProtocol for universal model loading"
```

---

## Task 4: Create Adapter Registry

**Files:**
- Create: `src/metrics_lie/model/adapter_registry.py`
- Test: `tests/test_v1_adapter_registry.py`

**Step 1: Write the failing test**

```python
# tests/test_v1_adapter_registry.py
from __future__ import annotations

import pytest

from metrics_lie.model.adapter_registry import AdapterRegistry


def _fake_factory(**kwargs):
    """Placeholder — actual adapter not needed for registry tests."""
    return "fake_adapter"


def test_register_and_resolve_by_format():
    reg = AdapterRegistry()
    reg.register("sklearn", factory=_fake_factory, extensions={".pkl", ".pickle"})
    assert reg.resolve_format("sklearn") == _fake_factory


def test_resolve_by_extension():
    reg = AdapterRegistry()
    reg.register("sklearn", factory=_fake_factory, extensions={".pkl", ".pickle"})
    assert reg.resolve_extension(".pkl") == _fake_factory
    assert reg.resolve_extension(".pickle") == _fake_factory


def test_resolve_unknown_format_raises():
    reg = AdapterRegistry()
    with pytest.raises(KeyError, match="Unknown model format"):
        reg.resolve_format("unknown_format")


def test_resolve_unknown_extension_raises():
    reg = AdapterRegistry()
    with pytest.raises(KeyError, match="No adapter registered for extension"):
        reg.resolve_extension(".xyz")


def test_list_formats():
    reg = AdapterRegistry()
    reg.register("sklearn", factory=_fake_factory, extensions={".pkl"})
    reg.register("onnx", factory=_fake_factory, extensions={".onnx"})
    formats = reg.list_formats()
    assert "sklearn" in formats
    assert "onnx" in formats


def test_duplicate_format_raises():
    reg = AdapterRegistry()
    reg.register("sklearn", factory=_fake_factory, extensions={".pkl"})
    with pytest.raises(ValueError, match="already registered"):
        reg.register("sklearn", factory=_fake_factory, extensions={".pkl"})
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_v1_adapter_registry.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/metrics_lie/model/adapter_registry.py
from __future__ import annotations

from typing import Any, Callable


AdapterFactory = Callable[..., Any]


class AdapterRegistry:
    """Registry mapping model format names and file extensions to adapter factories."""

    def __init__(self) -> None:
        self._by_format: dict[str, AdapterFactory] = {}
        self._by_extension: dict[str, AdapterFactory] = {}

    def register(
        self,
        format_name: str,
        *,
        factory: AdapterFactory,
        extensions: set[str],
    ) -> None:
        if format_name in self._by_format:
            raise ValueError(f"Format '{format_name}' already registered")
        self._by_format[format_name] = factory
        for ext in extensions:
            self._by_extension[ext.lower()] = factory

    def resolve_format(self, format_name: str) -> AdapterFactory:
        if format_name not in self._by_format:
            raise KeyError(
                f"Unknown model format: '{format_name}'. "
                f"Available: {sorted(self._by_format.keys())}"
            )
        return self._by_format[format_name]

    def resolve_extension(self, ext: str) -> AdapterFactory:
        ext = ext.lower()
        if ext not in self._by_extension:
            raise KeyError(
                f"No adapter registered for extension '{ext}'. "
                f"Available: {sorted(self._by_extension.keys())}"
            )
        return self._by_extension[ext]

    def list_formats(self) -> list[str]:
        return sorted(self._by_format.keys())
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_v1_adapter_registry.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/metrics_lie/model/adapter_registry.py tests/test_v1_adapter_registry.py
git commit -m "feat: add AdapterRegistry for format-based model adapter discovery"
```

---

## Task 5: Generalize Validation — Remove Binary-Only Constraints

Currently `validation.py:validate_binary_labels` and `surface.py:validate_surface` enforce binary `{0,1}` only. We need to keep binary validation working but add paths for multiclass and regression.

**Files:**
- Modify: `src/metrics_lie/validation.py`
- Modify: `src/metrics_lie/model/surface.py` (SurfaceType enum, validate_surface)
- Test: `tests/test_v1_validation.py`

**Step 1: Write the failing test**

```python
# tests/test_v1_validation.py
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.validation import validate_labels, validate_binary_labels
from metrics_lie.model.surface import SurfaceType, validate_surface


def test_validate_binary_labels_still_works():
    """Existing binary validation unchanged."""
    arr = np.array([0, 1, 0, 1])
    validate_binary_labels(arr, "test")  # should not raise


def test_validate_binary_labels_rejects_multiclass():
    arr = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="binary"):
        validate_binary_labels(arr, "test")


def test_validate_labels_accepts_multiclass():
    arr = np.array([0, 1, 2, 3, 0])
    validate_labels(arr, "test")  # should not raise


def test_validate_labels_accepts_binary():
    arr = np.array([0, 1, 0, 1])
    validate_labels(arr, "test")  # should not raise


def test_validate_labels_rejects_nan():
    arr = np.array([0.0, np.nan, 1.0])
    with pytest.raises(ValueError, match="NaN"):
        validate_labels(arr, "test")


def test_validate_surface_label_multiclass():
    """Label surface should accept multiclass labels when not enforcing binary."""
    arr = np.array([0, 1, 2, 3])
    result = validate_surface(
        surface_type=SurfaceType.LABEL,
        values=arr,
        expected_n_samples=4,
        threshold=None,
        enforce_binary=False,
    )
    assert len(result) == 4


def test_validate_surface_label_binary_default():
    """Default behavior: label surface enforces binary."""
    arr = np.array([0, 1, 0, 1])
    result = validate_surface(
        surface_type=SurfaceType.LABEL,
        values=arr,
        expected_n_samples=4,
        threshold=None,
    )
    assert len(result) == 4


def test_validate_surface_regression():
    """CONTINUOUS surface type for regression outputs."""
    arr = np.array([1.5, 2.3, -0.7, 100.0])
    result = validate_surface(
        surface_type=SurfaceType.CONTINUOUS,
        values=arr,
        expected_n_samples=4,
        threshold=None,
    )
    assert len(result) == 4


def test_surface_type_continuous_exists():
    assert SurfaceType.CONTINUOUS == "continuous"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_v1_validation.py -v`
Expected: FAIL — `validate_labels` does not exist, `SurfaceType.CONTINUOUS` does not exist, `enforce_binary` parameter does not exist

**Step 3: Add `validate_labels` to validation.py**

Add to `src/metrics_lie/validation.py` after `validate_binary_labels`:

```python
def validate_labels(values: np.ndarray, name: str) -> None:
    """Raise ValueError if *values* contains NaN. Accepts any integer labels."""
    validate_no_nan(values, name)
```

**Step 4: Add CONTINUOUS to SurfaceType and update validate_surface**

In `src/metrics_lie/model/surface.py`:

Add `CONTINUOUS = "continuous"` to the `SurfaceType` enum (after SCORE).

Update `validate_surface` signature to add `enforce_binary: bool = True`, and update the LABEL branch:

```python
def validate_surface(
    *,
    surface_type: SurfaceType,
    values: np.ndarray,
    expected_n_samples: int,
    threshold: float | None,
    enforce_binary: bool = True,
) -> np.ndarray:
```

In the LABEL branch (currently line 101-107), change to:

```python
    elif surface_type == SurfaceType.LABEL:
        if arr.ndim != 1:
            raise SurfaceValidationError(f"label surface must be 1d. Got {arr.shape}")
        if enforce_binary:
            try:
                validate_binary_labels(arr, "label surface")
            except ValueError as e:
                raise SurfaceValidationError(str(e)) from e
        else:
            try:
                validate_labels(arr, "label surface")
            except ValueError as e:
                raise SurfaceValidationError(str(e)) from e
```

Add a CONTINUOUS branch:

```python
    elif surface_type == SurfaceType.CONTINUOUS:
        if arr.ndim != 1:
            raise SurfaceValidationError(f"continuous surface must be 1d. Got {arr.shape}")
```

Update the import at top of surface.py to include `validate_labels`:

```python
from metrics_lie.validation import validate_binary_labels, validate_labels, validate_no_inf, validate_no_nan, validate_numeric_dtype
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_v1_validation.py -v`
Expected: PASS

**Step 6: Run all existing tests to verify no regression**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS (binary behavior unchanged since `enforce_binary` defaults to `True`)

**Step 7: Commit**

```bash
git add src/metrics_lie/validation.py src/metrics_lie/model/surface.py tests/test_v1_validation.py
git commit -m "feat: generalize validation for multiclass labels and continuous surfaces"
```

---

## Task 6: Refactor Existing ModelAdapter → SklearnAdapter

The current `ModelAdapter` class in `adapter.py` is sklearn-specific. Rename it to `SklearnAdapter` and make it implement `ModelAdapterProtocol`. Keep backward compat by re-exporting `ModelAdapter = SklearnAdapter`.

**Files:**
- Modify: `src/metrics_lie/model/adapter.py`
- Test: `tests/test_v1_sklearn_adapter.py`

**Step 1: Write the failing test**

```python
# tests/test_v1_sklearn_adapter.py
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from metrics_lie.model.adapter import SklearnAdapter, ModelAdapter
from metrics_lie.model.sources import ModelSourceCallable
from metrics_lie.model.surface import SurfaceType
from metrics_lie.model.metadata import ModelMetadata
from metrics_lie.task_types import TaskType


def _make_mock_proba_model():
    model = MagicMock()
    model.__module__ = "sklearn.linear_model"
    model.__class__.__name__ = "LogisticRegression"
    model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
    model.predict.return_value = np.array([1, 0])
    model.decision_function.return_value = np.array([0.7, -0.3])
    return model


def test_sklearn_adapter_is_model_adapter():
    """SklearnAdapter is the same as ModelAdapter (backward compat)."""
    assert SklearnAdapter is ModelAdapter


def test_sklearn_adapter_task_type():
    fn = lambda X: np.array([0.5] * len(X))
    source = ModelSourceCallable(fn=fn, name="test")
    adapter = SklearnAdapter(source)
    assert adapter.task_type == TaskType.BINARY_CLASSIFICATION


def test_sklearn_adapter_metadata():
    fn = lambda X: np.array([0.5] * len(X))
    source = ModelSourceCallable(fn=fn, name="test")
    adapter = SklearnAdapter(source)
    meta = adapter.metadata
    assert isinstance(meta, ModelMetadata)
    assert meta.model_format == "callable"


def test_sklearn_adapter_predict_raw():
    fn = lambda X: np.array([0.5] * len(X))
    source = ModelSourceCallable(fn=fn, name="test")
    adapter = SklearnAdapter(source)
    X = np.array([[1.0, 2.0]])
    raw = adapter.predict_raw(X)
    assert "probabilities" in raw
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_v1_sklearn_adapter.py -v`
Expected: FAIL — `SklearnAdapter` doesn't exist, `task_type` property doesn't exist, etc.

**Step 3: Refactor adapter.py**

Modify `src/metrics_lie/model/adapter.py`:

1. Rename `class ModelAdapter` → `class SklearnAdapter`
2. Add `task_type` property (returns `TaskType.BINARY_CLASSIFICATION`)
3. Add `metadata` property (returns `ModelMetadata`)
4. Add `predict_raw` method
5. Add alias `ModelAdapter = SklearnAdapter` at bottom
6. Add imports for `ModelMetadata` and `TaskType`

The key changes:

```python
from metrics_lie.model.metadata import ModelMetadata
from metrics_lie.task_types import TaskType

# ... (keep existing imports)

class SklearnAdapter:
    # ... (keep entire existing class body unchanged)

    @property
    def task_type(self) -> TaskType:
        return TaskType.BINARY_CLASSIFICATION

    @property
    def metadata(self) -> ModelMetadata:
        caps = self.detect_capabilities()
        cap_set = {k for k, v in caps.items() if v}
        fmt = "callable" if isinstance(self._source, ModelSourceCallable) else "pickle"
        return ModelMetadata(
            model_class=self._model.__class__.__name__,
            model_module=getattr(self._model, "__module__", "unknown"),
            model_format=fmt,
            model_hash=self._model_hash,
            capabilities=cap_set,
        )

    def predict_raw(self, X: np.ndarray) -> dict[str, Any]:
        surfaces = self.get_all_surfaces(X)
        result: dict[str, Any] = {}
        if SurfaceType.PROBABILITY in surfaces:
            result["probabilities"] = surfaces[SurfaceType.PROBABILITY].values
        if SurfaceType.LABEL in surfaces:
            result["labels"] = surfaces[SurfaceType.LABEL].values
        if SurfaceType.SCORE in surfaces:
            result["scores"] = surfaces[SurfaceType.SCORE].values
        return result


# Backward compatibility alias
ModelAdapter = SklearnAdapter
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_v1_sklearn_adapter.py -v`
Expected: PASS

**Step 5: Run all existing tests to verify no regression**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS (ModelAdapter is still importable and works identically)

**Step 6: Commit**

```bash
git add src/metrics_lie/model/adapter.py tests/test_v1_sklearn_adapter.py
git commit -m "refactor: rename ModelAdapter to SklearnAdapter, add protocol properties"
```

---

## Task 7: Create ONNX Adapter

**Files:**
- Create: `src/metrics_lie/model/adapters/__init__.py`
- Create: `src/metrics_lie/model/adapters/onnx_adapter.py`
- Test: `tests/test_v1_onnx_adapter.py`

**Step 1: Write the failing test**

```python
# tests/test_v1_onnx_adapter.py
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.task_types import TaskType


onnxruntime = pytest.importorskip("onnxruntime")


def _create_simple_onnx_model(tmp_path):
    """Create a minimal ONNX model for testing."""
    skl2onnx = pytest.importorskip("skl2onnx")
    from sklearn.linear_model import LogisticRegression
    from skl2onnx.common.data_types import FloatTensorType

    X_train = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.float32)
    y_train = np.array([0, 1, 0, 1])
    model = LogisticRegression()
    model.fit(X_train, y_train)

    onnx_model = skl2onnx.convert_sklearn(
        model,
        "test_model",
        [("input", FloatTensorType([None, 2]))],
    )
    path = tmp_path / "model.onnx"
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return str(path)


def test_onnx_adapter_loads(tmp_path):
    from metrics_lie.model.adapters.onnx_adapter import ONNXAdapter

    path = _create_simple_onnx_model(tmp_path)
    adapter = ONNXAdapter(path=path, task_type=TaskType.BINARY_CLASSIFICATION)
    assert adapter.task_type == TaskType.BINARY_CLASSIFICATION


def test_onnx_adapter_predict(tmp_path):
    from metrics_lie.model.adapters.onnx_adapter import ONNXAdapter

    path = _create_simple_onnx_model(tmp_path)
    adapter = ONNXAdapter(path=path, task_type=TaskType.BINARY_CLASSIFICATION)
    X = np.array([[0, 0], [1, 1]], dtype=np.float32)
    surface = adapter.predict(X)
    assert surface.n_samples == 2
    assert surface.surface_type.value == "label"


def test_onnx_adapter_predict_proba(tmp_path):
    from metrics_lie.model.adapters.onnx_adapter import ONNXAdapter

    path = _create_simple_onnx_model(tmp_path)
    adapter = ONNXAdapter(path=path, task_type=TaskType.BINARY_CLASSIFICATION)
    X = np.array([[0, 0], [1, 1]], dtype=np.float32)
    surface = adapter.predict_proba(X)
    assert surface is not None
    assert surface.surface_type.value == "probability"
    assert np.all(surface.values >= 0) and np.all(surface.values <= 1)


def test_onnx_adapter_metadata(tmp_path):
    from metrics_lie.model.adapters.onnx_adapter import ONNXAdapter

    path = _create_simple_onnx_model(tmp_path)
    adapter = ONNXAdapter(path=path, task_type=TaskType.BINARY_CLASSIFICATION)
    meta = adapter.metadata
    assert meta.model_format == "onnx"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_v1_onnx_adapter.py -v`
Expected: FAIL — module not found (or skip if onnxruntime not installed)

**Step 3: Write implementation**

```python
# src/metrics_lie/model/adapters/__init__.py
```

```python
# src/metrics_lie/model/adapters/onnx_adapter.py
from __future__ import annotations

import hashlib
from pathlib import Path
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


class ONNXAdapter:
    """Model adapter for ONNX format models via onnxruntime."""

    def __init__(
        self,
        *,
        path: str,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        threshold: float = 0.5,
        positive_label: int = 1,
    ) -> None:
        import onnxruntime as ort

        self._path = path
        self._task_type = task_type
        self._threshold = threshold
        self._positive_label = positive_label

        self._session = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
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
            model_class="ONNXModel",
            model_module="onnxruntime",
            model_format="onnx",
            model_hash=self._model_hash,
            capabilities={"predict", "predict_proba"},
        )

    def _run(self, X: np.ndarray) -> list[np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        return self._session.run(self._output_names, {self._input_name: X})

    def predict(self, X: np.ndarray) -> PredictionSurface:
        outputs = self._run(X)
        labels = np.asarray(outputs[0]).flatten()
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
            class_names=("negative", "positive"),
            positive_label=self._positive_label,
            threshold=None,
            calibration_state=CalibrationState.UNKNOWN,
            model_hash=self._model_hash,
            is_deterministic=True,
        )

    def predict_proba(self, X: np.ndarray) -> PredictionSurface | None:
        outputs = self._run(X)
        if len(outputs) < 2:
            return None
        proba = np.asarray(outputs[1])
        if proba.ndim == 2 and proba.shape[1] > 1:
            pos_idx = 1 if proba.shape[1] > 1 else 0
            proba_1d = proba[:, pos_idx]
        else:
            proba_1d = proba.flatten()
        arr = validate_surface(
            surface_type=SurfaceType.PROBABILITY,
            values=proba_1d,
            expected_n_samples=X.shape[0],
            threshold=self._threshold,
        )
        return PredictionSurface(
            surface_type=SurfaceType.PROBABILITY,
            values=arr.astype(float),
            dtype=arr.dtype,
            n_samples=int(arr.shape[0]),
            class_names=("negative", "positive"),
            positive_label=self._positive_label,
            threshold=self._threshold,
            calibration_state=CalibrationState.UNKNOWN,
            model_hash=self._model_hash,
            is_deterministic=True,
        )

    def predict_raw(self, X: np.ndarray) -> dict[str, Any]:
        outputs = self._run(X)
        result: dict[str, Any] = {}
        if len(outputs) >= 1:
            result["labels"] = np.asarray(outputs[0]).flatten()
        if len(outputs) >= 2:
            result["probabilities"] = np.asarray(outputs[1])
        return result

    def get_all_surfaces(self, X: np.ndarray) -> dict[SurfaceType, PredictionSurface]:
        surfaces: dict[SurfaceType, PredictionSurface] = {}
        surfaces[SurfaceType.LABEL] = self.predict(X)
        proba = self.predict_proba(X)
        if proba is not None:
            surfaces[SurfaceType.PROBABILITY] = proba
        return surfaces
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_v1_onnx_adapter.py -v`
Expected: PASS (or SKIP if onnxruntime not installed)

**Step 5: Commit**

```bash
git add src/metrics_lie/model/adapters/ tests/test_v1_onnx_adapter.py
git commit -m "feat: add ONNXAdapter for ONNX model loading via onnxruntime"
```

---

## Task 8: Create Boosting Adapter (XGBoost, LightGBM, CatBoost)

**Files:**
- Create: `src/metrics_lie/model/adapters/boosting_adapter.py`
- Test: `tests/test_v1_boosting_adapter.py`

**Step 1: Write the failing test**

```python
# tests/test_v1_boosting_adapter.py
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.task_types import TaskType


xgb = pytest.importorskip("xgboost")


def _create_xgb_model(tmp_path):
    X_train = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.float32)
    y_train = np.array([0, 1, 0, 1])
    model = xgb.XGBClassifier(n_estimators=5, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    path = tmp_path / "model.ubj"
    model.save_model(str(path))
    return str(path)


def test_xgb_adapter_loads(tmp_path):
    from metrics_lie.model.adapters.boosting_adapter import BoostingAdapter

    path = _create_xgb_model(tmp_path)
    adapter = BoostingAdapter(
        path=path, framework="xgboost", task_type=TaskType.BINARY_CLASSIFICATION,
    )
    assert adapter.task_type == TaskType.BINARY_CLASSIFICATION


def test_xgb_adapter_predict(tmp_path):
    from metrics_lie.model.adapters.boosting_adapter import BoostingAdapter

    path = _create_xgb_model(tmp_path)
    adapter = BoostingAdapter(
        path=path, framework="xgboost", task_type=TaskType.BINARY_CLASSIFICATION,
    )
    X = np.array([[0, 0], [1, 1]], dtype=np.float32)
    surface = adapter.predict(X)
    assert surface.n_samples == 2
    assert surface.surface_type.value == "label"


def test_xgb_adapter_predict_proba(tmp_path):
    from metrics_lie.model.adapters.boosting_adapter import BoostingAdapter

    path = _create_xgb_model(tmp_path)
    adapter = BoostingAdapter(
        path=path, framework="xgboost", task_type=TaskType.BINARY_CLASSIFICATION,
    )
    X = np.array([[0, 0], [1, 1]], dtype=np.float32)
    surface = adapter.predict_proba(X)
    assert surface is not None
    assert surface.surface_type.value == "probability"


def test_xgb_adapter_metadata(tmp_path):
    from metrics_lie.model.adapters.boosting_adapter import BoostingAdapter

    path = _create_xgb_model(tmp_path)
    adapter = BoostingAdapter(
        path=path, framework="xgboost", task_type=TaskType.BINARY_CLASSIFICATION,
    )
    meta = adapter.metadata
    assert meta.model_format == "xgboost"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_v1_boosting_adapter.py -v`
Expected: FAIL or SKIP

**Step 3: Write implementation**

```python
# src/metrics_lie/model/adapters/boosting_adapter.py
from __future__ import annotations

import hashlib
from typing import Any, Literal

import numpy as np

from metrics_lie.model.metadata import ModelMetadata
from metrics_lie.model.surface import (
    CalibrationState,
    PredictionSurface,
    SurfaceType,
    validate_surface,
)
from metrics_lie.task_types import TaskType


BoostingFramework = Literal["xgboost", "lightgbm", "catboost"]


class BoostingAdapter:
    """Adapter for gradient boosting models (XGBoost, LightGBM, CatBoost)."""

    def __init__(
        self,
        *,
        path: str,
        framework: BoostingFramework,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        threshold: float = 0.5,
        positive_label: int = 1,
    ) -> None:
        self._path = path
        self._framework = framework
        self._task_type = task_type
        self._threshold = threshold
        self._positive_label = positive_label
        self._model = self._load_model(path, framework)
        self._model_hash = self._compute_hash(path)

    @staticmethod
    def _compute_hash(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return f"sha256:{h.hexdigest()}"

    @staticmethod
    def _load_model(path: str, framework: BoostingFramework) -> Any:
        if framework == "xgboost":
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(path)
            return model
        elif framework == "lightgbm":
            import lightgbm as lgb
            return lgb.Booster(model_file=path)
        elif framework == "catboost":
            from catboost import CatBoostClassifier
            model = CatBoostClassifier()
            model.load_model(path)
            return model
        raise ValueError(f"Unknown boosting framework: {framework}")

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_class=self._model.__class__.__name__,
            model_module=self._framework,
            model_format=self._framework,
            model_hash=self._model_hash,
            capabilities={"predict", "predict_proba"},
        )

    def predict(self, X: np.ndarray) -> PredictionSurface:
        labels = np.asarray(self._model.predict(X)).flatten()
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
            class_names=("negative", "positive"),
            positive_label=self._positive_label,
            threshold=None,
            calibration_state=CalibrationState.UNKNOWN,
            model_hash=self._model_hash,
            is_deterministic=True,
        )

    def predict_proba(self, X: np.ndarray) -> PredictionSurface | None:
        if not callable(getattr(self._model, "predict_proba", None)):
            return None
        raw = np.asarray(self._model.predict_proba(X))
        if raw.ndim == 2 and raw.shape[1] > 1:
            proba = raw[:, 1]
        else:
            proba = raw.flatten()
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
            class_names=("negative", "positive"),
            positive_label=self._positive_label,
            threshold=self._threshold,
            calibration_state=CalibrationState.UNKNOWN,
            model_hash=self._model_hash,
            is_deterministic=True,
        )

    def predict_raw(self, X: np.ndarray) -> dict[str, Any]:
        result: dict[str, Any] = {"labels": self._model.predict(X)}
        if callable(getattr(self._model, "predict_proba", None)):
            result["probabilities"] = self._model.predict_proba(X)
        return result

    def get_all_surfaces(self, X: np.ndarray) -> dict[SurfaceType, PredictionSurface]:
        surfaces: dict[SurfaceType, PredictionSurface] = {}
        surfaces[SurfaceType.LABEL] = self.predict(X)
        proba = self.predict_proba(X)
        if proba is not None:
            surfaces[SurfaceType.PROBABILITY] = proba
        return surfaces
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_v1_boosting_adapter.py -v`
Expected: PASS (or SKIP if xgboost not installed)

**Step 5: Commit**

```bash
git add src/metrics_lie/model/adapters/boosting_adapter.py tests/test_v1_boosting_adapter.py
git commit -m "feat: add BoostingAdapter for XGBoost/LightGBM/CatBoost models"
```

---

## Task 9: Create HTTP Endpoint Adapter

**Files:**
- Create: `src/metrics_lie/model/adapters/http_adapter.py`
- Test: `tests/test_v1_http_adapter.py`

**Step 1: Write the failing test**

```python
# tests/test_v1_http_adapter.py
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from metrics_lie.model.adapters.http_adapter import HTTPAdapter
from metrics_lie.model.surface import SurfaceType
from metrics_lie.task_types import TaskType


def _mock_response(predictions):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"predictions": predictions}
    return resp


def test_http_adapter_predict():
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response([
            {"label": 0, "probability": [0.8, 0.2]},
            {"label": 1, "probability": [0.3, 0.7]},
        ])
        adapter = HTTPAdapter(
            endpoint="http://localhost:8080/predict",
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        surface = adapter.predict(X)
        assert surface.n_samples == 2
        assert surface.surface_type == SurfaceType.LABEL


def test_http_adapter_predict_proba():
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response([
            {"label": 0, "probability": [0.8, 0.2]},
            {"label": 1, "probability": [0.3, 0.7]},
        ])
        adapter = HTTPAdapter(
            endpoint="http://localhost:8080/predict",
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        surface = adapter.predict_proba(X)
        assert surface is not None
        assert surface.surface_type == SurfaceType.PROBABILITY


def test_http_adapter_metadata():
    adapter = HTTPAdapter(
        endpoint="http://localhost:8080/predict",
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    meta = adapter.metadata
    assert meta.model_format == "http"
    assert meta.model_class == "HTTPModel"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_v1_http_adapter.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/metrics_lie/model/adapters/http_adapter.py
from __future__ import annotations

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


class HTTPAdapter:
    """Adapter for models served via HTTP endpoints."""

    def __init__(
        self,
        *,
        endpoint: str,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        threshold: float = 0.5,
        positive_label: int = 1,
        headers: dict[str, str] | None = None,
        batch_size: int = 256,
    ) -> None:
        self._endpoint = endpoint
        self._task_type = task_type
        self._threshold = threshold
        self._positive_label = positive_label
        self._headers = headers or {"Content-Type": "application/json"}
        self._batch_size = batch_size

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_class="HTTPModel",
            model_module="http",
            model_format="http",
            model_hash=None,
            capabilities={"predict", "predict_proba"},
        )

    def _call_endpoint(self, X: np.ndarray) -> list[dict[str, Any]]:
        import requests

        payload = {"instances": X.tolist()}
        resp = requests.post(
            self._endpoint,
            json=payload,
            headers=self._headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("predictions", [])

    def predict(self, X: np.ndarray) -> PredictionSurface:
        preds = self._call_endpoint(X)
        labels = np.array([p["label"] for p in preds])
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
            class_names=("negative", "positive"),
            positive_label=self._positive_label,
            threshold=None,
            calibration_state=CalibrationState.UNKNOWN,
            model_hash=None,
            is_deterministic=False,
        )

    def predict_proba(self, X: np.ndarray) -> PredictionSurface | None:
        preds = self._call_endpoint(X)
        if not preds or "probability" not in preds[0]:
            return None
        raw = np.array([p["probability"] for p in preds])
        if raw.ndim == 2 and raw.shape[1] > 1:
            proba = raw[:, 1]
        else:
            proba = raw.flatten()
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
            class_names=("negative", "positive"),
            positive_label=self._positive_label,
            threshold=self._threshold,
            calibration_state=CalibrationState.UNKNOWN,
            model_hash=None,
            is_deterministic=False,
        )

    def predict_raw(self, X: np.ndarray) -> dict[str, Any]:
        preds = self._call_endpoint(X)
        return {"predictions": preds}

    def get_all_surfaces(self, X: np.ndarray) -> dict[SurfaceType, PredictionSurface]:
        surfaces: dict[SurfaceType, PredictionSurface] = {}
        surfaces[SurfaceType.LABEL] = self.predict(X)
        proba = self.predict_proba(X)
        if proba is not None:
            surfaces[SurfaceType.PROBABILITY] = proba
        return surfaces
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_v1_http_adapter.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/metrics_lie/model/adapters/http_adapter.py tests/test_v1_http_adapter.py
git commit -m "feat: add HTTPAdapter for REST endpoint model evaluation"
```

---

## Task 10: Update ExperimentSpec for Multi-Task & Multi-Format

**Files:**
- Modify: `src/metrics_lie/spec.py`
- Test: `tests/test_v1_spec.py`

**Step 1: Write the failing test**

```python
# tests/test_v1_spec.py
from __future__ import annotations

import pytest

from metrics_lie.spec import ExperimentSpec, ModelSourceSpec, load_experiment_spec


def test_spec_accepts_multiclass_task():
    spec = load_experiment_spec({
        "name": "multiclass test",
        "task": "multiclass_classification",
        "dataset": {"path": "data.csv", "y_true_col": "label", "y_score_col": "pred"},
        "metric": "accuracy",
    })
    assert spec.task == "multiclass_classification"


def test_spec_accepts_regression_task():
    spec = load_experiment_spec({
        "name": "regression test",
        "task": "regression",
        "dataset": {"path": "data.csv", "y_true_col": "target", "y_score_col": "pred"},
        "metric": "accuracy",
    })
    assert spec.task == "regression"


def test_spec_still_defaults_to_binary():
    spec = load_experiment_spec({
        "name": "binary test",
        "dataset": {"path": "data.csv", "y_true_col": "y", "y_score_col": "p"},
        "metric": "auc",
    })
    assert spec.task == "binary_classification"


def test_spec_onnx_model_source():
    spec = load_experiment_spec({
        "name": "onnx test",
        "dataset": {"path": "data.csv", "y_true_col": "y", "y_score_col": "p"},
        "metric": "auc",
        "model_source": {"kind": "onnx", "path": "model.onnx"},
    })
    assert spec.model_source.kind == "onnx"


def test_spec_xgboost_model_source():
    spec = load_experiment_spec({
        "name": "xgb test",
        "dataset": {"path": "data.csv", "y_true_col": "y", "y_score_col": "p"},
        "metric": "auc",
        "model_source": {"kind": "xgboost", "path": "model.ubj"},
    })
    assert spec.model_source.kind == "xgboost"


def test_spec_http_model_source():
    spec = load_experiment_spec({
        "name": "http test",
        "dataset": {"path": "data.csv", "y_true_col": "y", "y_score_col": "p"},
        "metric": "auc",
        "model_source": {"kind": "http", "endpoint": "http://localhost:8080/predict"},
    })
    assert spec.model_source.kind == "http"
    assert spec.model_source.endpoint == "http://localhost:8080/predict"


def test_spec_rejects_invalid_task():
    with pytest.raises(Exception):
        load_experiment_spec({
            "name": "bad",
            "task": "quantum_computing",
            "dataset": {"path": "data.csv", "y_true_col": "y", "y_score_col": "p"},
            "metric": "auc",
        })
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_v1_spec.py -v`
Expected: FAIL — multiclass_classification not in `Literal["binary_classification"]`, onnx not in model_source kinds

**Step 3: Update spec.py**

In `src/metrics_lie/spec.py`:

1. Expand `TaskType` Literal:
```python
TaskType = Literal[
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "regression",
    "ranking",
]
```

2. Expand `ModelSourceSpec.kind`:
```python
class ModelSourceSpec(BaseModel):
    kind: Literal["pickle", "import", "onnx", "xgboost", "lightgbm", "catboost", "http"] = Field(
        ...,
        description="Model source type.",
    )
    path: Optional[str] = Field(default=None, description="Path to model file.")
    import_path: Optional[str] = Field(default=None, description="Import path (if kind=import).")
    endpoint: Optional[str] = Field(default=None, description="HTTP endpoint (if kind=http).")
    threshold: Optional[float] = Field(default=0.5, description="Decision threshold.")
    positive_label: Optional[int] = Field(default=1, description="Positive label index.")
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_v1_spec.py -v`
Expected: PASS

**Step 5: Run all existing tests to verify no regression**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/metrics_lie/spec.py tests/test_v1_spec.py
git commit -m "feat: expand ExperimentSpec for multi-task types and model formats"
```

---

## Task 11: Generalize Dataset Loader

The current `load_binary_csv` enforces binary labels. We need a more general loader that routes based on task type.

**Files:**
- Modify: `src/metrics_lie/datasets/loaders.py`
- Test: `tests/test_v1_dataset_loader.py`

**Step 1: Write the failing test**

```python
# tests/test_v1_dataset_loader.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def _write_csv(tmp_path, filename, df):
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return str(path)


def test_load_binary_csv_unchanged(tmp_path):
    """Existing binary loading still works."""
    from metrics_lie.datasets.loaders import load_binary_csv

    df = pd.DataFrame({"y": [0, 1, 0, 1], "p": [0.2, 0.8, 0.3, 0.9]})
    path = _write_csv(tmp_path, "binary.csv", df)
    ds = load_binary_csv(path, "y", "p")
    assert len(ds.y_true) == 4


def test_load_dataset_multiclass(tmp_path):
    from metrics_lie.datasets.loaders import load_dataset

    df = pd.DataFrame({"label": [0, 1, 2, 3], "pred": [0, 2, 1, 3]})
    path = _write_csv(tmp_path, "multi.csv", df)
    ds = load_dataset(
        path=path,
        y_true_col="label",
        y_score_col="pred",
        task_type="multiclass_classification",
    )
    assert len(ds.y_true) == 4
    assert set(ds.y_true.unique()) == {0, 1, 2, 3}


def test_load_dataset_regression(tmp_path):
    from metrics_lie.datasets.loaders import load_dataset

    df = pd.DataFrame({"target": [1.5, 2.3, -0.7, 100.0], "pred": [1.4, 2.5, -0.5, 99.0]})
    path = _write_csv(tmp_path, "reg.csv", df)
    ds = load_dataset(
        path=path,
        y_true_col="target",
        y_score_col="pred",
        task_type="regression",
    )
    assert len(ds.y_true) == 4


def test_load_dataset_binary_backward_compat(tmp_path):
    from metrics_lie.datasets.loaders import load_dataset

    df = pd.DataFrame({"y": [0, 1, 0, 1], "p": [0.2, 0.8, 0.3, 0.9]})
    path = _write_csv(tmp_path, "binary2.csv", df)
    ds = load_dataset(
        path=path,
        y_true_col="y",
        y_score_col="p",
        task_type="binary_classification",
    )
    assert len(ds.y_true) == 4
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_v1_dataset_loader.py -v`
Expected: FAIL — `load_dataset` does not exist

**Step 3: Add `load_dataset` function to loaders.py**

Add after the existing `load_binary_csv` function in `src/metrics_lie/datasets/loaders.py`:

```python
@dataclass(frozen=True)
class LoadedDataset:
    y_true: pd.Series
    y_score: pd.Series
    subgroup: Optional[pd.Series] = None
    X: Optional[pd.DataFrame] = None
    feature_cols: Optional[list[str]] = None


def load_dataset(
    path: str,
    y_true_col: str,
    y_score_col: str,
    task_type: str = "binary_classification",
    subgroup_col: str | None = None,
    *,
    feature_cols: list[str] | None = None,
    require_features: bool = False,
    allow_missing_score: bool = False,
) -> LoadedDataset:
    """Load a dataset for any task type."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    df = pd.read_csv(p)

    if y_true_col not in df.columns:
        raise ValueError(
            f"Missing required column '{y_true_col}'. Available: {list(df.columns)}"
        )
    if y_score_col not in df.columns:
        if allow_missing_score:
            df[y_score_col] = 0.0
        else:
            raise ValueError(
                f"Missing required column '{y_score_col}'. Available: {list(df.columns)}"
            )

    y_true = df[y_true_col]
    y_score = df[y_score_col]

    # Task-type-specific validation
    if task_type == "binary_classification":
        _validate_binary_labels(y_true, y_true_col)
        _validate_probability_series(y_score, y_score_col)
    elif task_type in ("multiclass_classification", "multilabel_classification"):
        arr = np.asarray(y_true)
        validate_no_nan(arr, y_true_col)
    elif task_type == "regression":
        arr_t = np.asarray(y_true, dtype=float)
        validate_no_nan(arr_t, y_true_col)
        validate_no_inf(arr_t, y_true_col)
        arr_s = np.asarray(y_score, dtype=float)
        validate_no_nan(arr_s, y_score_col)
        validate_no_inf(arr_s, y_score_col)
    # ranking and other types: minimal validation
    else:
        arr = np.asarray(y_true)
        validate_no_nan(arr, y_true_col)

    subgroup = None
    if subgroup_col:
        if subgroup_col not in df.columns:
            raise ValueError(
                f"Missing subgroup column '{subgroup_col}'. Available: {list(df.columns)}"
            )
        subgroup = df[subgroup_col]

    X = None
    resolved_feature_cols = None
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns: {missing}. Available: {list(df.columns)}"
            )
        resolved_feature_cols = list(feature_cols)
        X = df[resolved_feature_cols]
    elif require_features:
        excluded = {y_true_col, y_score_col}
        if subgroup_col:
            excluded.add(subgroup_col)
        inferred = [c for c in df.columns if c not in excluded]
        if not inferred:
            raise ValueError(
                "No feature columns inferred. Provide feature_cols explicitly."
            )
        resolved_feature_cols = inferred
        X = df[resolved_feature_cols]

    return LoadedDataset(
        y_true=y_true,
        y_score=y_score,
        subgroup=subgroup,
        X=X,
        feature_cols=resolved_feature_cols,
    )
```

Also add to imports at top: `from metrics_lie.validation import validate_no_nan, validate_no_inf` (they're already imported for binary, but verify).

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_v1_dataset_loader.py -v`
Expected: PASS

**Step 5: Run all existing tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/metrics_lie/datasets/loaders.py tests/test_v1_dataset_loader.py
git commit -m "feat: add load_dataset for multi-task CSV loading with task-aware validation"
```

---

## Task 12: Wire Adapter Registry into Default Instance

**Files:**
- Create: `src/metrics_lie/model/default_registry.py`
- Test: `tests/test_v1_default_registry.py`

**Step 1: Write the failing test**

```python
# tests/test_v1_default_registry.py
from __future__ import annotations

from metrics_lie.model.default_registry import get_default_registry


def test_default_registry_has_sklearn():
    reg = get_default_registry()
    formats = reg.list_formats()
    assert "sklearn" in formats


def test_default_registry_resolves_pkl():
    reg = get_default_registry()
    factory = reg.resolve_extension(".pkl")
    assert factory is not None


def test_default_registry_resolves_pickle():
    reg = get_default_registry()
    factory = reg.resolve_extension(".pickle")
    assert factory is not None


def test_default_registry_has_http():
    reg = get_default_registry()
    formats = reg.list_formats()
    assert "http" in formats
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_v1_default_registry.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/metrics_lie/model/default_registry.py
from __future__ import annotations

from metrics_lie.model.adapter_registry import AdapterRegistry


def _sklearn_factory(**kwargs):
    from metrics_lie.model.adapter import SklearnAdapter
    return SklearnAdapter(**kwargs)


def _onnx_factory(**kwargs):
    from metrics_lie.model.adapters.onnx_adapter import ONNXAdapter
    return ONNXAdapter(**kwargs)


def _boosting_factory(**kwargs):
    from metrics_lie.model.adapters.boosting_adapter import BoostingAdapter
    return BoostingAdapter(**kwargs)


def _http_factory(**kwargs):
    from metrics_lie.model.adapters.http_adapter import HTTPAdapter
    return HTTPAdapter(**kwargs)


_DEFAULT_REGISTRY: AdapterRegistry | None = None


def get_default_registry() -> AdapterRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is not None:
        return _DEFAULT_REGISTRY

    reg = AdapterRegistry()

    # sklearn (always available)
    reg.register("sklearn", factory=_sklearn_factory, extensions={".pkl", ".pickle", ".joblib"})

    # HTTP (always available — no special deps)
    reg.register("http", factory=_http_factory, extensions=set())

    # ONNX (optional)
    try:
        import onnxruntime  # noqa: F401
        reg.register("onnx", factory=_onnx_factory, extensions={".onnx"})
    except ImportError:
        pass

    # XGBoost (optional)
    try:
        import xgboost  # noqa: F401
        reg.register("xgboost", factory=_boosting_factory, extensions={".ubj", ".xgb"})
    except ImportError:
        pass

    # LightGBM (optional)
    try:
        import lightgbm  # noqa: F401
        reg.register("lightgbm", factory=_boosting_factory, extensions={".lgb", ".txt"})
    except ImportError:
        pass

    # CatBoost (optional)
    try:
        import catboost  # noqa: F401
        reg.register("catboost", factory=_boosting_factory, extensions={".cbm"})
    except ImportError:
        pass

    _DEFAULT_REGISTRY = reg
    return reg
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_v1_default_registry.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/metrics_lie/model/default_registry.py tests/test_v1_default_registry.py
git commit -m "feat: add default adapter registry with auto-discovery of installed formats"
```

---

## Task 13: Update execution.py to Use Adapter Registry

This is the integration task. Modify `execution.py` to:
1. Use the adapter registry instead of hardcoded sklearn/pickle/import logic
2. Use `load_dataset` for non-binary tasks
3. Pass `task_type` through the pipeline

**Files:**
- Modify: `src/metrics_lie/execution.py`
- Test: `tests/test_v1_execution.py`

**Step 1: Write the failing test**

```python
# tests/test_v1_execution.py
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def _create_binary_csv(tmp_path):
    df = pd.DataFrame({
        "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "p": [0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.25, 0.75, 0.3, 0.7],
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_execution_binary_still_works(tmp_path):
    """Existing binary classification execution pipeline unchanged."""
    from metrics_lie.execution import run_from_spec_dict

    csv_path = _create_binary_csv(tmp_path)
    spec = {
        "name": "binary test",
        "task": "binary_classification",
        "dataset": {"path": csv_path, "y_true_col": "y", "y_score_col": "p"},
        "metric": "auc",
        "n_trials": 5,
        "seed": 42,
    }
    run_id = run_from_spec_dict(spec)
    assert run_id is not None
    assert len(run_id) == 10


def test_execution_default_task_binary(tmp_path):
    """When task is omitted, defaults to binary_classification."""
    from metrics_lie.execution import run_from_spec_dict

    csv_path = _create_binary_csv(tmp_path)
    spec = {
        "name": "default task test",
        "dataset": {"path": csv_path, "y_true_col": "y", "y_score_col": "p"},
        "metric": "auc",
        "n_trials": 5,
        "seed": 42,
    }
    run_id = run_from_spec_dict(spec)
    assert run_id is not None
```

**Step 2: Run test to verify current behavior works**

Run: `python -m pytest tests/test_v1_execution.py -v`
Expected: PASS (both tests should work with existing code since they test binary)

**Step 3: Update execution.py model loading section**

In `src/metrics_lie/execution.py`, replace the hardcoded model loading block (lines 130-168) with registry-based loading:

Replace the section starting with `if spec.model_source is not None:` (line 133) through the end of the model loading logic (line 168) with:

```python
    # If model source is provided, run inference via adapter registry.
    prediction_surface = None
    surface_type = SurfaceType.PROBABILITY
    if spec.model_source is not None:
        from metrics_lie.model.default_registry import get_default_registry
        from metrics_lie.model.surface import CalibrationState

        if ds.X is None:
            raise ValueError("Model inference requires feature columns in dataset.")

        registry = get_default_registry()
        kind = spec.model_source.kind

        if kind in ("pickle", "import"):
            # Legacy sklearn path — use SklearnAdapter directly
            from metrics_lie.model.adapter import SklearnAdapter
            from metrics_lie.model.sources import ModelSourceImport, ModelSourcePickle

            if kind == "pickle":
                if not spec.model_source.path:
                    raise ValueError("model_source.path is required for kind=pickle")
                source = ModelSourcePickle(path=spec.model_source.path)
            else:
                if not spec.model_source.import_path:
                    raise ValueError("model_source.import_path is required for kind=import")
                source = ModelSourceImport(import_path=spec.model_source.import_path)

            adapter = SklearnAdapter(
                source,
                threshold=spec.model_source.threshold or DEFAULT_THRESHOLD,
                positive_label=spec.model_source.positive_label or 1,
                calibration_state=CalibrationState.UNKNOWN,
            )
        elif kind == "http":
            factory = registry.resolve_format("http")
            adapter = factory(
                endpoint=spec.model_source.endpoint,
                task_type=TaskType(spec.task),
                threshold=spec.model_source.threshold or DEFAULT_THRESHOLD,
                positive_label=spec.model_source.positive_label or 1,
            )
        else:
            # Registry-based loading for all other formats
            factory = registry.resolve_format(kind)
            adapter_kwargs = {
                "path": spec.model_source.path,
                "task_type": TaskType(spec.task),
                "threshold": spec.model_source.threshold or DEFAULT_THRESHOLD,
                "positive_label": spec.model_source.positive_label or 1,
            }
            if kind in ("xgboost", "lightgbm", "catboost"):
                adapter_kwargs["framework"] = kind
            adapter = factory(**adapter_kwargs)

        surfaces = adapter.get_all_surfaces(ds.X.to_numpy())
        # Prefer probability surface when available.
        if SurfaceType.PROBABILITY in surfaces:
            prediction_surface = surfaces[SurfaceType.PROBABILITY]
        elif SurfaceType.SCORE in surfaces:
            prediction_surface = surfaces[SurfaceType.SCORE]
        elif SurfaceType.LABEL in surfaces:
            prediction_surface = surfaces[SurfaceType.LABEL]
        else:
            raise ValueError("Model adapter produced no usable surfaces.")
        surface_type = prediction_surface.surface_type
        y_score = prediction_surface.values.astype(float)
```

Also add at top of file:
```python
from metrics_lie.task_types import TaskType
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_v1_execution.py -v`
Expected: PASS

**Step 5: Run ALL existing tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/metrics_lie/execution.py tests/test_v1_execution.py
git commit -m "feat: wire adapter registry into execution pipeline for multi-format model loading"
```

---

## Task 14: Update Surface Compatibility for New Task Types

**Files:**
- Modify: `src/metrics_lie/surface_compat.py`
- Test: `tests/test_v1_surface_compat.py`

**Step 1: Write the failing test**

```python
# tests/test_v1_surface_compat.py
from __future__ import annotations

from metrics_lie.model.surface import SurfaceType
from metrics_lie.surface_compat import SCENARIO_SURFACE_COMPAT


def test_probability_surface_has_all_scenarios():
    """Probability surface supports all existing scenarios."""
    allowed = SCENARIO_SURFACE_COMPAT[SurfaceType.PROBABILITY]
    assert "label_noise" in allowed
    assert "score_noise" in allowed
    assert "class_imbalance" in allowed
    assert "threshold_gaming" in allowed


def test_continuous_surface_exists():
    """CONTINUOUS surface type should have scenario compatibility."""
    assert SurfaceType.CONTINUOUS in SCENARIO_SURFACE_COMPAT


def test_continuous_surface_allows_score_noise():
    allowed = SCENARIO_SURFACE_COMPAT[SurfaceType.CONTINUOUS]
    assert "score_noise" in allowed


def test_continuous_surface_excludes_threshold():
    allowed = SCENARIO_SURFACE_COMPAT[SurfaceType.CONTINUOUS]
    assert "threshold_gaming" not in allowed
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_v1_surface_compat.py -v`
Expected: FAIL — `SurfaceType.CONTINUOUS` not in `SCENARIO_SURFACE_COMPAT`

**Step 3: Update surface_compat.py**

Add the CONTINUOUS entry to `SCENARIO_SURFACE_COMPAT` and update the `SURFACE_TYPE_MAP`:

```python
SURFACE_TYPE_MAP: dict[str, SurfaceType] = {
    "probability": SurfaceType.PROBABILITY,
    "score": SurfaceType.SCORE,
    "label": SurfaceType.LABEL,
    "continuous": SurfaceType.CONTINUOUS,
}

SCENARIO_SURFACE_COMPAT: dict[SurfaceType, set[str]] = {
    SurfaceType.PROBABILITY: {
        "label_noise",
        "score_noise",
        "class_imbalance",
        "threshold_gaming",
    },
    SurfaceType.SCORE: {"label_noise", "score_noise", "class_imbalance"},
    SurfaceType.LABEL: {"label_noise", "class_imbalance"},
    SurfaceType.CONTINUOUS: {"score_noise"},
}
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_v1_surface_compat.py -v`
Expected: PASS

**Step 5: Run ALL existing tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/metrics_lie/surface_compat.py tests/test_v1_surface_compat.py
git commit -m "feat: add CONTINUOUS surface type to compatibility matrix"
```

---

## Task 15: Update pyproject.toml with Optional Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml**

Add optional dependency groups for new model formats:

```toml
[project.optional-dependencies]
dev = [
  "pytest>=7.0",
  "ruff>=0.3"
]
web = [
  "fastapi>=0.104.0",
  "uvicorn[standard]>=0.24.0",
  "requests>=2.31.0"
]
onnx = [
  "onnxruntime>=1.16",
  "skl2onnx>=1.16"
]
boosting = [
  "xgboost>=2.0",
  "lightgbm>=4.0",
  "catboost>=1.2"
]
all = [
  "metrics_lie[dev,web,onnx,boosting]"
]
```

**Step 2: Run existing tests to verify no regression**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add optional dependency groups for onnx, boosting model formats"
```

---

## Task 16: Full Integration Test — End-to-End with ONNX

**Files:**
- Test: `tests/test_v1_e2e_onnx.py`

**Step 1: Write the integration test**

```python
# tests/test_v1_e2e_onnx.py
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

onnxruntime = pytest.importorskip("onnxruntime")
skl2onnx = pytest.importorskip("skl2onnx")


def _create_onnx_model_and_data(tmp_path):
    """Create an ONNX model and matching dataset for e2e testing."""
    from sklearn.linear_model import LogisticRegression
    from skl2onnx.common.data_types import FloatTensorType

    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 3)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = LogisticRegression()
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]

    # Save as ONNX
    onnx_model = skl2onnx.convert_sklearn(
        model, "test", [("input", FloatTensorType([None, 3]))]
    )
    model_path = tmp_path / "model.onnx"
    with open(model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Save dataset
    df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    df["y_true"] = y
    df["y_score"] = proba
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    return str(model_path), str(csv_path)


def test_e2e_onnx_model(tmp_path):
    from metrics_lie.execution import run_from_spec_dict

    model_path, csv_path = _create_onnx_model_and_data(tmp_path)
    spec = {
        "name": "ONNX E2E test",
        "task": "binary_classification",
        "dataset": {
            "path": csv_path,
            "y_true_col": "y_true",
            "y_score_col": "y_score",
            "feature_cols": ["f1", "f2", "f3"],
        },
        "model_source": {
            "kind": "onnx",
            "path": model_path,
        },
        "metric": "auc",
        "scenarios": [
            {"id": "label_noise", "params": {"p": 0.05}},
            {"id": "score_noise", "params": {"sigma": 0.02}},
        ],
        "n_trials": 10,
        "seed": 42,
    }
    run_id = run_from_spec_dict(spec)
    assert run_id is not None
    assert len(run_id) == 10

    # Verify results file was written
    from metrics_lie.utils.paths import get_run_dir
    paths = get_run_dir(run_id)
    assert paths.results_json.exists()

    # Load and verify result bundle
    result = json.loads(paths.results_json.read_text())
    assert result["metric_name"] == "auc"
    assert result["run_id"] == run_id
    assert len(result["scenarios"]) == 2
```

**Step 2: Run test**

Run: `python -m pytest tests/test_v1_e2e_onnx.py -v`
Expected: PASS (or SKIP if onnxruntime not installed)

**Step 3: Commit**

```bash
git add tests/test_v1_e2e_onnx.py
git commit -m "test: add end-to-end ONNX model integration test"
```

---

## Task 17: Run Full Test Suite + Lint Check

**Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 2: Run linter**

Run: `ruff check src tests`
Expected: No errors

**Step 3: Fix any lint issues**

If ruff reports issues, fix them.

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: fix lint issues from Phase 1 implementation"
```

---

## Summary: Files Created/Modified

### New Files (10)
| File | Purpose |
|------|---------|
| `src/metrics_lie/task_types.py` | TaskType enum |
| `src/metrics_lie/model/metadata.py` | ModelMetadata dataclass |
| `src/metrics_lie/model/protocol.py` | ModelAdapterProtocol |
| `src/metrics_lie/model/adapter_registry.py` | AdapterRegistry |
| `src/metrics_lie/model/default_registry.py` | Default registry with auto-discovery |
| `src/metrics_lie/model/adapters/__init__.py` | Adapters package |
| `src/metrics_lie/model/adapters/onnx_adapter.py` | ONNXAdapter |
| `src/metrics_lie/model/adapters/boosting_adapter.py` | BoostingAdapter |
| `src/metrics_lie/model/adapters/http_adapter.py` | HTTPAdapter |
| 8 test files | TDD tests for each task |

### Modified Files (5)
| File | Change |
|------|--------|
| `src/metrics_lie/spec.py` | Expand TaskType Literal, ModelSourceSpec kinds |
| `src/metrics_lie/validation.py` | Add `validate_labels` for multiclass |
| `src/metrics_lie/model/surface.py` | Add CONTINUOUS to SurfaceType, `enforce_binary` param |
| `src/metrics_lie/model/adapter.py` | Rename to SklearnAdapter, add protocol properties |
| `src/metrics_lie/surface_compat.py` | Add CONTINUOUS to compat matrix |
| `src/metrics_lie/execution.py` | Use adapter registry for model loading |
| `src/metrics_lie/datasets/loaders.py` | Add `load_dataset` for multi-task |
| `pyproject.toml` | Add optional dependency groups |

### Parallel Execution Assignment

| Stream | Tasks | Can Start |
|--------|-------|-----------|
| **Stream A** (Core Foundation) | 1, 2, 3, 4, 5 | Immediately |
| **Stream B** (Adapters) | 7, 8, 9 | After Task 3 |
| **Stream C** (Integration) | 6, 10, 11, 12, 13, 14, 15, 16, 17 | After Tasks 3-5 |

**Optimal parallel schedule:**
1. Stream A: Tasks 1-5 in parallel (independent)
2. When Tasks 1-5 done: Stream B (Tasks 7-9) + Stream C starts (Task 6, 10-11)
3. When Task 6 done: Task 12
4. When Tasks 12 + 10-11 done: Task 13
5. When Task 13 done: Tasks 14-17 (sequential)

**Total estimated commits: 17**
