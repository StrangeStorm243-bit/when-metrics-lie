# V1.0 Phase 2: Multi-Task Metrics & Scenarios Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make metrics and scenarios work across binary classification, multiclass classification, and regression task types — transforming Spectra from binary-only to multi-task.

**Architecture:** Extend the existing metric registry with task-type awareness (`task_types` field on `MetricRequirement`). Add new metric functions for multiclass (macro/weighted F1, macro AUC, Cohen's Kappa, top-k accuracy) and regression (MAE, MSE, RMSE, R², max error). Generalize existing scenarios (label_noise, score_noise, class_imbalance) to handle multiclass labels and regression targets. Wire everything through execution.py by replacing `load_binary_csv` with the Phase 1 `load_dataset`, adding task-type-conditional logic for label casting, analysis gating, and schema updates.

**Tech Stack:** Python 3.11+, scikit-learn (metrics), NumPy, Pydantic v2, pytest (TDD)

**Scope boundaries:** This phase covers tabular metrics and scenario generalization only. NLP/text metrics (ROUGE, BLEU), LLM metrics, and image scenarios are deferred to later phases. Ranking metrics (NDCG, MAP) are also deferred — they require a fundamentally different input format (query-document pairs) that warrants separate planning.

---

## Work Streams

The tasks below are organized in dependency order. Tasks within the same group (A, B, C) are independent and can be dispatched as parallel subagents.

### Stream A: Core Foundation (Tasks 1-5)
### Stream B: Scenario Generalization (Tasks 6-9)
### Stream C: Pipeline Integration (Tasks 10-15)

---

### Task 1: Add task-type convenience properties to TaskType

**Files:**
- Modify: `src/metrics_lie/task_types.py`
- Test: `tests/test_v1_task_types.py`

**Context:** The `TaskType` enum from Phase 1 has `is_classification`. We need `is_binary`, `is_regression`, and `supports_threshold` to drive routing decisions in metrics, runner, and execution.

**Step 1: Write the failing tests**

Add these tests to the existing `tests/test_v1_task_types.py`:

```python
def test_task_type_is_binary():
    assert TaskType.BINARY_CLASSIFICATION.is_binary is True
    assert TaskType.MULTICLASS_CLASSIFICATION.is_binary is False
    assert TaskType.REGRESSION.is_binary is False
    assert TaskType.RANKING.is_binary is False

def test_task_type_is_regression():
    assert TaskType.REGRESSION.is_regression is True
    assert TaskType.BINARY_CLASSIFICATION.is_regression is False
    assert TaskType.MULTICLASS_CLASSIFICATION.is_regression is False

def test_task_type_supports_threshold():
    assert TaskType.BINARY_CLASSIFICATION.supports_threshold is True
    assert TaskType.MULTICLASS_CLASSIFICATION.supports_threshold is False
    assert TaskType.REGRESSION.supports_threshold is False
    assert TaskType.RANKING.supports_threshold is False
    assert TaskType.MULTILABEL_CLASSIFICATION.supports_threshold is True
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_v1_task_types.py -v
```

**Step 3: Implement the properties**

In `src/metrics_lie/task_types.py`, add after the existing `is_classification` property:

```python
@property
def is_binary(self) -> bool:
    return self == TaskType.BINARY_CLASSIFICATION

@property
def is_regression(self) -> bool:
    return self == TaskType.REGRESSION

@property
def supports_threshold(self) -> bool:
    return self in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION)
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_v1_task_types.py -v
```

**Step 5: Commit**

```bash
git add src/metrics_lie/task_types.py tests/test_v1_task_types.py
git commit -m "feat: add is_binary, is_regression, supports_threshold to TaskType"
```

---

### Task 2: Add multiclass metric functions

**Files:**
- Modify: `src/metrics_lie/metrics/core.py`
- Create: `tests/test_v2_multiclass_metrics.py`

**Context:** Current `core.py` has 10 binary-only metric functions. We need multiclass variants: macro F1, weighted F1, macro precision, macro recall, macro AUC (OvR), Cohen's Kappa, and top-k accuracy. These functions take `y_true` (integer class labels) and either `y_score` (1D predicted labels from argmax) or `y_proba` (2D probability matrix).

**Important:** DO NOT modify existing binary metric functions. Add new functions and new entries to the `METRICS` dict.

**Step 1: Write the failing tests**

Create `tests/test_v2_multiclass_metrics.py`:

```python
"""Tests for multiclass metric functions."""
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.metrics.core import METRICS


@pytest.fixture
def multiclass_data():
    """3-class classification data."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 0, 2, 2, 0])
    # Probability matrix (10 samples, 3 classes)
    y_proba = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.5, 0.2],
        [0.1, 0.8, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.1, 0.8],
        [0.5, 0.2, 0.3],
        [0.8, 0.1, 0.1],
        [0.2, 0.2, 0.6],
        [0.1, 0.2, 0.7],
        [0.6, 0.3, 0.1],
    ])
    return y_true, y_pred, y_proba


def test_macro_f1_registered():
    assert "macro_f1" in METRICS


def test_weighted_f1_registered():
    assert "weighted_f1" in METRICS


def test_macro_precision_registered():
    assert "macro_precision" in METRICS


def test_macro_recall_registered():
    assert "macro_recall" in METRICS


def test_macro_auc_registered():
    assert "macro_auc" in METRICS


def test_cohens_kappa_registered():
    assert "cohens_kappa" in METRICS


def test_top_k_accuracy_registered():
    assert "top_k_accuracy" in METRICS


def test_macro_f1_value(multiclass_data):
    y_true, y_pred, _ = multiclass_data
    fn = METRICS["macro_f1"]
    result = fn(y_true, y_pred)
    assert 0.0 <= result <= 1.0


def test_weighted_f1_value(multiclass_data):
    y_true, y_pred, _ = multiclass_data
    fn = METRICS["weighted_f1"]
    result = fn(y_true, y_pred)
    assert 0.0 <= result <= 1.0


def test_macro_auc_value(multiclass_data):
    y_true, _, y_proba = multiclass_data
    fn = METRICS["macro_auc"]
    result = fn(y_true, y_proba)
    assert 0.0 <= result <= 1.0


def test_cohens_kappa_value(multiclass_data):
    y_true, y_pred, _ = multiclass_data
    fn = METRICS["cohens_kappa"]
    result = fn(y_true, y_pred)
    assert -1.0 <= result <= 1.0


def test_top_k_accuracy_value(multiclass_data):
    y_true, _, y_proba = multiclass_data
    fn = METRICS["top_k_accuracy"]
    # top-2 accuracy should be >= top-1 accuracy
    result = fn(y_true, y_proba)
    assert 0.0 <= result <= 1.0
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_v2_multiclass_metrics.py -v
```

**Step 3: Implement multiclass metric functions**

Add to `src/metrics_lie/metrics/core.py`:

```python
# Add these imports at the top (alongside existing sklearn imports):
from sklearn.metrics import (
    cohen_kappa_score,
    top_k_accuracy_score,
)

# --- Multiclass metric functions ---

def metric_macro_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro-averaged F1. y_score = predicted class labels."""
    y_pred = y_score.astype(int) if y_score.ndim == 1 else np.argmax(y_score, axis=1)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def metric_weighted_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Weighted-average F1. y_score = predicted class labels."""
    y_pred = y_score.astype(int) if y_score.ndim == 1 else np.argmax(y_score, axis=1)
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def metric_macro_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_pred = y_score.astype(int) if y_score.ndim == 1 else np.argmax(y_score, axis=1)
    return float(precision_score(y_true, y_pred, average="macro", zero_division=0))


def metric_macro_recall(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_pred = y_score.astype(int) if y_score.ndim == 1 else np.argmax(y_score, axis=1)
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))


def metric_macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro-averaged AUC via One-vs-Rest. y_score = probability matrix (n, K)."""
    return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))


def metric_cohens_kappa(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_pred = y_score.astype(int) if y_score.ndim == 1 else np.argmax(y_score, axis=1)
    return float(cohen_kappa_score(y_true, y_pred))


def metric_top_k_accuracy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Top-2 accuracy. y_score = probability matrix (n, K)."""
    k = min(2, y_score.shape[1]) if y_score.ndim == 2 else 1
    if k <= 1 or y_score.ndim == 1:
        y_pred = y_score.astype(int) if y_score.ndim == 1 else np.argmax(y_score, axis=1)
        return float(accuracy_score(y_true, y_pred))
    return float(top_k_accuracy_score(y_true, y_score, k=k))
```

Then add the multiclass metrics to the `METRICS` dict (append after existing entries) and add a new category set:

```python
# Multiclass metrics (no threshold — use argmax or probability matrix directly)
MULTICLASS_METRICS: set[str] = {
    "macro_f1", "weighted_f1", "macro_precision", "macro_recall",
    "macro_auc", "cohens_kappa", "top_k_accuracy",
}
```

Add to `METRICS` dict:

```python
    "macro_f1": metric_macro_f1,
    "weighted_f1": metric_weighted_f1,
    "macro_precision": metric_macro_precision,
    "macro_recall": metric_macro_recall,
    "macro_auc": metric_macro_auc,
    "cohens_kappa": metric_cohens_kappa,
    "top_k_accuracy": metric_top_k_accuracy,
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_v2_multiclass_metrics.py -v
```

**Step 5: Commit**

```bash
git add src/metrics_lie/metrics/core.py tests/test_v2_multiclass_metrics.py
git commit -m "feat: add multiclass metric functions (macro F1, AUC, Kappa, top-k)"
```

---

### Task 3: Add regression metric functions

**Files:**
- Modify: `src/metrics_lie/metrics/core.py`
- Create: `tests/test_v2_regression_metrics.py`

**Context:** Regression needs MAE, MSE, RMSE, R², and max error. These take continuous `y_true` and `y_score` (both floats). No threshold concept.

**Step 1: Write the failing tests**

Create `tests/test_v2_regression_metrics.py`:

```python
"""Tests for regression metric functions."""
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.metrics.core import METRICS, REGRESSION_METRICS


@pytest.fixture
def regression_data():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
    return y_true, y_pred


def test_mae_registered():
    assert "mae" in METRICS


def test_mse_registered():
    assert "mse" in METRICS


def test_rmse_registered():
    assert "rmse" in METRICS


def test_r2_registered():
    assert "r2" in METRICS


def test_max_error_registered():
    assert "max_error" in METRICS


def test_regression_metrics_category():
    expected = {"mae", "mse", "rmse", "r2", "max_error"}
    assert expected == REGRESSION_METRICS


def test_mae_value(regression_data):
    y_true, y_pred = regression_data
    result = METRICS["mae"](y_true, y_pred)
    assert result == pytest.approx(0.14, abs=0.01)


def test_mse_value(regression_data):
    y_true, y_pred = regression_data
    result = METRICS["mse"](y_true, y_pred)
    assert result >= 0.0


def test_rmse_value(regression_data):
    y_true, y_pred = regression_data
    mse = METRICS["mse"](y_true, y_pred)
    rmse = METRICS["rmse"](y_true, y_pred)
    assert rmse == pytest.approx(np.sqrt(mse), abs=1e-10)


def test_r2_value(regression_data):
    y_true, y_pred = regression_data
    result = METRICS["r2"](y_true, y_pred)
    assert result <= 1.0  # Can be negative for bad predictions


def test_max_error_value(regression_data):
    y_true, y_pred = regression_data
    result = METRICS["max_error"](y_true, y_pred)
    # max |y_true - y_pred| = |3.0 - 2.8| = 0.2
    assert result == pytest.approx(0.2, abs=0.01)
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_v2_regression_metrics.py -v
```

**Step 3: Implement regression metric functions**

Add to `src/metrics_lie/metrics/core.py`:

```python
# Add these imports at the top:
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    max_error as sklearn_max_error,
)

# --- Regression metric functions ---

def metric_mae(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_score))


def metric_mse(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_score))


def metric_rmse(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_score)))


def metric_r2(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(r2_score(y_true, y_score))


def metric_max_error(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(sklearn_max_error(y_true, y_score))
```

Add category set:

```python
REGRESSION_METRICS: set[str] = {"mae", "mse", "rmse", "r2", "max_error"}
```

Add to `METRICS` dict:

```python
    "mae": metric_mae,
    "mse": metric_mse,
    "rmse": metric_rmse,
    "r2": metric_r2,
    "max_error": metric_max_error,
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_v2_regression_metrics.py -v
```

**Step 5: Commit**

```bash
git add src/metrics_lie/metrics/core.py tests/test_v2_regression_metrics.py
git commit -m "feat: add regression metric functions (MAE, MSE, RMSE, R2, max_error)"
```

---

### Task 4: Add task_types field to MetricRequirement and extend registry

**Files:**
- Modify: `src/metrics_lie/metrics/registry.py`
- Create: `tests/test_v2_metric_registry.py`

**Context:** `MetricRequirement` currently has no `task_types` field. We need to add it so the resolver can filter metrics by task type. The field should be `frozenset[str] | None` where `None` means "applies to all task types".

**Step 1: Write the failing tests**

Create `tests/test_v2_metric_registry.py`:

```python
"""Tests for extended metric registry with task-type awareness."""
from __future__ import annotations

from metrics_lie.metrics.registry import METRIC_REQUIREMENTS, MetricRequirement
from metrics_lie.model.surface import SurfaceType


def test_metric_requirement_has_task_types_field():
    req = METRIC_REQUIREMENTS[0]
    assert hasattr(req, "task_types")


def test_binary_metrics_include_binary_task():
    binary_ids = {"accuracy", "auc", "f1", "precision", "recall", "logloss",
                  "brier_score", "ece", "pr_auc", "matthews_corrcoef"}
    for req in METRIC_REQUIREMENTS:
        if req.metric_id in binary_ids:
            assert req.task_types is None or "binary_classification" in req.task_types


def test_regression_metrics_exclude_binary():
    regression_ids = {"mae", "mse", "rmse", "r2", "max_error"}
    for req in METRIC_REQUIREMENTS:
        if req.metric_id in regression_ids:
            assert req.task_types is not None
            assert "binary_classification" not in req.task_types
            assert "regression" in req.task_types


def test_multiclass_metrics_include_multiclass():
    mc_ids = {"macro_f1", "weighted_f1", "macro_precision", "macro_recall",
              "macro_auc", "cohens_kappa", "top_k_accuracy"}
    for req in METRIC_REQUIREMENTS:
        if req.metric_id in mc_ids:
            assert req.task_types is not None
            assert "multiclass_classification" in req.task_types


def test_regression_requirements_use_continuous_surface():
    regression_ids = {"mae", "mse", "rmse", "r2", "max_error"}
    for req in METRIC_REQUIREMENTS:
        if req.metric_id in regression_ids:
            assert SurfaceType.CONTINUOUS in req.requires_surface


def test_all_metrics_have_requirements():
    """Every metric in METRICS dict should have a MetricRequirement entry."""
    from metrics_lie.metrics.core import METRICS
    registered_ids = {req.metric_id for req in METRIC_REQUIREMENTS}
    for metric_id in METRICS:
        assert metric_id in registered_ids, f"Missing MetricRequirement for {metric_id}"
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_v2_metric_registry.py -v
```

**Step 3: Implement**

In `src/metrics_lie/metrics/registry.py`:

1. Change the `MetricId` type to `str` (no longer restrictive Literal — the METRICS dict is the source of truth).
2. Add `task_types: frozenset[str] | None = None` to `MetricRequirement` (None = all tasks).
3. Update existing requirements to set `task_types=None` (keeping backward compat — binary metrics also work as-is for binary).
4. Add new requirements for multiclass and regression metrics.

```python
from __future__ import annotations

from dataclasses import dataclass, field

from metrics_lie.model.surface import SurfaceType


@dataclass(frozen=True)
class MetricRequirement:
    metric_id: str
    requires_surface: set[SurfaceType]
    requires_labels: bool
    min_samples: int
    requires_both_classes: bool
    task_types: frozenset[str] | None = None  # None = all task types


# --- Binary classification metrics (backward compatible: task_types=None) ---
METRIC_REQUIREMENTS: list[MetricRequirement] = [
    # Existing binary metrics — keep task_types=None for now
    # (they technically work for binary_classification only, but we don't
    # restrict yet since the surface-type check is sufficient for binary)
    MetricRequirement(
        metric_id="accuracy",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"binary_classification"}),
    ),
    MetricRequirement(
        metric_id="auc",
        requires_surface={SurfaceType.PROBABILITY, SurfaceType.SCORE},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"binary_classification"}),
    ),
    MetricRequirement(
        metric_id="pr_auc",
        requires_surface={SurfaceType.PROBABILITY, SurfaceType.SCORE},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"binary_classification"}),
    ),
    MetricRequirement(
        metric_id="f1",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"binary_classification"}),
    ),
    MetricRequirement(
        metric_id="precision",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"binary_classification"}),
    ),
    MetricRequirement(
        metric_id="recall",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"binary_classification"}),
    ),
    MetricRequirement(
        metric_id="logloss",
        requires_surface={SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"binary_classification"}),
    ),
    MetricRequirement(
        metric_id="brier_score",
        requires_surface={SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"binary_classification"}),
    ),
    MetricRequirement(
        metric_id="ece",
        requires_surface={SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"binary_classification"}),
    ),
    MetricRequirement(
        metric_id="matthews_corrcoef",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"binary_classification"}),
    ),

    # --- Multiclass classification metrics ---
    MetricRequirement(
        metric_id="macro_f1",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"multiclass_classification"}),
    ),
    MetricRequirement(
        metric_id="weighted_f1",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"multiclass_classification"}),
    ),
    MetricRequirement(
        metric_id="macro_precision",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"multiclass_classification"}),
    ),
    MetricRequirement(
        metric_id="macro_recall",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"multiclass_classification"}),
    ),
    MetricRequirement(
        metric_id="macro_auc",
        requires_surface={SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"multiclass_classification"}),
    ),
    MetricRequirement(
        metric_id="cohens_kappa",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"multiclass_classification"}),
    ),
    MetricRequirement(
        metric_id="top_k_accuracy",
        requires_surface={SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"multiclass_classification"}),
    ),

    # --- Regression metrics ---
    MetricRequirement(
        metric_id="mae",
        requires_surface={SurfaceType.CONTINUOUS},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=False,
        task_types=frozenset({"regression"}),
    ),
    MetricRequirement(
        metric_id="mse",
        requires_surface={SurfaceType.CONTINUOUS},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=False,
        task_types=frozenset({"regression"}),
    ),
    MetricRequirement(
        metric_id="rmse",
        requires_surface={SurfaceType.CONTINUOUS},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=False,
        task_types=frozenset({"regression"}),
    ),
    MetricRequirement(
        metric_id="r2",
        requires_surface={SurfaceType.CONTINUOUS},
        requires_labels=True,
        min_samples=2,
        requires_both_classes=False,
        task_types=frozenset({"regression"}),
    ),
    MetricRequirement(
        metric_id="max_error",
        requires_surface={SurfaceType.CONTINUOUS},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=False,
        task_types=frozenset({"regression"}),
    ),
]
```

Note: Remove the `MetricId` Literal type alias entirely — `metric_id` is now `str`.

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_v2_metric_registry.py -v
```

Also run existing registry tests to verify no regression:

```bash
python -m pytest tests/test33_metric_applicability.py -v
```

**Step 5: Commit**

```bash
git add src/metrics_lie/metrics/registry.py tests/test_v2_metric_registry.py
git commit -m "feat: add task_types to MetricRequirement, register multiclass and regression metrics"
```

---

### Task 5: Add task-type filtering to MetricResolver

**Files:**
- Modify: `src/metrics_lie/metrics/applicability.py`
- Create: `tests/test_v2_metric_resolver.py`

**Context:** `MetricResolver.resolve()` accepts `task_type` but ignores it. We need to add a filter step: if `req.task_types is not None and task_type not in req.task_types`, exclude the metric. Also `DatasetProperties` needs `n_classes` for multiclass and needs to not require `n_positive`/`n_negative` for regression.

**Step 1: Write the failing tests**

Create `tests/test_v2_metric_resolver.py`:

```python
"""Tests for task-type-aware MetricResolver."""
from __future__ import annotations

import pytest

from metrics_lie.metrics.applicability import DatasetProperties, MetricResolver
from metrics_lie.model.surface import SurfaceType


def test_resolver_binary_returns_only_binary_metrics():
    resolver = MetricResolver()
    props = DatasetProperties(
        n_samples=100, n_positive=50, n_negative=50,
        has_subgroups=False, positive_rate=0.5,
    )
    result = resolver.resolve(
        task_type="binary_classification",
        surface_type=SurfaceType.PROBABILITY,
        dataset_props=props,
    )
    regression_metrics = {"mae", "mse", "rmse", "r2", "max_error"}
    multiclass_metrics = {"macro_f1", "weighted_f1", "macro_precision",
                          "macro_recall", "macro_auc", "cohens_kappa", "top_k_accuracy"}
    for m in regression_metrics | multiclass_metrics:
        assert m not in result.metrics, f"{m} should not be in binary results"


def test_resolver_multiclass_returns_multiclass_metrics():
    resolver = MetricResolver()
    props = DatasetProperties(
        n_samples=100, n_positive=0, n_negative=0,
        has_subgroups=False, positive_rate=0.0,
        n_classes=3,
    )
    result = resolver.resolve(
        task_type="multiclass_classification",
        surface_type=SurfaceType.PROBABILITY,
        dataset_props=props,
    )
    assert "macro_f1" in result.metrics
    assert "macro_auc" in result.metrics
    # Binary-only metrics should be excluded
    assert "auc" not in result.metrics
    assert "f1" not in result.metrics


def test_resolver_regression_returns_regression_metrics():
    resolver = MetricResolver()
    props = DatasetProperties(
        n_samples=100, n_positive=0, n_negative=0,
        has_subgroups=False, positive_rate=0.0,
    )
    result = resolver.resolve(
        task_type="regression",
        surface_type=SurfaceType.CONTINUOUS,
        dataset_props=props,
    )
    assert "mae" in result.metrics
    assert "mse" in result.metrics
    assert "rmse" in result.metrics
    assert "r2" in result.metrics
    assert "max_error" in result.metrics
    # No classification metrics
    assert "auc" not in result.metrics
    assert "f1" not in result.metrics
    assert "macro_f1" not in result.metrics


def test_resolver_regression_excludes_imbalance_warnings():
    """Regression should not get severe_imbalance_warning."""
    resolver = MetricResolver()
    props = DatasetProperties(
        n_samples=100, n_positive=0, n_negative=0,
        has_subgroups=False, positive_rate=0.0,
    )
    result = resolver.resolve(
        task_type="regression",
        surface_type=SurfaceType.CONTINUOUS,
        dataset_props=props,
    )
    assert "severe_imbalance_warning" not in result.warnings


def test_dataset_properties_accepts_n_classes():
    props = DatasetProperties(
        n_samples=100, n_positive=30, n_negative=70,
        has_subgroups=False, positive_rate=0.3, n_classes=3,
    )
    assert props.n_classes == 3


def test_dataset_properties_defaults_n_classes_none():
    props = DatasetProperties(
        n_samples=100, n_positive=50, n_negative=50,
        has_subgroups=False, positive_rate=0.5,
    )
    assert props.n_classes is None
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_v2_metric_resolver.py -v
```

**Step 3: Implement**

Modify `src/metrics_lie/metrics/applicability.py`:

1. Add `n_classes: int | None = None` to `DatasetProperties`.
2. In `resolve()`, add a task-type filter check in the loop.
3. Gate the `requires_both_classes` check on `task_type != "regression"`.
4. Gate the `severe_imbalance_warning` on classification task types.

```python
@dataclass(frozen=True)
class DatasetProperties:
    n_samples: int
    n_positive: int
    n_negative: int
    has_subgroups: bool
    positive_rate: float
    n_classes: int | None = None
```

In the `resolve` loop, add after the surface check:

```python
        for req in self._requirements:
            # Task-type filter
            if req.task_types is not None and task_type not in req.task_types:
                excluded.append((req.metric_id, f"not applicable to task_type={task_type}"))
                continue

            if surface_type not in req.requires_surface:
                # ... existing code
```

Gate the binary-specific warnings:

```python
        is_classification = task_type in (
            "binary_classification", "multiclass_classification",
            "multilabel_classification",
        )

        if dataset_props.n_samples < 30:
            warnings.append("low_sample_warning")
        if is_classification and (
            dataset_props.positive_rate < 0.05 or dataset_props.positive_rate > 0.95
        ):
            warnings.append("severe_imbalance_warning")
            # ... existing pr_auc force-add logic
```

Also gate `requires_both_classes` check:

```python
            if req.requires_both_classes and task_type != "regression" and (
                dataset_props.n_positive == 0 or dataset_props.n_negative == 0
            ):
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_v2_metric_resolver.py tests/test33_metric_applicability.py -v
```

**Step 5: Commit**

```bash
git add src/metrics_lie/metrics/applicability.py tests/test_v2_metric_resolver.py
git commit -m "feat: add task-type filtering to MetricResolver, extend DatasetProperties"
```

---

### Task 6: Generalize label_noise scenario for multiclass and regression

**Files:**
- Modify: `src/metrics_lie/scenarios/label_noise.py`
- Create: `tests/test_v2_label_noise.py`

**Context:** Current `label_noise` does `1 - y[flips]` which only works for binary {0,1}. For multiclass: randomly reassign flipped labels to a different valid class. For regression: add Gaussian noise to target values.

**Step 1: Write the failing tests**

Create `tests/test_v2_label_noise.py`:

```python
"""Tests for generalized label_noise scenario."""
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.scenarios.label_noise import LabelNoiseScenario
from metrics_lie.scenarios.base import ScenarioContext


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_binary_unchanged(rng):
    """Binary behavior is preserved."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    ctx = ScenarioContext(task="binary_classification")
    scenario = LabelNoiseScenario(p=0.3)
    y_out, s_out = scenario.apply(y_true, y_score, rng, ctx)
    assert set(np.unique(y_out)).issubset({0, 1})
    assert np.array_equal(s_out, y_score)


def test_multiclass_flips_to_different_class(rng):
    """Multiclass flips should produce different valid class labels."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_score = np.zeros(10)  # dummy
    ctx = ScenarioContext(task="multiclass_classification")
    scenario = LabelNoiseScenario(p=0.5)
    y_out, _ = scenario.apply(y_true, y_score, rng, ctx)
    # All output labels should be valid classes
    assert set(np.unique(y_out)).issubset({0, 1, 2})
    # At least some labels should have changed
    assert not np.array_equal(y_out, y_true)
    # Flipped labels should be different from originals
    changed = y_out != y_true
    assert np.all(y_out[changed] != y_true[changed])


def test_regression_adds_noise(rng):
    """Regression label noise adds Gaussian noise to targets."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_score = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    ctx = ScenarioContext(task="regression")
    scenario = LabelNoiseScenario(p=0.1)  # p controls noise magnitude for regression
    y_out, s_out = scenario.apply(y_true, y_score, rng, ctx)
    # y_true should be modified (noise added)
    assert not np.allclose(y_out, y_true)
    # y_score should be unchanged
    assert np.array_equal(s_out, y_score)
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_v2_label_noise.py -v
```

**Step 3: Implement**

Modify `src/metrics_lie/scenarios/label_noise.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from metrics_lie.scenarios.base import ScenarioContext


@dataclass(frozen=True)
class LabelNoiseScenario:
    id: str = "label_noise"
    p: float = 0.1

    def apply(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        rng: np.random.Generator,
        ctx: ScenarioContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not (0.0 <= self.p <= 0.5):
            raise ValueError("label_noise.p must be in [0, 0.5]")

        y = y_true.copy()

        if ctx.task == "regression":
            # For regression: add Gaussian noise proportional to target std
            std = float(np.std(y)) if np.std(y) > 0 else 1.0
            noise = rng.normal(loc=0.0, scale=self.p * std, size=y.shape[0])
            y = y.astype(float) + noise
        elif ctx.task == "multiclass_classification":
            # For multiclass: flip to a random different valid class
            classes = np.unique(y_true)
            flips = rng.random(size=y.shape[0]) < self.p
            for i in np.where(flips)[0]:
                other_classes = classes[classes != y[i]]
                if len(other_classes) > 0:
                    y[i] = rng.choice(other_classes)
        else:
            # Binary classification (original behavior)
            flips = rng.random(size=y.shape[0]) < self.p
            y[flips] = 1 - y[flips]

        return y, y_score

    def describe(self) -> Dict[str, Any]:
        return {"id": self.id, "p": self.p}
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_v2_label_noise.py -v
```

Also run existing tests to verify binary behavior unchanged:

```bash
python -m pytest tests/ -k "label_noise or label_ingest or golden" -v
```

**Step 5: Commit**

```bash
git add src/metrics_lie/scenarios/label_noise.py tests/test_v2_label_noise.py
git commit -m "feat: generalize label_noise for multiclass and regression tasks"
```

---

### Task 7: Generalize score_noise scenario for multiclass and regression

**Files:**
- Modify: `src/metrics_lie/scenarios/score_noise.py`
- Create: `tests/test_v2_score_noise.py`

**Context:** Current `score_noise` uses `rng.normal(size=s.shape[0])` which only works for 1D. For multiclass probability matrices (2D), noise must be added per-element and rows re-normalized. For regression, no clipping should apply.

**Step 1: Write the failing tests**

Create `tests/test_v2_score_noise.py`:

```python
"""Tests for generalized score_noise scenario."""
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.scenarios.score_noise import ScoreNoiseScenario
from metrics_lie.scenarios.base import ScenarioContext


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_binary_unchanged(rng):
    """Binary probability behavior preserved."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.3, 0.7, 0.9])
    ctx = ScenarioContext(task="binary_classification", surface_type="probability")
    scenario = ScoreNoiseScenario(sigma=0.05)
    _, s_out = scenario.apply(y_true, y_score, rng, ctx)
    assert s_out.ndim == 1
    assert np.all(s_out >= 0.0) and np.all(s_out <= 1.0)


def test_multiclass_2d_preserved(rng):
    """Multiclass probability matrix stays 2D with valid row sums."""
    y_true = np.array([0, 1, 2])
    y_proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]])
    ctx = ScenarioContext(task="multiclass_classification", surface_type="probability")
    scenario = ScoreNoiseScenario(sigma=0.05)
    _, s_out = scenario.apply(y_true, y_proba, rng, ctx)
    assert s_out.ndim == 2
    assert s_out.shape == (3, 3)
    # Rows should sum to ~1.0
    np.testing.assert_allclose(s_out.sum(axis=1), 1.0, atol=1e-10)
    # All values should be >= 0
    assert np.all(s_out >= 0.0)


def test_regression_no_clip(rng):
    """Regression scores should not be clipped to [0, 1]."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_score = np.array([105.0, 195.0, 310.0])
    ctx = ScenarioContext(task="regression", surface_type="continuous")
    scenario = ScoreNoiseScenario(sigma=5.0)
    _, s_out = scenario.apply(y_true, y_score, rng, ctx)
    # Values should NOT be clipped to [0, 1]
    assert s_out.max() > 1.0 or s_out.min() < 0.0
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_v2_score_noise.py -v
```

**Step 3: Implement**

Modify `src/metrics_lie/scenarios/score_noise.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from metrics_lie.scenarios.base import ScenarioContext


@dataclass(frozen=True)
class ScoreNoiseScenario:
    id: str = "score_noise"
    sigma: float = 0.05

    def apply(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        rng: np.random.Generator,
        ctx: ScenarioContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.sigma < 0:
            raise ValueError("score_noise.sigma must be >= 0")
        s = y_score.astype(float).copy()
        s = s + rng.normal(loc=0.0, scale=self.sigma, size=s.shape)
        if ctx.surface_type == "probability":
            s = np.clip(s, 0.0, 1.0)
            # For 2D probability matrices, re-normalize rows to sum to 1
            if s.ndim == 2:
                row_sums = s.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1.0, row_sums)
                s = s / row_sums
        return y_true, s

    def describe(self) -> Dict[str, Any]:
        return {"id": self.id, "sigma": self.sigma}
```

Key change: `size=s.shape` instead of `size=s.shape[0]`, plus 2D normalization.

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_v2_score_noise.py -v
```

Also run existing tests:

```bash
python -m pytest tests/ -k "score_noise or score_ingest or golden" -v
```

**Step 5: Commit**

```bash
git add src/metrics_lie/scenarios/score_noise.py tests/test_v2_score_noise.py
git commit -m "feat: generalize score_noise for 2D multiclass probabilities and regression"
```

---

### Task 8: Generalize class_imbalance scenario for multiclass

**Files:**
- Modify: `src/metrics_lie/scenarios/class_imbalance.py`
- Create: `tests/test_v2_class_imbalance.py`

**Context:** Current `class_imbalance` uses `y_true == 1` (binary only). For multiclass, subsample a specified target class. For regression, this scenario is not applicable.

**Step 1: Write the failing tests**

Create `tests/test_v2_class_imbalance.py`:

```python
"""Tests for generalized class_imbalance scenario."""
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.scenarios.class_imbalance import ClassImbalanceScenario
from metrics_lie.scenarios.base import ScenarioContext


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_binary_unchanged(rng):
    """Binary behavior is preserved."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.15, 0.25, 0.8, 0.9, 0.7, 0.85, 0.75])
    ctx = ScenarioContext(task="binary_classification")
    scenario = ClassImbalanceScenario(target_pos_rate=0.2)
    y_out, s_out = scenario.apply(y_true, y_score, rng, ctx)
    assert len(y_out) < len(y_true)
    assert set(np.unique(y_out)).issubset({0, 1})


def test_multiclass_reduces_majority_class(rng):
    """Multiclass: subsample to shift class distribution."""
    # 30 class 0, 30 class 1, 30 class 2
    y_true = np.array([0]*30 + [1]*30 + [2]*30)
    y_score = np.zeros(90)
    ctx = ScenarioContext(task="multiclass_classification")
    scenario = ClassImbalanceScenario(target_pos_rate=0.1)
    y_out, s_out = scenario.apply(y_true, y_score, rng, ctx)
    # Should have fewer samples
    assert len(y_out) <= len(y_true)
    # All output labels should be valid classes
    assert set(np.unique(y_out)).issubset({0, 1, 2})


def test_regression_returns_unchanged(rng):
    """Regression: class_imbalance is a no-op, returns data unchanged."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_score = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    ctx = ScenarioContext(task="regression")
    scenario = ClassImbalanceScenario(target_pos_rate=0.2)
    y_out, s_out = scenario.apply(y_true, y_score, rng, ctx)
    # Should return unchanged for regression
    assert np.array_equal(y_out, y_true)
    assert np.array_equal(s_out, y_score)
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_v2_class_imbalance.py -v
```

**Step 3: Implement**

Read the current `class_imbalance.py` first, then modify it to branch on `ctx.task`:

- **Binary** (`binary_classification`): Keep existing behavior exactly.
- **Multiclass** (`multiclass_classification`): Find the largest class and subsample it. Use `target_pos_rate` as the target fraction for the subsampled class (reinterpreted as "target fraction for the majority class").
- **Regression**: Return data unchanged (no-op).

The subagent should read the full current implementation before modifying, preserving the existing binary logic.

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_v2_class_imbalance.py -v
```

Also run existing tests:

```bash
python -m pytest tests/ -k "class_imbalance or imbalance or golden" -v
```

**Step 5: Commit**

```bash
git add src/metrics_lie/scenarios/class_imbalance.py tests/test_v2_class_imbalance.py
git commit -m "feat: generalize class_imbalance for multiclass, no-op for regression"
```

---

### Task 9: Add SCENARIO_TASK_COMPAT table and filter

**Files:**
- Modify: `src/metrics_lie/surface_compat.py`
- Create: `tests/test_v2_scenario_task_compat.py`

**Context:** `surface_compat.py` has `SCENARIO_SURFACE_COMPAT` but no task-type-level compatibility table. We need `SCENARIO_TASK_COMPAT` so the execution pipeline can filter scenarios that don't apply to the current task type (e.g., `threshold_gaming` doesn't apply to regression).

**Step 1: Write the failing tests**

Create `tests/test_v2_scenario_task_compat.py`:

```python
"""Tests for scenario-task-type compatibility filtering."""
from __future__ import annotations

from metrics_lie.surface_compat import SCENARIO_TASK_COMPAT, filter_compatible_scenarios_by_task


def test_compat_table_exists():
    assert isinstance(SCENARIO_TASK_COMPAT, dict)


def test_regression_excludes_threshold_gaming():
    assert "threshold_gaming" not in SCENARIO_TASK_COMPAT["regression"]


def test_regression_excludes_class_imbalance():
    assert "class_imbalance" not in SCENARIO_TASK_COMPAT["regression"]


def test_regression_includes_label_noise():
    assert "label_noise" in SCENARIO_TASK_COMPAT["regression"]


def test_regression_includes_score_noise():
    assert "score_noise" in SCENARIO_TASK_COMPAT["regression"]


def test_binary_includes_all_four():
    compat = SCENARIO_TASK_COMPAT["binary_classification"]
    assert {"label_noise", "score_noise", "class_imbalance", "threshold_gaming"} == compat


def test_multiclass_excludes_threshold_gaming():
    assert "threshold_gaming" not in SCENARIO_TASK_COMPAT["multiclass_classification"]


def test_filter_function():
    class FakeScenario:
        def __init__(self, sid):
            self.id = sid
    scenarios = [FakeScenario("label_noise"), FakeScenario("threshold_gaming"),
                 FakeScenario("score_noise")]
    kept, skipped = filter_compatible_scenarios_by_task(scenarios, "regression")
    assert [s.id for s in kept] == ["label_noise", "score_noise"]
    assert skipped == ["threshold_gaming"]
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_v2_scenario_task_compat.py -v
```

**Step 3: Implement**

Add to `src/metrics_lie/surface_compat.py`:

```python
SCENARIO_TASK_COMPAT: dict[str, set[str]] = {
    "binary_classification": {"label_noise", "score_noise", "class_imbalance", "threshold_gaming"},
    "multiclass_classification": {"label_noise", "score_noise", "class_imbalance"},
    "multilabel_classification": {"label_noise", "class_imbalance"},
    "regression": {"label_noise", "score_noise"},
    "ranking": {"label_noise", "score_noise"},
}


def filter_compatible_scenarios_by_task(
    scenarios: Sequence[Any],
    task_type: str,
) -> tuple[list[Any], list[str]]:
    """Filter scenarios by task-type compatibility."""
    allowed = SCENARIO_TASK_COMPAT.get(task_type, set())
    compatible = [s for s in scenarios if s.id in allowed]
    skipped = [s.id for s in scenarios if s.id not in allowed]
    return compatible, skipped
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_v2_scenario_task_compat.py -v
```

**Step 5: Commit**

```bash
git add src/metrics_lie/surface_compat.py tests/test_v2_scenario_task_compat.py
git commit -m "feat: add SCENARIO_TASK_COMPAT table and filter_compatible_scenarios_by_task"
```

---

### Task 10: Add compute_metric routing for multiclass and regression

**Files:**
- Modify: `src/metrics_lie/metrics/core.py`
- Create: `tests/test_v2_compute_metric.py`

**Context:** `compute_metric()` currently routes by `metric_id in THRESHOLD_METRICS` only. For multiclass metrics that take 2D probability matrices (like `macro_auc`, `top_k_accuracy`), the function must NOT apply a threshold. For regression metrics, no threshold either. The key insight: `MULTICLASS_METRICS` and `REGRESSION_METRICS` should be treated like non-threshold metrics (just `fn(y_true, y_score)`).

**Step 1: Write the failing tests**

Create `tests/test_v2_compute_metric.py`:

```python
"""Tests for compute_metric routing with new metric categories."""
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.metrics.core import compute_metric, METRICS


def test_compute_regression_metric():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.0, 3.2])
    result = compute_metric("mae", METRICS["mae"], y_true, y_pred)
    assert result == pytest.approx(0.1, abs=0.01)


def test_compute_multiclass_label_metric():
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 1, 0])
    result = compute_metric("macro_f1", METRICS["macro_f1"], y_true, y_pred)
    assert 0.0 <= result <= 1.0


def test_compute_binary_threshold_metric_still_works():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.2, 0.4, 0.6, 0.8])
    result = compute_metric("accuracy", METRICS["accuracy"], y_true, y_score, threshold=0.5)
    assert result == 1.0
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_v2_compute_metric.py -v
```

**Step 3: Implement**

The current `compute_metric` already works correctly for this case — `MULTICLASS_METRICS` and `REGRESSION_METRICS` are not in `THRESHOLD_METRICS`, so they fall through to the `return float(metric_fn(y_true, y_score))` branch. The tests should pass without code changes.

If tests pass, this task validates the routing is already correct. If they fail, add logic.

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_v2_compute_metric.py -v
```

**Step 5: Commit**

```bash
git add tests/test_v2_compute_metric.py
git commit -m "test: verify compute_metric routing works for multiclass and regression"
```

---

### Task 11: Add task_type to ResultBundle schema

**Files:**
- Modify: `src/metrics_lie/schema.py`
- Create: `tests/test_v2_schema.py`

**Context:** `ResultBundle` has no `task_type` field. Add it with a default of `"binary_classification"` for backward compatibility.

**Step 1: Write the failing tests**

Create `tests/test_v2_schema.py`:

```python
"""Tests for ResultBundle schema extension."""
from __future__ import annotations

from metrics_lie.schema import ResultBundle


def test_result_bundle_has_task_type():
    bundle = ResultBundle(
        run_id="test",
        experiment_name="test",
        metric_name="auc",
    )
    assert bundle.task_type == "binary_classification"


def test_result_bundle_accepts_regression():
    bundle = ResultBundle(
        run_id="test",
        experiment_name="test",
        metric_name="mae",
        task_type="regression",
    )
    assert bundle.task_type == "regression"


def test_result_bundle_schema_version():
    bundle = ResultBundle(
        run_id="test",
        experiment_name="test",
        metric_name="auc",
    )
    assert bundle.schema_version == "0.2"
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_v2_schema.py -v
```

**Step 3: Implement**

In `src/metrics_lie/schema.py`, modify `ResultBundle`:

```python
class ResultBundle(BaseModel):
    schema_version: str = "0.2"
    run_id: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    experiment_name: str
    metric_name: str
    task_type: str = "binary_classification"
    # ... rest unchanged
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_v2_schema.py -v
```

Also check existing tests still pass (the `schema_version` change may affect golden tests):

```bash
python -m pytest tests/test_milestone3_golden.py -v
```

If the golden test compares `schema_version`, update the golden file.

**Step 5: Commit**

```bash
git add src/metrics_lie/schema.py tests/test_v2_schema.py
git commit -m "feat: add task_type to ResultBundle, bump schema_version to 0.2"
```

---

### Task 12: Wire multi-task execution pipeline

**Files:**
- Modify: `src/metrics_lie/execution.py`
- Create: `tests/test_v2_execution_multiclass.py`
- Create: `tests/test_v2_execution_regression.py`

**Context:** This is the critical integration task. `execution.py` currently uses `load_binary_csv`, `y_true.to_numpy(dtype=int)`, and hardcoded binary assumptions. Wire in `load_dataset`, task-type-conditional label casting, task-type-aware metric resolution, scenario filtering by task type, and analysis gating.

**Changes needed in `execution.py`:**

1. Replace `from metrics_lie.datasets.loaders import load_binary_csv` with `from metrics_lie.datasets.loaders import load_binary_csv, load_dataset`.
2. After `spec = load_experiment_spec(spec_dict)`, determine the task type: `task_type = TaskType(spec.task)`.
3. For dataset loading: if `task_type.is_binary`, use existing `load_binary_csv` path. Otherwise, use `load_dataset(task_type=spec.task, ...)`.
4. For label casting: `y_true.to_numpy(dtype=int)` for classification; `y_true.to_numpy(dtype=float)` for regression.
5. For metric validation: `if spec.metric not in METRICS` stays, but METRICS now includes all metric types.
6. For `DatasetProperties`: compute `n_classes` for multiclass; use zeros for `n_positive`/`n_negative` for regression.
7. For scenario filtering: also filter by task type using `filter_compatible_scenarios_by_task`.
8. For analysis (threshold sweep, disagreement): skip for non-binary tasks.
9. For `ResultBundle`: pass `task_type=spec.task`.
10. For calibration diagnostics: skip `brier`/`ece` for regression.

**Step 1: Write the failing tests**

Create `tests/test_v2_execution_multiclass.py`:

```python
"""Tests for multiclass execution pipeline."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from metrics_lie.db.session import DB_PATH, engine, init_db
from metrics_lie.execution import run_from_spec_dict
from metrics_lie.schema import ResultBundle
from metrics_lie.utils.paths import get_run_dir


@pytest.fixture(autouse=True)
def fresh_db():
    engine.dispose()
    if DB_PATH.exists():
        DB_PATH.unlink()
    init_db()


def test_multiclass_execution(tmp_path: Path):
    """Run multiclass classification through the full pipeline."""
    rng = np.random.default_rng(42)
    n = 60
    X = rng.standard_normal((n, 2))
    y = np.array([0]*20 + [1]*20 + [2]*20)
    model = LogisticRegression(random_state=0, max_iter=200).fit(X, y)
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    df = pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "y_true": y, "y_score": np.zeros(n)})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    spec = {
        "name": "multiclass_test",
        "task": "multiclass_classification",
        "dataset": {
            "source": "csv",
            "path": str(csv_path),
            "y_true_col": "y_true",
            "y_score_col": "y_score",
            "feature_cols": ["f1", "f2"],
        },
        "metric": "macro_f1",
        "model_source": {"kind": "pickle", "path": str(model_path)},
        "scenarios": [
            {"id": "label_noise", "params": {"p": 0.1}},
            {"id": "score_noise", "params": {"sigma": 0.05}},
        ],
        "n_trials": 5,
        "seed": 42,
    }

    run_id = run_from_spec_dict(spec)
    bundle_json = get_run_dir(run_id).results_json.read_text(encoding="utf-8")
    bundle = ResultBundle.model_validate_json(bundle_json)

    assert bundle.task_type == "multiclass_classification"
    assert bundle.metric_name == "macro_f1"
    assert "macro_f1" in bundle.applicable_metrics
    # Binary-only metrics should NOT be present
    assert "auc" not in bundle.applicable_metrics
    assert "f1" not in bundle.applicable_metrics
```

Create `tests/test_v2_execution_regression.py`:

```python
"""Tests for regression execution pipeline."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from metrics_lie.db.session import DB_PATH, engine, init_db
from metrics_lie.execution import run_from_spec_dict
from metrics_lie.schema import ResultBundle
from metrics_lie.utils.paths import get_run_dir


@pytest.fixture(autouse=True)
def fresh_db():
    engine.dispose()
    if DB_PATH.exists():
        DB_PATH.unlink()
    init_db()


def test_regression_execution(tmp_path: Path):
    """Run regression through the full pipeline."""
    rng = np.random.default_rng(42)
    n = 50
    X = rng.standard_normal((n, 2))
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + rng.normal(0, 0.5, n)
    model = LinearRegression().fit(X, y)
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    df = pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "y_true": y, "y_score": np.zeros(n)})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    spec = {
        "name": "regression_test",
        "task": "regression",
        "dataset": {
            "source": "csv",
            "path": str(csv_path),
            "y_true_col": "y_true",
            "y_score_col": "y_score",
            "feature_cols": ["f1", "f2"],
        },
        "metric": "mae",
        "model_source": {"kind": "pickle", "path": str(model_path)},
        "scenarios": [
            {"id": "label_noise", "params": {"p": 0.1}},
            {"id": "score_noise", "params": {"sigma": 0.5}},
        ],
        "n_trials": 5,
        "seed": 42,
    }

    run_id = run_from_spec_dict(spec)
    bundle_json = get_run_dir(run_id).results_json.read_text(encoding="utf-8")
    bundle = ResultBundle.model_validate_json(bundle_json)

    assert bundle.task_type == "regression"
    assert bundle.metric_name == "mae"
    assert "mae" in bundle.applicable_metrics
    assert "mse" in bundle.applicable_metrics
    assert "rmse" in bundle.applicable_metrics
    # Classification metrics should NOT be present
    assert "auc" not in bundle.applicable_metrics
    assert "f1" not in bundle.applicable_metrics
    assert "macro_f1" not in bundle.applicable_metrics
    # No threshold sweep for regression
    assert "threshold_sweep" not in bundle.analysis_artifacts
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_v2_execution_multiclass.py tests/test_v2_execution_regression.py -v
```

**Step 3: Implement the execution pipeline changes**

This is the largest single task. The subagent should:

1. Read the current `execution.py` completely.
2. Add `from metrics_lie.datasets.loaders import load_dataset` import.
3. Add `from metrics_lie.surface_compat import filter_compatible_scenarios_by_task` import.
4. After `spec = load_experiment_spec(spec_dict)`, add: `task_type = TaskType(spec.task)`.
5. Replace the dataset loading section: if `task_type.is_binary`, use existing `load_binary_csv` path. Otherwise use `load_dataset(path=..., task_type=spec.task, ...)`.
6. For label casting: `dtype=float if task_type.is_regression else int`.
7. For `DatasetProperties`:
   - If regression: `n_positive=0, n_negative=0, positive_rate=0.0`
   - If multiclass: `n_positive=0, n_negative=0, positive_rate=0.0, n_classes=len(np.unique(y_true))`
   - If binary: keep existing
8. For scenario filtering: add `filter_compatible_scenarios_by_task` after surface filtering.
9. For analysis gating: wrap threshold sweep/sensitivity/disagreement in `if task_type.is_binary:`.
10. For `ResultBundle`: add `task_type=spec.task`.
11. For model adapter surface selection: for regression, prefer `SurfaceType.CONTINUOUS`.
12. For runner `ctx`: pass surface_type correctly.

**Critical:** Keep all existing binary classification behavior EXACTLY the same. Only add new branches for multiclass/regression.

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_v2_execution_multiclass.py tests/test_v2_execution_regression.py -v
```

Also run full suite to verify no regression:

```bash
python -m pytest tests/ -v
```

**Step 5: Commit**

```bash
git add src/metrics_lie/execution.py tests/test_v2_execution_multiclass.py tests/test_v2_execution_regression.py
git commit -m "feat: wire multi-task execution pipeline for multiclass and regression"
```

---

### Task 13: Gate runner diagnostics on task type

**Files:**
- Modify: `src/metrics_lie/runner.py`
- Modify: `src/metrics_lie/scenarios/base.py` (add `n_classes` to `ScenarioContext`)
- Create: `tests/test_v2_runner.py`

**Context:** `runner.py` unconditionally computes `brier_score`/`ece` for probability surfaces and gaming diagnostics for `accuracy`. These should be gated: brier/ece only for binary classification (with probability surface), gaming only for binary accuracy.

**Step 1: Write the failing tests**

Create `tests/test_v2_runner.py`:

```python
"""Tests for task-type-aware runner diagnostics."""
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.metrics.core import METRICS
from metrics_lie.runner import RunConfig, run_scenarios
from metrics_lie.scenarios.base import ScenarioContext


def test_regression_runner_no_brier_ece():
    """Regression runner should not compute brier/ece diagnostics."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_score = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    ctx = ScenarioContext(task="regression", surface_type="continuous")
    results = run_scenarios(
        y_true=y_true,
        y_score=y_score,
        metric_name="mae",
        metric_fn=METRICS["mae"],
        scenario_specs=[{"id": "label_noise", "params": {"p": 0.1}}],
        cfg=RunConfig(n_trials=3, seed=42),
        ctx=ctx,
    )
    assert len(results) == 1
    assert "brier" not in results[0].diagnostics
    assert "ece" not in results[0].diagnostics


def test_multiclass_runner_no_gaming():
    """Multiclass runner should not compute accuracy gaming diagnostics."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    ctx = ScenarioContext(task="multiclass_classification", surface_type="label")
    results = run_scenarios(
        y_true=y_true,
        y_score=y_pred,
        metric_name="macro_f1",
        metric_fn=METRICS["macro_f1"],
        scenario_specs=[{"id": "label_noise", "params": {"p": 0.1}}],
        cfg=RunConfig(n_trials=3, seed=42),
        ctx=ctx,
    )
    assert len(results) == 1
    assert "metric_inflation" not in results[0].diagnostics


def test_scenario_context_has_n_classes():
    ctx = ScenarioContext(task="multiclass_classification", n_classes=3)
    assert ctx.n_classes == 3

    ctx_default = ScenarioContext()
    assert ctx_default.n_classes is None
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_v2_runner.py -v
```

**Step 3: Implement**

In `src/metrics_lie/scenarios/base.py`, add `n_classes` to `ScenarioContext`:

```python
@dataclass(frozen=True)
class ScenarioContext:
    task: str = "binary_classification"
    surface_type: str = "probability"
    n_classes: int | None = None
```

In `src/metrics_lie/runner.py`, gate the diagnostics:

1. For brier/ece: change `if ctx.surface_type == "probability":` to `if ctx.surface_type == "probability" and ctx.task == "binary_classification":`.
2. For gaming: change `if metric_name == "accuracy":` to `if metric_name == "accuracy" and ctx.task == "binary_classification":`.
3. Also gate subgroup calibration (brier/ece) the same way.

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_v2_runner.py -v
```

Also run full suite:

```bash
python -m pytest tests/ -v
```

**Step 5: Commit**

```bash
git add src/metrics_lie/runner.py src/metrics_lie/scenarios/base.py tests/test_v2_runner.py
git commit -m "feat: gate runner diagnostics on task type, add n_classes to ScenarioContext"
```

---

### Task 14: Gate analysis modules on task type

**Files:**
- Modify: `src/metrics_lie/analysis/disagreement.py`
- Modify: `src/metrics_lie/analysis/threshold_sweep.py`
- Create: `tests/test_v2_analysis_guards.py`

**Context:** `analyze_metric_disagreements` and `run_threshold_sweep` are binary-only. For non-binary tasks, they should return empty/no-op results. Also fix duplicated `THRESHOLD_METRICS` constant in `disagreement.py`.

**Step 1: Write the failing tests**

Create `tests/test_v2_analysis_guards.py`:

```python
"""Tests for analysis module task-type guards."""
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.model.surface import PredictionSurface, SurfaceType, CalibrationState


def test_threshold_sweep_rejects_continuous():
    from metrics_lie.analysis.threshold_sweep import run_threshold_sweep
    surface = PredictionSurface(
        surface_type=SurfaceType.CONTINUOUS,
        values=np.array([1.0, 2.0, 3.0]),
        dtype=np.dtype("float64"),
        n_samples=3,
        class_names=(),
        positive_label=None,
        threshold=None,
        calibration_state=CalibrationState.UNKNOWN,
        model_hash=None,
        is_deterministic=True,
    )
    with pytest.raises(ValueError, match="PROBABILITY|SCORE"):
        run_threshold_sweep(
            y_true=np.array([1.0, 2.0, 3.0]),
            surface=surface,
            metrics=["mae"],
        )


def test_disagreement_returns_empty_for_continuous():
    from metrics_lie.analysis.disagreement import analyze_metric_disagreements
    surface = PredictionSurface(
        surface_type=SurfaceType.CONTINUOUS,
        values=np.array([1.0, 2.0, 3.0]),
        dtype=np.dtype("float64"),
        n_samples=3,
        class_names=(),
        positive_label=None,
        threshold=None,
        calibration_state=CalibrationState.UNKNOWN,
        model_hash=None,
        is_deterministic=True,
    )
    result = analyze_metric_disagreements(
        y_true=np.array([1.0, 2.0, 3.0]),
        surface=surface,
        thresholds={},
        metrics=["mae"],
    )
    assert result == []


def test_disagreement_uses_core_threshold_metrics():
    """disagreement.py should import THRESHOLD_METRICS from core, not redefine."""
    from metrics_lie.analysis import disagreement
    from metrics_lie.metrics.core import THRESHOLD_METRICS as CORE_TM
    assert disagreement.THRESHOLD_METRICS is CORE_TM
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_v2_analysis_guards.py -v
```

**Step 3: Implement**

In `src/metrics_lie/analysis/disagreement.py`:
1. Remove the local `THRESHOLD_METRICS` definition.
2. Add `from metrics_lie.metrics.core import THRESHOLD_METRICS`.
3. At the top of `analyze_metric_disagreements`, add an early return for non-binary surfaces:

```python
    if surface.surface_type not in (SurfaceType.PROBABILITY, SurfaceType.SCORE):
        return []
```

In `src/metrics_lie/analysis/threshold_sweep.py`:
The existing code already raises `ValueError` for non-PROBABILITY/SCORE surfaces, which covers CONTINUOUS. Verify this works.

For `PredictionSurface` construction in tests: `class_names` may need to accept empty tuple for regression. Read `surface.py` to check if this needs adjustment — if `class_names` is typed as `tuple[str, str]`, change to `tuple[str, ...]`.

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_v2_analysis_guards.py -v
```

**Step 5: Commit**

```bash
git add src/metrics_lie/analysis/disagreement.py tests/test_v2_analysis_guards.py
git commit -m "fix: import THRESHOLD_METRICS from core, guard analysis for non-binary tasks"
```

---

### Task 15: Relax PredictionSurface for multiclass and regression

**Files:**
- Modify: `src/metrics_lie/model/surface.py`
- Create: `tests/test_v2_surface.py`

**Context:** `PredictionSurface.class_names` is `tuple[str, str]` (binary only). Multiclass needs variable-length. Regression may have no class names at all. The 2D probability validation enforces `shape[1] == 2` — must allow `>= 2` for multiclass.

**Step 1: Write the failing tests**

Create `tests/test_v2_surface.py`:

```python
"""Tests for multi-task PredictionSurface."""
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.model.surface import (
    CalibrationState,
    PredictionSurface,
    SurfaceType,
    validate_surface,
)


def test_surface_accepts_empty_class_names():
    """Regression surfaces should accept empty class_names."""
    surface = PredictionSurface(
        surface_type=SurfaceType.CONTINUOUS,
        values=np.array([1.0, 2.0, 3.0]),
        dtype=np.dtype("float64"),
        n_samples=3,
        class_names=(),
        positive_label=None,
        threshold=None,
        calibration_state=CalibrationState.UNKNOWN,
        model_hash=None,
        is_deterministic=True,
    )
    assert surface.class_names == ()


def test_surface_accepts_multiclass_names():
    """Multiclass surfaces should accept 3+ class names."""
    surface = PredictionSurface(
        surface_type=SurfaceType.PROBABILITY,
        values=np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]),
        dtype=np.dtype("float64"),
        n_samples=2,
        class_names=("class_0", "class_1", "class_2"),
        positive_label=None,
        threshold=None,
        calibration_state=CalibrationState.UNKNOWN,
        model_hash=None,
        is_deterministic=True,
    )
    assert len(surface.class_names) == 3


def test_validate_surface_2d_probability_multiclass():
    """2D probability array with 3 classes should validate."""
    values = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]])
    result = validate_surface(
        surface_type=SurfaceType.PROBABILITY,
        values=values,
        expected_n_samples=3,
        threshold=None,
        enforce_binary=False,
    )
    assert result.shape == (3, 3)
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_v2_surface.py -v
```

**Step 3: Implement**

In `src/metrics_lie/model/surface.py`:

1. Change `class_names: tuple[str, str]` to `class_names: tuple[str, ...]`.
2. In `validate_surface` for `PROBABILITY`: change the 2D check from `if arr.shape[1] != 2` to `if arr.shape[1] < 2`.
3. In `to_jsonable`: handle 2D `values` for statistics (compute per-class or overall).

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_v2_surface.py -v
```

Also run existing surface/validation tests:

```bash
python -m pytest tests/ -k "surface or validation" -v
```

**Step 5: Commit**

```bash
git add src/metrics_lie/model/surface.py tests/test_v2_surface.py
git commit -m "feat: relax PredictionSurface for multiclass class_names and 2D probabilities"
```

---

### Task 16: End-to-end multiclass and regression tests with existing tests passing

**Files:**
- Create: `tests/test_v2_e2e.py`

**Context:** Final validation that the full pipeline works end-to-end for multiclass and regression, including scenario Monte Carlo, metric computation, and bundle output — alongside all existing binary tests.

**Step 1: Write the tests**

Create `tests/test_v2_e2e.py`:

```python
"""End-to-end tests for Phase 2 multi-task pipeline."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression

from metrics_lie.db.session import DB_PATH, engine, init_db
from metrics_lie.execution import run_from_spec_dict
from metrics_lie.schema import ResultBundle
from metrics_lie.utils.paths import get_run_dir


@pytest.fixture(autouse=True)
def fresh_db():
    engine.dispose()
    if DB_PATH.exists():
        DB_PATH.unlink()
    init_db()


def test_e2e_multiclass_with_scenarios(tmp_path: Path):
    """Full multiclass pipeline with label_noise and score_noise scenarios."""
    rng = np.random.default_rng(42)
    n = 90
    X = rng.standard_normal((n, 3))
    y = np.array([0]*30 + [1]*30 + [2]*30)
    model = LogisticRegression(random_state=0, max_iter=300).fit(X, y)
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    df = pd.DataFrame({
        "f1": X[:, 0], "f2": X[:, 1], "f3": X[:, 2],
        "y_true": y, "y_score": np.zeros(n),
        "group": (["A"]*15 + ["B"]*15) * 3,
    })
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    spec = {
        "name": "e2e_multiclass",
        "task": "multiclass_classification",
        "dataset": {
            "source": "csv", "path": str(csv_path),
            "y_true_col": "y_true", "y_score_col": "y_score",
            "feature_cols": ["f1", "f2", "f3"], "subgroup_col": "group",
        },
        "metric": "macro_f1",
        "model_source": {"kind": "pickle", "path": str(model_path)},
        "scenarios": [
            {"id": "label_noise", "params": {"p": 0.1}},
            {"id": "score_noise", "params": {"sigma": 0.05}},
            {"id": "class_imbalance", "params": {"target_pos_rate": 0.15}},
        ],
        "n_trials": 5,
        "seed": 42,
    }

    run_id = run_from_spec_dict(spec)
    bundle = ResultBundle.model_validate_json(
        get_run_dir(run_id).results_json.read_text(encoding="utf-8")
    )
    assert bundle.task_type == "multiclass_classification"
    assert len(bundle.scenarios) > 0
    assert bundle.baseline is not None
    assert bundle.baseline.mean > 0.0


def test_e2e_regression_with_scenarios(tmp_path: Path):
    """Full regression pipeline with label_noise and score_noise scenarios."""
    rng = np.random.default_rng(42)
    n = 50
    X = rng.standard_normal((n, 2))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(0, 0.5, n)
    model = LinearRegression().fit(X, y)
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    df = pd.DataFrame({
        "f1": X[:, 0], "f2": X[:, 1],
        "y_true": y, "y_score": np.zeros(n),
    })
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    spec = {
        "name": "e2e_regression",
        "task": "regression",
        "dataset": {
            "source": "csv", "path": str(csv_path),
            "y_true_col": "y_true", "y_score_col": "y_score",
            "feature_cols": ["f1", "f2"],
        },
        "metric": "mae",
        "model_source": {"kind": "pickle", "path": str(model_path)},
        "scenarios": [
            {"id": "label_noise", "params": {"p": 0.1}},
            {"id": "score_noise", "params": {"sigma": 0.5}},
        ],
        "n_trials": 5,
        "seed": 42,
    }

    run_id = run_from_spec_dict(spec)
    bundle = ResultBundle.model_validate_json(
        get_run_dir(run_id).results_json.read_text(encoding="utf-8")
    )
    assert bundle.task_type == "regression"
    assert "mae" in bundle.applicable_metrics
    assert "r2" in bundle.applicable_metrics
    assert len(bundle.scenarios) > 0
    # No binary analysis artifacts
    assert "threshold_sweep" not in bundle.analysis_artifacts
    assert "metric_disagreements" not in bundle.analysis_artifacts
```

**Step 2: Run tests**

```bash
python -m pytest tests/test_v2_e2e.py -v
```

These should pass after Tasks 12-15 are complete.

**Step 3: Run full test suite**

```bash
python -m pytest tests/ -v
python -m ruff check src tests
```

All tests must pass. All lint must pass.

**Step 4: Commit**

```bash
git add tests/test_v2_e2e.py
git commit -m "test: add end-to-end multiclass and regression pipeline tests"
```

---

## Summary

| Task | Component | New Tests | Files Modified |
|------|-----------|-----------|----------------|
| 1 | TaskType properties | 3 | `task_types.py` |
| 2 | Multiclass metrics | 13 | `metrics/core.py` |
| 3 | Regression metrics | 11 | `metrics/core.py` |
| 4 | MetricRequirement + registry | 6 | `metrics/registry.py` |
| 5 | MetricResolver filtering | 6 | `metrics/applicability.py` |
| 6 | label_noise generalization | 3 | `scenarios/label_noise.py` |
| 7 | score_noise generalization | 3 | `scenarios/score_noise.py` |
| 8 | class_imbalance generalization | 3 | `scenarios/class_imbalance.py` |
| 9 | SCENARIO_TASK_COMPAT | 8 | `surface_compat.py` |
| 10 | compute_metric validation | 3 | `metrics/core.py` (test only) |
| 11 | ResultBundle schema | 3 | `schema.py` |
| 12 | Execution pipeline wiring | 2 | `execution.py` |
| 13 | Runner diagnostics gating | 3 | `runner.py`, `scenarios/base.py` |
| 14 | Analysis guards | 3 | `analysis/disagreement.py` |
| 15 | PredictionSurface relaxation | 3 | `model/surface.py` |
| 16 | E2E tests | 2 | test only |

**Total: 16 tasks, ~75 new tests, 12 files modified**

**Parallelism opportunities:**
- Tasks 1, 2, 3 can run in parallel (no shared files after metrics/core.py — but 2 and 3 both modify core.py, so they should be sequential or carefully merged)
- Tasks 6, 7, 8, 9 can run in parallel (different scenario files)
- Tasks 11, 14, 15 can run in parallel (different files)
- Tasks 12, 13 must be sequential (both modify pipeline)
- Task 16 depends on all previous tasks
