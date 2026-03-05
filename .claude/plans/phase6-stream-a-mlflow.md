# Plan: Phase 6 Stream A — MLflow Integration

> **Execution model:** This plan was written by Opus for execution by Sonnet.
> Run: `claude --model claude-sonnet-4-6` then say "Execute plan .claude/plans/phase6-stream-a-mlflow.md"

## Goal

Add MLflow integration: `log_to_mlflow(result)` for logging Spectra results to MLflow, and an MLflow model adapter for loading models from the MLflow registry.

## Context

- The adapter registry pattern is in `src/metrics_lie/model/adapter_registry.py` (format-based + extension-based lookup)
- Default registry in `src/metrics_lie/model/default_registry.py` uses lazy factory functions with optional dependency guards
- `ModelAdapterProtocol` in `src/metrics_lie/model/protocol.py` defines: `task_type`, `metadata`, `predict()`, `predict_proba()`, `predict_raw()`, `get_all_surfaces()`
- `ResultBundle` in `src/metrics_lie/schema.py` has: `metric_name`, `baseline` (MetricSummary), `scenarios`, `notes`, `analysis_artifacts`, `metric_results`, `applicable_metrics`
- Public SDK in `src/metrics_lie/__init__.py` exports `evaluate`, `compare`, `score`, etc.
- `pyproject.toml` has optional dependency groups; `mlflow` group does not yet exist

## Prerequisites

- [ ] Read all files listed in each task before modifying them

## Tasks

### Task A1: Create MLflow logging module

**Files:**
- Create: `src/metrics_lie/integrations/__init__.py`
- Create: `src/metrics_lie/integrations/mlflow.py`

**Steps:**

1. Create the integrations package init:

```python
"""Optional integrations with external ML platforms."""
```

2. Create `mlflow.py`:

```python
"""MLflow integration — log Spectra results to MLflow tracking."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def log_to_mlflow(
    result: Any,
    *,
    run_id: str | None = None,
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
) -> str:
    """Log a Spectra ResultBundle to MLflow.

    Logs metrics, scenario results (as JSON artifact), experiment params,
    and any matplotlib plot artifacts.

    Args:
        result: A ResultBundle instance.
        run_id: Existing MLflow run ID to log to. If None, creates a new run.
        experiment_name: MLflow experiment name (used only when creating a new run).
        tracking_uri: MLflow tracking server URI. If None, uses MLflow default.

    Returns:
        The MLflow run ID that was logged to.
    """
    try:
        import mlflow
    except ImportError as e:
        raise ImportError(
            "MLflow integration requires: pip install metrics_lie[mlflow]"
        ) from e

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if experiment_name and run_id is None:
        mlflow.set_experiment(experiment_name)

    def _do_log(active_run: Any) -> str:
        rid = active_run.info.run_id

        # Log params
        mlflow.log_param("spectra.experiment_name", result.experiment_name)
        mlflow.log_param("spectra.metric_name", result.metric_name)
        mlflow.log_param("spectra.task_type", getattr(result, "task_type", "binary_classification"))
        mlflow.log_param("spectra.run_id", result.run_id)

        # Log baseline metric
        if result.baseline:
            mlflow.log_metric(f"spectra.{result.metric_name}.baseline_mean", result.baseline.mean)
            mlflow.log_metric(f"spectra.{result.metric_name}.baseline_std", result.baseline.std)
            mlflow.log_metric(f"spectra.{result.metric_name}.baseline_q05", result.baseline.q05)
            mlflow.log_metric(f"spectra.{result.metric_name}.baseline_q95", result.baseline.q95)

        # Log per-metric results
        for metric_name, summary in result.metric_results.items():
            mlflow.log_metric(f"spectra.{metric_name}.mean", summary.mean)
            mlflow.log_metric(f"spectra.{metric_name}.std", summary.std)

        # Log scenario deltas as metrics
        for scenario in result.scenarios:
            delta = scenario.metric.mean - (result.baseline.mean if result.baseline else 0.0)
            safe_id = scenario.scenario_id.replace(" ", "_")
            mlflow.log_metric(f"spectra.scenario.{safe_id}.mean", scenario.metric.mean)
            mlflow.log_metric(f"spectra.scenario.{safe_id}.delta", delta)

        # Log full ResultBundle as JSON artifact
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="spectra_result_"
        ) as f:
            f.write(result.to_pretty_json())
            temp_path = f.name
        try:
            mlflow.log_artifact(temp_path, artifact_path="spectra")
        finally:
            Path(temp_path).unlink(missing_ok=True)

        logger.info("Logged Spectra result to MLflow run %s", rid)
        return rid

    if run_id:
        with mlflow.start_run(run_id=run_id):
            return _do_log(mlflow.active_run())
    else:
        with mlflow.start_run() as active_run:
            return _do_log(active_run)
```

3. Run: `cd /c/GitHubProjects/when-metrics-lie && python -c "from metrics_lie.integrations.mlflow import log_to_mlflow; print('OK')"`
   Expected: OK (import succeeds; mlflow not needed until called)

**Acceptance:** Module imports without requiring mlflow at import time. `log_to_mlflow` raises ImportError with install hint if mlflow is missing.

### Task A2: Create MLflow model adapter

**Files:**
- Create: `src/metrics_lie/model/adapters/mlflow_adapter.py`

**Steps:**

1. Create the adapter:

```python
"""MLflow model adapter — load models from MLflow registry."""
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


class MLflowAdapter:
    """Adapter for models loaded from MLflow model registry.

    Supports MLflow model URIs:
      - runs:/run_id/model
      - models:/model_name/version
      - models:/model_name/stage
    """

    def __init__(
        self,
        *,
        uri: str,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        threshold: float = 0.5,
        positive_label: int = 1,
    ) -> None:
        try:
            import mlflow.pyfunc
        except ImportError as e:
            raise ImportError(
                "MLflow adapter requires: pip install metrics_lie[mlflow]"
            ) from e

        self._uri = uri
        self._task_type = task_type
        self._threshold = threshold
        self._positive_label = positive_label
        self._model = mlflow.pyfunc.load_model(uri)
        self._model_info = self._model.metadata

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    @property
    def metadata(self) -> ModelMetadata:
        model_class = "mlflow.pyfunc"
        if self._model_info and hasattr(self._model_info, "flavors"):
            flavors = self._model_info.flavors or {}
            if "sklearn" in flavors:
                model_class = "mlflow.sklearn"
            elif "xgboost" in flavors:
                model_class = "mlflow.xgboost"
            elif "lightgbm" in flavors:
                model_class = "mlflow.lightgbm"
        return ModelMetadata(
            model_class=model_class,
            model_module="mlflow",
            model_format="mlflow",
            model_hash=None,
            capabilities={"predict", "predict_proba"},
        )

    def predict(self, X: np.ndarray) -> PredictionSurface:
        import pandas as pd

        df = pd.DataFrame(X)
        raw = self._model.predict(df)
        preds = np.asarray(raw)

        if self._task_type.is_regression:
            arr = validate_surface(
                surface_type=SurfaceType.CONTINUOUS,
                values=preds.flatten(),
                expected_n_samples=X.shape[0],
                threshold=None,
            )
            return PredictionSurface(
                surface_type=SurfaceType.CONTINUOUS,
                values=arr.astype(float),
                dtype=arr.dtype,
                n_samples=int(arr.shape[0]),
                class_names=None,
                positive_label=None,
                threshold=None,
                calibration_state=CalibrationState.UNKNOWN,
                model_hash=None,
                is_deterministic=True,
            )

        # Classification: predict returns labels
        if preds.ndim == 2:
            labels = np.argmax(preds, axis=1)
        else:
            labels = preds.astype(int)

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
            model_hash=None,
            is_deterministic=True,
        )

    def predict_proba(self, X: np.ndarray) -> PredictionSurface | None:
        """Attempt probabilistic prediction via MLflow model.

        MLflow pyfunc.predict() may return probabilities if the underlying
        model supports them. We detect this from output shape.
        """
        import pandas as pd

        df = pd.DataFrame(X)
        raw = self._model.predict(df)
        preds = np.asarray(raw)

        # If output is 2D with >1 columns, treat as class probabilities
        if preds.ndim == 2 and preds.shape[1] > 1:
            if self._task_type == TaskType.BINARY_CLASSIFICATION:
                proba = preds[:, 1]
            else:
                # Multiclass: return full probability array
                proba = preds
        elif preds.ndim == 1:
            vals = preds.astype(float)
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
            model_hash=None,
            is_deterministic=True,
        )

    def predict_raw(self, X: np.ndarray) -> dict[str, Any]:
        import pandas as pd

        df = pd.DataFrame(X)
        raw = self._model.predict(df)
        return {"predictions": np.asarray(raw).tolist()}

    def get_all_surfaces(self, X: np.ndarray) -> dict[SurfaceType, PredictionSurface]:
        surfaces: dict[SurfaceType, PredictionSurface] = {}
        label_surface = self.predict(X)
        surfaces[label_surface.surface_type] = label_surface
        proba = self.predict_proba(X)
        if proba is not None:
            surfaces[SurfaceType.PROBABILITY] = proba
        return surfaces
```

2. Run: `cd /c/GitHubProjects/when-metrics-lie && python -c "from metrics_lie.model.adapters.mlflow_adapter import MLflowAdapter; print('OK')"`
   Expected: ImportError mentioning `pip install metrics_lie[mlflow]` (mlflow not installed — that's correct)

**Acceptance:** Adapter class loads. Import fails cleanly when mlflow is missing.

### Task A3: Register MLflow adapter in default registry

**Files:**
- Modify: `src/metrics_lie/model/default_registry.py`

**Steps:**

1. Add the MLflow factory function after `_http_factory` (around line 23):

```python
def _mlflow_factory(**kwargs):
    from metrics_lie.model.adapters.mlflow_adapter import MLflowAdapter
    return MLflowAdapter(**kwargs)
```

2. Add MLflow registration in `get_default_registry()`, after the CatBoost block (around line 74):

```python
    # MLflow (optional)
    try:
        import mlflow  # noqa: F401
        reg.register("mlflow", factory=_mlflow_factory, extensions=set())
    except ImportError:
        pass
```

3. Run: `cd /c/GitHubProjects/when-metrics-lie && python -c "from metrics_lie.model.default_registry import get_default_registry; r = get_default_registry(); print(r.list_formats())"`
   Expected: List of formats (mlflow appears only if installed)

**Acceptance:** Registry loads without error. MLflow format registered when mlflow package is available.

### Task A4: Add mlflow dependency group to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

**Steps:**

1. Add `mlflow` optional dependency group after the `drift` group (line 48):

```toml
mlflow = [
  "mlflow>=2.10"
]
```

2. Update the `all` group to include mlflow:

Replace:
```toml
all = [
  "metrics_lie[dev,web,onnx,boosting,fairness,drift]"
]
```
With:
```toml
all = [
  "metrics_lie[dev,web,onnx,boosting,fairness,drift,mlflow]"
]
```

3. Run: `cd /c/GitHubProjects/when-metrics-lie && python -m pip install -e ".[dev]" 2>&1 | tail -3`
   Expected: Successfully installed (no pyproject.toml parse errors)

**Acceptance:** `pip install -e ".[mlflow]"` would install mlflow>=2.10. Core install unchanged.

### Task A5: Export log_to_mlflow from public SDK

**Files:**
- Modify: `src/metrics_lie/__init__.py`

**Steps:**

1. Add import after the existing imports (after line 11):

```python
from metrics_lie.integrations.mlflow import log_to_mlflow
```

2. Add `"log_to_mlflow"` to the `__all__` list.

3. Run: `cd /c/GitHubProjects/when-metrics-lie && python -c "from metrics_lie import log_to_mlflow; print('OK')"`
   Expected: OK

**Acceptance:** `import metrics_lie; metrics_lie.log_to_mlflow` is accessible. Import of main package still works without mlflow installed.

**WAIT** — this will fail if mlflow is not installed because the import happens at module load time. Fix: use lazy import.

Replace the direct import with a lazy accessor. Instead of modifying `__init__.py` imports, add to `sdk.py`:

1. Add to `src/metrics_lie/sdk.py` at the end:

```python
def log_to_mlflow(
    result: Any,
    *,
    run_id: str | None = None,
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
) -> str:
    """Log a Spectra ResultBundle to MLflow. Requires: pip install metrics_lie[mlflow]"""
    from metrics_lie.integrations.mlflow import log_to_mlflow as _log
    return _log(result, run_id=run_id, experiment_name=experiment_name, tracking_uri=tracking_uri)
```

2. In `__init__.py`, add `log_to_mlflow` to the import from sdk (line 6):

Replace:
```python
from metrics_lie.sdk import compare, evaluate, evaluate_file, score
```
With:
```python
from metrics_lie.sdk import compare, evaluate, evaluate_file, log_to_mlflow, score
```

3. Add `"log_to_mlflow"` to `__all__`.

4. Run: `cd /c/GitHubProjects/when-metrics-lie && python -c "from metrics_lie import log_to_mlflow; print(type(log_to_mlflow))"`
   Expected: `<class 'function'>` (lazy — mlflow not imported until called)

**Acceptance:** `spectra.log_to_mlflow(result)` is in the public API. No ImportError at import time.

### Task A6: Lint and verify

**Steps:**

1. Run: `cd /c/GitHubProjects/when-metrics-lie && python -m ruff check src/metrics_lie/integrations/ src/metrics_lie/model/adapters/mlflow_adapter.py src/metrics_lie/model/default_registry.py src/metrics_lie/__init__.py src/metrics_lie/sdk.py --fix`
2. Run: `cd /c/GitHubProjects/when-metrics-lie && python -c "from metrics_lie import log_to_mlflow, evaluate, compare; print('All imports OK')"`
3. Run: `cd /c/GitHubProjects/when-metrics-lie && python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5`
   Expected: All existing tests pass, no regressions.

**Acceptance:** Lint clean, all tests pass, imports work.

## Boundaries

**DO:**
- Follow steps exactly
- Use lazy imports so mlflow is never required at package import time
- Run verification commands

**DO NOT:**
- Modify frontend files
- Modify existing adapter files (only add new ones + registry)
- Install mlflow in the environment (tests should pass without it)
- Create a git commit (parent agent handles this)

## Escalation Triggers

Stop and flag for Opus review if:
- Importing `metrics_lie` fails after changes
- Existing tests break
- The `ModelAdapterProtocol` doesn't match what's documented here

When escalating, write to `.claude/plans/phase6-stream-a-blockers.md`.

## Verification

After all tasks complete:
- [ ] `python -c "from metrics_lie import log_to_mlflow; print('OK')"` works
- [ ] `python -m ruff check src/` passes
- [ ] `python -m pytest tests/ -x -q` passes (all existing tests)
- [ ] No files modified outside the plan's file list
