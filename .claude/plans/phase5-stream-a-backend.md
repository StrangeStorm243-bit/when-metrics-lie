# Plan: Phase 5 Stream A — Backend (Multi-Task Web API)

> **Execution model:** This plan was written by Opus for execution by Sonnet.
> Run: `claude --model claude-sonnet-4-6` then say "Execute plan .claude/plans/phase5-stream-a-backend.md"

## Goal

Make the FastAPI backend support all task types (binary, multiclass, regression, ranking) and model formats (sklearn, ONNX, boosting). After this stream, the API accepts a `task_type` field, validates models per task, returns task-specific diagnostics, and filters presets by task type.

## Context

- Core engine (Phases 1-3) already supports all task types end-to-end
- `ResultBundle` already has a `task_type` field (schema.py:63)
- `execution.py` already routes by task_type for dataset loading, metrics, scenarios, analysis
- Binary classification is hardcoded in 5 backend files — this plan fixes all of them
- The frontend (Stream B) depends on the contract changes in Task A1 but NOT on the implementation

## Prerequisites

- [ ] Create branch: `git checkout -b phase5-stream-a-backend`
- [ ] Read files listed in each task before modifying them

## Tasks

### Task A1: Add task_type to API contracts

**Files:**
- Modify: `web/backend/app/contracts.py`

**Steps:**

1. Add `task_type` field to `ExperimentCreateRequest`:

```python
# After the 'config' field (line 25), this is already passed via config.
# No change needed to ExperimentCreateRequest — task_type flows through config dict.
```

2. Add `task_type` field to `ExperimentSummary` (after `stress_suite_id`, line 34):

```python
    task_type: str = Field(
        "binary_classification",
        description="Task type: binary_classification, multiclass_classification, regression, ranking",
    )
```

3. Add task-specific fields to `ResultSummary` (after `dashboard_summary`, line 127):

```python
    task_type: str = Field(
        "binary_classification",
        description="Task type for this result",
    )
    confusion_matrix: Optional[list[list[int]]] = Field(
        None, description="Confusion matrix [n_classes x n_classes] (classification only)"
    )
    class_names: Optional[list[str]] = Field(
        None, description="Class label names (classification only)"
    )
    per_class_metrics: Optional[dict[str, dict[str, float]]] = Field(
        None, description="Per-class precision/recall/f1 (multiclass only)"
    )
    residual_stats: Optional[dict[str, float]] = Field(
        None, description="Residual statistics: mean, std, min, max, median (regression only)"
    )
    ranking_metrics: Optional[dict[str, float]] = Field(
        None, description="Ranking metrics: ndcg, mrr, map (ranking only)"
    )
```

4. Add `task_type` to `ModelUploadResponse` (after `model_class`, line 153):

```python
    task_type: str = Field(
        "binary_classification",
        description="Validated task type for this model",
    )
    n_classes: Optional[int] = Field(
        None, description="Number of output classes (classification only)"
    )
```

5. Add `SupportedFormat` response model (at end of file):

```python
class SupportedFormat(BaseModel):
    """A supported model format."""
    format_id: str = Field(..., description="Format identifier")
    name: str = Field(..., description="Human-readable name")
    extensions: list[str] = Field(..., description="File extensions")
    task_types: list[str] = Field(..., description="Supported task types")
```

6. Run: `cd web/backend && python -c "from app.contracts import *; print('OK')"`
   Expected: OK (no import errors)

**Acceptance:** All new fields have defaults so existing API calls still work.

### Task A2: Add task-specific artifacts to core execution

**Files:**
- Modify: `src/metrics_lie/execution.py` (lines 578-588, inside the notes/analysis block)

**Steps:**

1. After the `notes = { ... }` block (line 588) and before the `analysis_artifacts` dict (line 590), add confusion matrix and per-class metrics computation:

```python
        # Task-specific summary artifacts for web display
        task_specific: dict[str, Any] = {}

        if task_type.is_classification and prediction_surface is not None:
            from sklearn.metrics import confusion_matrix as _confusion_matrix

            if prediction_surface.surface_type == SurfaceType.LABEL:
                y_pred_labels = prediction_surface.values.astype(int)
            elif prediction_surface.surface_type == SurfaceType.PROBABILITY:
                if prediction_surface.values.ndim == 2:
                    y_pred_labels = np.argmax(prediction_surface.values, axis=1)
                else:
                    threshold = prediction_surface.threshold or 0.5
                    y_pred_labels = (prediction_surface.values >= threshold).astype(int)
            else:
                y_pred_labels = None

            if y_pred_labels is not None:
                cm = _confusion_matrix(y_true, y_pred_labels)
                task_specific["confusion_matrix"] = cm.tolist()

                unique_classes = sorted(set(y_true.tolist()) | set(y_pred_labels.tolist()))
                task_specific["class_names"] = [str(c) for c in unique_classes]

                if not task_type.is_binary:
                    from sklearn.metrics import precision_recall_fscore_support
                    prec, rec, f1s, sup = precision_recall_fscore_support(
                        y_true, y_pred_labels, labels=unique_classes, zero_division=0
                    )
                    per_class = {}
                    for i, cls in enumerate(unique_classes):
                        per_class[str(cls)] = {
                            "precision": float(prec[i]),
                            "recall": float(rec[i]),
                            "f1": float(f1s[i]),
                            "support": int(sup[i]),
                        }
                    task_specific["per_class_metrics"] = per_class

        elif task_type.is_regression and prediction_surface is not None:
            residuals = y_true - prediction_surface.values
            task_specific["residual_stats"] = {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals)),
                "median": float(np.median(residuals)),
                "mae": float(np.mean(np.abs(residuals))),
                "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            }
```

2. Add `task_specific` to the `notes` dict (modify the notes dict around line 578):

```python
        notes = {
            "phase": "1.7B",
            "spec_path": spec_path_for_notes,
            "baseline_diagnostics": baseline_cal,
            "applicable_metrics": applicable.metrics,
            "metric_resolution": {
                "excluded": applicable.excluded,
                "warnings": applicable.warnings,
                "reasoning_trace": applicable.reasoning_trace,
            },
            "task_specific": task_specific,
        }
```

3. Run: `pytest tests/ -x -q --tb=short 2>&1 | tail -5`
   Expected: All existing tests still pass.

**Acceptance:** `notes["task_specific"]` contains confusion_matrix for classification runs, residual_stats for regression runs.

### Task A3: Generalize model validation

**Files:**
- Modify: `web/backend/app/model_validation.py`

**Steps:**

1. Replace the entire file content with task-type-aware validation:

```python
"""Validate uploaded models for web API.

Supports binary, multiclass, and regression models across pickle, ONNX, and boosting formats.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np


ACCEPTED_EXTENSIONS = {
    ".pkl": "pickle",
    ".joblib": "pickle",
    ".onnx": "onnx",
    ".ubj": "xgboost",
    ".xgb": "xgboost",
    ".lgb": "lightgbm",
    ".cbm": "catboost",
}


@dataclass
class ModelValidationResult:
    """Result of validating an uploaded model."""

    valid: bool
    model_class: str = ""
    task_type: str = "binary_classification"
    n_classes: int | None = None
    capabilities: dict[str, bool] = field(
        default_factory=lambda: {
            "predict": False,
            "predict_proba": False,
            "decision_function": False,
        }
    )
    error: str | None = None


def validate_model(
    file_bytes: bytes,
    file_ext: str,
    task_type: str = "binary_classification",
) -> ModelValidationResult:
    """
    Validate an uploaded model file.

    Args:
        file_bytes: Raw file bytes
        file_ext: File extension (e.g., '.pkl', '.onnx')
        task_type: Target task type

    Returns:
        ModelValidationResult with validation outcome
    """
    fmt = ACCEPTED_EXTENSIONS.get(file_ext.lower())
    if fmt is None:
        return ModelValidationResult(
            valid=False,
            error=f"Unsupported file extension: {file_ext}. "
            f"Accepted: {', '.join(sorted(ACCEPTED_EXTENSIONS.keys()))}",
        )

    if fmt == "pickle":
        return _validate_pickle(file_bytes, task_type)
    elif fmt == "onnx":
        return _validate_onnx(file_bytes, task_type)
    else:
        # Boosting formats — accept if file is non-empty
        # Full validation happens at inference time via BoostingAdapter
        return ModelValidationResult(
            valid=True,
            model_class=f"{fmt}_model",
            task_type=task_type,
            capabilities={"predict": True, "predict_proba": fmt != "catboost"},
        )


def _validate_pickle(file_bytes: bytes, task_type: str) -> ModelValidationResult:
    """Validate a pickle/joblib model."""
    try:
        model = pickle.loads(file_bytes)
    except Exception as e:
        return ModelValidationResult(valid=False, error=f"Invalid pickle file: {e!s}")

    model_class = type(model).__name__
    module = type(model).__module__
    if module:
        model_class = f"{module}.{model_class}"

    capabilities = {
        "predict": hasattr(model, "predict") and callable(getattr(model, "predict")),
        "predict_proba": hasattr(model, "predict_proba")
        and callable(getattr(model, "predict_proba")),
        "decision_function": hasattr(model, "decision_function")
        and callable(getattr(model, "decision_function")),
    }

    if not capabilities["predict"]:
        return ModelValidationResult(
            valid=False,
            model_class=model_class,
            capabilities=capabilities,
            error="Model must support predict()",
        )

    n_features = getattr(model, "n_features_in_", 1)
    n_classes = None

    if task_type in ("binary_classification", "multiclass_classification"):
        if not capabilities["predict_proba"]:
            return ModelValidationResult(
                valid=False,
                model_class=model_class,
                capabilities=capabilities,
                error="Classification models must support predict_proba()",
            )
        try:
            X = np.zeros((2, n_features), dtype=np.float64)
            proba = np.asarray(model.predict_proba(X))
        except Exception as e:
            return ModelValidationResult(
                valid=False,
                model_class=model_class,
                capabilities=capabilities,
                error=f"predict_proba failed: {e!s}",
            )

        if proba.ndim != 2:
            return ModelValidationResult(
                valid=False,
                model_class=model_class,
                capabilities=capabilities,
                error="predict_proba output must be 2D (n_samples, n_classes)",
            )

        n_classes = proba.shape[1]
        if task_type == "binary_classification" and n_classes != 2:
            return ModelValidationResult(
                valid=False,
                model_class=model_class,
                capabilities=capabilities,
                error=f"Binary classification requires 2 classes, got {n_classes}",
            )
        if task_type == "multiclass_classification" and n_classes < 3:
            return ModelValidationResult(
                valid=False,
                model_class=model_class,
                capabilities=capabilities,
                error=f"Multiclass classification requires 3+ classes, got {n_classes}",
            )

    elif task_type == "regression":
        try:
            X = np.zeros((2, n_features), dtype=np.float64)
            preds = np.asarray(model.predict(X))
        except Exception as e:
            return ModelValidationResult(
                valid=False,
                model_class=model_class,
                capabilities=capabilities,
                error=f"predict failed: {e!s}",
            )
        if preds.ndim != 1:
            return ModelValidationResult(
                valid=False,
                model_class=model_class,
                capabilities=capabilities,
                error=f"Regression predict() must return 1D array, got shape {preds.shape}",
            )

    return ModelValidationResult(
        valid=True,
        model_class=model_class,
        task_type=task_type,
        n_classes=n_classes,
        capabilities=capabilities,
    )


def _validate_onnx(file_bytes: bytes, task_type: str) -> ModelValidationResult:
    """Validate an ONNX model file."""
    try:
        import onnx
        model = onnx.load_from_string(file_bytes)
        onnx.checker.check_model(model)
    except ImportError:
        return ModelValidationResult(
            valid=False,
            error="ONNX validation requires: pip install onnx",
        )
    except Exception as e:
        return ModelValidationResult(
            valid=False,
            error=f"Invalid ONNX model: {e!s}",
        )

    model_class = "onnx_model"
    graph = model.graph
    if graph and graph.name:
        model_class = f"onnx.{graph.name}"

    return ModelValidationResult(
        valid=True,
        model_class=model_class,
        task_type=task_type,
        capabilities={"predict": True, "predict_proba": True},
    )


# Backward compatibility alias
def validate_sklearn_pickle(file_bytes: bytes) -> ModelValidationResult:
    """Legacy validator — delegates to validate_model with binary_classification."""
    return _validate_pickle(file_bytes, "binary_classification")
```

2. Run: `cd web/backend && python -c "from app.model_validation import validate_model, validate_sklearn_pickle; print('OK')"`
   Expected: OK

**Acceptance:** `validate_model(bytes, '.pkl', 'multiclass_classification')` validates for n_classes >= 3. `validate_model(bytes, '.pkl', 'regression')` only requires `predict()`. Old `validate_sklearn_pickle` still works.

### Task A4: Update engine bridge for multi-task

**Files:**
- Modify: `web/backend/app/engine_bridge.py`

**Steps:**

1. Replace `_get_default_scenarios` function (lines 50-61) with task-aware version:

```python
def _get_default_scenarios(stress_suite_id: str, task_type: str = "binary_classification") -> list[dict]:
    """Map stress_suite_id to default scenario configurations, filtered by task type."""
    # Base scenarios for all classification
    classification_scenarios = [
        {"id": "label_noise", "params": {"p": 0.1}},
        {"id": "score_noise", "params": {"sigma": 0.05}},
        {
            "id": "class_imbalance",
            "params": {"target_pos_rate": 0.2, "max_remove_frac": 0.8},
        },
    ]

    if task_type == "regression":
        return [
            {"id": "label_noise", "params": {"p": 0.1}},
            {"id": "score_noise", "params": {"sigma": 0.05}},
        ]
    elif task_type == "ranking":
        return [
            {"id": "label_noise", "params": {"p": 0.1}},
            {"id": "score_noise", "params": {"sigma": 0.05}},
        ]
    else:
        return classification_scenarios
```

2. Replace `_get_default_dataset` function (lines 152-169) with task-aware version:

```python
def _get_default_dataset(create_req: ExperimentCreateRequest, task_type: str = "binary_classification") -> dict:
    """Get default dataset configuration based on task type."""
    dataset_path = _get_dataset_path(create_req)
    path_str = str(dataset_path.resolve())

    dataset_dict: dict = {
        "source": "csv",
        "path": path_str,
        "y_true_col": create_req.config.get("y_true_col", "y_true"),
        "y_score_col": create_req.config.get("y_score_col", "y_score"),
    }

    subgroup_col = create_req.config.get("subgroup_col", "group")
    if subgroup_col:
        dataset_dict["subgroup_col"] = subgroup_col

    feature_cols = create_req.config.get("feature_cols")
    if isinstance(feature_cols, list) and feature_cols:
        dataset_dict["feature_cols"] = feature_cols

    return dataset_dict
```

3. In `run_experiment` function (line 233), replace hardcoded task with config value:

Replace:
```python
        "task": "binary_classification",
```
With:
```python
        "task": create_req.config.get("task_type", "binary_classification"),
```

4. Update the scenarios call (line 229):

Replace:
```python
    scenarios = _get_default_scenarios(create_req.stress_suite_id)
```
With:
```python
    task_type = create_req.config.get("task_type", "binary_classification")
    scenarios = _get_default_scenarios(create_req.stress_suite_id, task_type)
```

5. Update the dataset call (line 228):

Replace:
```python
    dataset_dict = _get_default_dataset(create_req)
```
With:
```python
    dataset_dict = _get_default_dataset(create_req, task_type)
```

6. In the model_id resolution block (around line 268), make the model_source kind dynamic based on file extension:

After `resolved_path, temp_path_to_clean = _resolve_model_path(model_id, owner_id)` add:

```python
        # Detect model format from stored metadata
        model_kind = "pickle"  # default
        import json as _json
        meta_path = Path(resolved_path).with_suffix(".meta.json")
        if meta_path.exists():
            meta = _json.loads(meta_path.read_text())
            ext = Path(meta.get("original_filename", "")).suffix.lower()
            ext_to_kind = {".onnx": "onnx", ".ubj": "xgboost", ".xgb": "xgboost", ".lgb": "lightgbm", ".cbm": "catboost"}
            model_kind = ext_to_kind.get(ext, "pickle")
```

Then replace `"kind": "pickle"` in the model_source dict with `"kind": model_kind`.

7. Run: `cd web/backend && python -c "from app.engine_bridge import run_experiment; print('OK')"`
   Expected: OK

**Acceptance:** `config.task_type` flows through to core engine. Regression experiments don't get `class_imbalance` scenario.

### Task A5: Update bundle transform for multi-task

**Files:**
- Modify: `web/backend/app/bundle_transform.py`

**Steps:**

1. Replace the entire file with task-aware transform:

```python
"""Transform engine ResultBundle to web API ResultSummary."""
from __future__ import annotations

from datetime import datetime

from metrics_lie.schema import ResultBundle

from .contracts import (
    ComponentScore,
    FindingFlag,
    ResultSummary,
    ScenarioResult as ContractScenarioResult,
)


def bundle_to_result_summary(
    bundle: ResultBundle, experiment_id: str, run_id: str
) -> ResultSummary:
    """Convert a core engine ResultBundle to the web API ResultSummary contract."""
    task_type = getattr(bundle, "task_type", "binary_classification")
    headline_score = bundle.baseline.mean if bundle.baseline else 0.0

    # Convert scenario results
    scenario_results = []
    for sr in bundle.scenarios:
        delta = sr.metric.mean - headline_score if bundle.baseline else 0.0
        scenario_results.append(
            ContractScenarioResult(
                scenario_id=sr.scenario_id,
                scenario_name=sr.scenario_id.replace("_", " ").title(),
                delta=delta,
                score=sr.metric.mean,
                severity=_classify_severity(delta, task_type),
                notes=None,
            )
        )

    # Extract component scores (task-aware)
    component_scores = _extract_component_scores(bundle, task_type)

    # Extract flags (task-aware)
    flags = _extract_flags(bundle, task_type)

    # Extract task-specific fields from notes
    task_specific = bundle.notes.get("task_specific", {})

    # Dashboard summary from analysis artifacts
    dashboard_summary = None
    if bundle.analysis_artifacts and "dashboard_summary" in bundle.analysis_artifacts:
        dashboard_summary = bundle.analysis_artifacts["dashboard_summary"]

    return ResultSummary(
        experiment_id=experiment_id,
        run_id=run_id,
        headline_score=headline_score,
        weighted_score=None,
        component_scores=component_scores,
        scenario_results=scenario_results,
        flags=flags,
        prediction_surface=bundle.prediction_surface,
        applicable_metrics=bundle.applicable_metrics,
        analysis_artifacts=bundle.analysis_artifacts,
        dashboard_summary=dashboard_summary,
        task_type=task_type,
        confusion_matrix=task_specific.get("confusion_matrix"),
        class_names=task_specific.get("class_names"),
        per_class_metrics=task_specific.get("per_class_metrics"),
        residual_stats=task_specific.get("residual_stats"),
        ranking_metrics=task_specific.get("ranking_metrics"),
        generated_at=datetime.fromisoformat(bundle.created_at.replace("Z", "+00:00")),
    )


def _classify_severity(delta: float, task_type: str) -> str | None:
    """Classify scenario severity based on delta magnitude."""
    abs_delta = abs(delta)
    if abs_delta >= 0.1:
        return "high"
    elif abs_delta >= 0.05:
        return "med"
    elif abs_delta >= 0.01:
        return "low"
    return None


def _extract_component_scores(bundle: ResultBundle, task_type: str) -> list[ComponentScore]:
    """Extract component scores from baseline diagnostics."""
    scores = []
    if not bundle.baseline:
        return scores

    baseline_diag = bundle.notes.get("baseline_diagnostics", {})

    # Binary calibration
    if "brier" in baseline_diag:
        scores.append(ComponentScore(
            name="brier_score", score=baseline_diag["brier"],
            weight=None, notes="Baseline Brier score",
        ))
    if "ece" in baseline_diag:
        scores.append(ComponentScore(
            name="ece_score", score=baseline_diag["ece"],
            weight=None, notes="Baseline ECE",
        ))

    # Multiclass calibration
    if "multiclass_brier" in baseline_diag:
        scores.append(ComponentScore(
            name="multiclass_brier", score=baseline_diag["multiclass_brier"],
            weight=None, notes="Multiclass Brier score",
        ))
    if "multiclass_ece" in baseline_diag:
        scores.append(ComponentScore(
            name="multiclass_ece", score=baseline_diag["multiclass_ece"],
            weight=None, notes="Multiclass ECE",
        ))

    # Task-specific summary scores from notes
    task_specific = bundle.notes.get("task_specific", {})
    residual_stats = task_specific.get("residual_stats", {})
    if residual_stats:
        scores.append(ComponentScore(
            name="mae", score=residual_stats.get("mae", 0.0),
            weight=None, notes="Mean Absolute Error",
        ))
        scores.append(ComponentScore(
            name="rmse", score=residual_stats.get("rmse", 0.0),
            weight=None, notes="Root Mean Squared Error",
        ))

    return scores


def _extract_flags(bundle: ResultBundle, task_type: str) -> list[FindingFlag]:
    """Extract finding flags from diagnostics."""
    flags = []
    if not bundle.baseline:
        return flags

    baseline_diag = bundle.notes.get("baseline_diagnostics", {})

    # Binary ECE warning
    if baseline_diag.get("ece", 0) > 0.1:
        flags.append(FindingFlag(
            code="high_ece",
            title="High Expected Calibration Error",
            detail=f"ECE is {baseline_diag.get('ece', 0):.4f}, indicating poor calibration",
            severity="warn",
        ))

    # Multiclass ECE warning
    if baseline_diag.get("multiclass_ece", 0) > 0.1:
        flags.append(FindingFlag(
            code="high_multiclass_ece",
            title="High Multiclass Calibration Error",
            detail=f"Multiclass ECE is {baseline_diag.get('multiclass_ece', 0):.4f}",
            severity="warn",
        ))

    # Regression high-residual warning
    task_specific = bundle.notes.get("task_specific", {})
    residual_stats = task_specific.get("residual_stats", {})
    if residual_stats:
        std = residual_stats.get("std", 0)
        max_abs = max(abs(residual_stats.get("min", 0)), abs(residual_stats.get("max", 0)))
        if max_abs > 3 * std and std > 0:
            flags.append(FindingFlag(
                code="high_residual_outliers",
                title="Large Residual Outliers",
                detail=f"Max residual ({max_abs:.4f}) exceeds 3x std ({std:.4f})",
                severity="warn",
            ))

    return flags
```

2. Run: `cd web/backend && python -c "from app.bundle_transform import bundle_to_result_summary; print('OK')"`
   Expected: OK

**Acceptance:** Transform returns `task_type`, `confusion_matrix`, `per_class_metrics`, `residual_stats` when present in bundle. Severity classification works for all deltas.

### Task A6: Update presets with task-type filtering

**Files:**
- Modify: `web/backend/app/storage.py`
- Modify: `web/backend/app/routers/presets.py`

**Steps:**

1. In `storage.py`, add `task_types` field to each metric preset. Replace the `METRIC_PRESETS` list:

```python
METRIC_PRESETS = [
    {
        "id": "auc",
        "name": "AUC-ROC",
        "description": "Area under the ROC curve",
        "requires_surface": ["probability", "score"],
        "task_types": ["binary_classification"],
    },
    {
        "id": "pr_auc",
        "name": "PR-AUC",
        "description": "Area under the Precision-Recall curve",
        "requires_surface": ["probability", "score"],
        "task_types": ["binary_classification"],
    },
    {
        "id": "accuracy",
        "name": "Accuracy",
        "description": "Classification accuracy",
        "requires_surface": ["label", "probability"],
        "task_types": ["binary_classification", "multiclass_classification"],
    },
    {
        "id": "logloss",
        "name": "Log Loss",
        "description": "Logarithmic loss (cross-entropy)",
        "requires_surface": ["probability"],
        "task_types": ["binary_classification", "multiclass_classification"],
    },
    {
        "id": "f1",
        "name": "F1 Score",
        "description": "Harmonic mean of precision and recall",
        "requires_surface": ["label", "probability"],
        "task_types": ["binary_classification", "multiclass_classification"],
    },
    {
        "id": "precision",
        "name": "Precision",
        "description": "Positive predictive value",
        "requires_surface": ["label", "probability"],
        "task_types": ["binary_classification", "multiclass_classification"],
    },
    {
        "id": "recall",
        "name": "Recall",
        "description": "True positive rate",
        "requires_surface": ["label", "probability"],
        "task_types": ["binary_classification", "multiclass_classification"],
    },
    {
        "id": "matthews_corrcoef",
        "name": "Matthews Corrcoef",
        "description": "Matthews correlation coefficient",
        "requires_surface": ["label", "probability"],
        "task_types": ["binary_classification", "multiclass_classification"],
    },
    {
        "id": "brier_score",
        "name": "Brier Score",
        "description": "Mean squared error of probabilistic predictions",
        "requires_surface": ["probability"],
        "task_types": ["binary_classification"],
    },
    {
        "id": "ece",
        "name": "Expected Calibration Error",
        "description": "Calibration error across probability bins",
        "requires_surface": ["probability"],
        "task_types": ["binary_classification", "multiclass_classification"],
    },
    # Regression metrics
    {
        "id": "mae",
        "name": "Mean Absolute Error",
        "description": "Average absolute prediction error",
        "requires_surface": ["continuous"],
        "task_types": ["regression"],
    },
    {
        "id": "mse",
        "name": "Mean Squared Error",
        "description": "Average squared prediction error",
        "requires_surface": ["continuous"],
        "task_types": ["regression"],
    },
    {
        "id": "rmse",
        "name": "Root Mean Squared Error",
        "description": "Square root of average squared error",
        "requires_surface": ["continuous"],
        "task_types": ["regression"],
    },
    {
        "id": "r_squared",
        "name": "R-Squared",
        "description": "Coefficient of determination",
        "requires_surface": ["continuous"],
        "task_types": ["regression"],
    },
    # Multiclass-specific
    {
        "id": "weighted_f1",
        "name": "Weighted F1",
        "description": "Weighted average F1 across classes",
        "requires_surface": ["label", "probability"],
        "task_types": ["multiclass_classification"],
    },
    {
        "id": "macro_f1",
        "name": "Macro F1",
        "description": "Unweighted average F1 across classes",
        "requires_surface": ["label", "probability"],
        "task_types": ["multiclass_classification"],
    },
    {
        "id": "cohens_kappa",
        "name": "Cohen's Kappa",
        "description": "Agreement beyond chance",
        "requires_surface": ["label", "probability"],
        "task_types": ["multiclass_classification"],
    },
]
```

2. In `routers/presets.py`, add `task_type` query parameter to the metrics endpoint:

Replace `get_metric_presets`:
```python
from fastapi import Query

@router.get("/metrics")
async def get_metric_presets(
    task_type: str | None = Query(None, description="Filter by task type"),
):
    """Get available metric presets, optionally filtered by task type."""
    if task_type is None:
        return METRIC_PRESETS
    return [p for p in METRIC_PRESETS if task_type in p.get("task_types", [])]
```

3. Add a `/models/formats` endpoint in `routers/models.py`. Add at the end:

```python
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
```

4. Run: `cd web/backend && python -c "from app.routers.presets import router; from app.routers.models import router; print('OK')"`
   Expected: OK

**Acceptance:** `GET /presets/metrics?task_type=regression` returns only regression metrics. `GET /models/formats` returns all supported formats.

### Task A7: Update model upload for multi-format

**Files:**
- Modify: `web/backend/app/routers/models.py`

**Steps:**

1. Update the `upload_model` endpoint to accept multiple extensions and task_type. Replace the endpoint function:

```python
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
```

2. Update `_resolve_model_path` in `engine_bridge.py` to handle non-.pkl files. Replace the local path resolution:

In the `else` branch (local), replace:
```python
    pkl_path = repo_root / ".spectra_ui" / "models" / owner_id / f"{model_id}.pkl"
```
With:
```python
    models_dir = repo_root / ".spectra_ui" / "models" / owner_id
    # Find model file by model_id with any extension
    candidates = list(models_dir.glob(f"{model_id}.*"))
    candidates = [c for c in candidates if not c.suffix == ".meta.json"]
    if not candidates:
        raise ValueError(f"Model {model_id} not found for owner")
    pkl_path = candidates[0]
```

3. Run: `ruff check web/backend/app/ --fix`
   Expected: No errors or auto-fixed

4. Run: `cd web/backend && python -c "from app.routers.models import router; print('OK')"`
   Expected: OK

**Acceptance:** Upload accepts `.pkl`, `.onnx`, `.joblib`, `.ubj`, `.xgb`, `.lgb`, `.cbm`. Multiclass models auto-detected.

### Task A8: Update experiments router to pass task_type

**Files:**
- Modify: `web/backend/app/routers/experiments.py`

**Steps:**

1. In `create_experiment` (line 57), add task_type to ExperimentSummary:

Replace the `summary = ExperimentSummary(...)` block:
```python
    task_type = create_req.config.get("task_type", "binary_classification") if create_req.config else "binary_classification"

    summary = ExperimentSummary(
        id=experiment_id,
        name=create_req.name,
        metric_id=create_req.metric_id,
        stress_suite_id=create_req.stress_suite_id,
        task_type=task_type,
        status="created",
        created_at=now,
        last_run_at=None,
        error_message=None,
    )
```

2. Run: `ruff check web/backend/app/ --fix`
3. Run: `cd web/backend && python -c "from app.routers.experiments import router; print('OK')"`
   Expected: OK

**Acceptance:** ExperimentSummary includes task_type from config.

### Task A9: Commit all backend changes

**Steps:**

1. Run full lint: `ruff check src/ web/backend/app/ --fix`
2. Run tests: `pytest tests/ -x -q --tb=short 2>&1 | tail -10`
3. Fix any failures.
4. Commit:

```bash
git add web/backend/app/ src/metrics_lie/execution.py
git commit -m "feat: Phase 5 Stream A — multi-task web backend

- Add task_type to API contracts (ExperimentSummary, ResultSummary, ModelUploadResponse)
- Add task-specific fields: confusion_matrix, per_class_metrics, residual_stats
- Generalize model validation for pickle/ONNX/boosting, binary/multiclass/regression
- Make engine bridge pass task_type from config to core engine
- Task-aware bundle transform with severity classification and diagnostic extraction
- Add task-type filtering to metric presets endpoint
- Add /models/formats endpoint listing supported model formats
- Compute confusion matrix and per-class metrics in execution.py for classification
- Compute residual stats in execution.py for regression"
```

**Acceptance:** All tests pass, lint clean, commit created.

## Boundaries

**DO:**
- Follow steps exactly as written
- Commit after all tasks pass
- Run the specified verification commands

**DO NOT:**
- Modify frontend files (that's Stream B)
- Add new dependencies
- Change the core engine logic beyond what's specified in Task A2
- Refactor existing test files

## Escalation Triggers

Stop and flag for Opus review if:
- Existing tests break after Task A2 changes to execution.py
- Import errors when loading modified modules
- The core engine's `ResultBundle` schema doesn't have the expected fields
- You need to modify files not listed in any task

When escalating, write to `.claude/plans/phase5-stream-a-blockers.md`.

## Verification

After all tasks complete:
- [ ] `ruff check src/ web/backend/app/` passes
- [ ] `pytest tests/ -x -q` passes (all existing tests)
- [ ] `cd web/backend && python -c "from app.main import app; print('OK')"` works
- [ ] No files modified outside the plan's file list
- [ ] Single commit with all changes
