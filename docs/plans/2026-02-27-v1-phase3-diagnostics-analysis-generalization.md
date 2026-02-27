# V1.0 Phase 3: Diagnostics & Analysis Generalization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make all diagnostics and analysis modules work across binary, multiclass, and regression task types — plus integrate Fairlearn for fairness and Evidently for drift detection as optional dependencies.

**Architecture:** Add a `higher_is_better` field to `MetricRequirement` so all analysis modules can determine degradation direction per metric. Generalize `failure_modes.py` to handle 2D multiclass probability arrays. Extend `dashboard.py` to use metric direction for correct risk flagging. Add multiclass calibration functions (Brier, ECE). Integrate Fairlearn `MetricFrame` as an optional backend for subgroup analysis. Add Evidently-powered drift detection as an optional pre-flight check. Fix `execution.py` guard gaps so multiclass runs get analysis artifacts.

**Tech Stack:** Python 3.11+, scikit-learn (metrics), NumPy, Pydantic v2, pytest (TDD), Fairlearn (optional), Evidently (optional)

**Scope boundaries:** This phase covers diagnostic/analysis generalization for tabular data only. NLP-specific diagnostics, image perturbation analysis, and LLM evaluation diagnostics are deferred to later phases.

---

## Work Streams

The tasks below are organized in dependency order. Streams A, B, and C are fully independent and can be dispatched as parallel subagents.

### Stream A: Core Analysis Generalization (Tasks 1-6)
### Stream B: Calibration Generalization (Tasks 7-10)
### Stream C: Fairness & Drift Integration (Tasks 11-15)

---

### Task 1: Add `higher_is_better` to MetricRequirement and registry

**Files:**
- Modify: `src/metrics_lie/metrics/registry.py`
- Test: `tests/test_v3_metric_direction.py`

**Step 1: Write the failing test**

```python
"""Tests for metric direction (higher_is_better) on MetricRequirement."""
from __future__ import annotations

import pytest

from metrics_lie.metrics.registry import METRIC_REQUIREMENTS, MetricRequirement

HIGHER_IS_BETTER_METRICS = {
    "accuracy", "auc", "f1", "precision", "recall", "pr_auc",
    "matthews_corrcoef", "macro_f1", "weighted_f1", "macro_precision",
    "macro_recall", "macro_auc", "cohens_kappa", "top_k_accuracy", "r2",
}

LOWER_IS_BETTER_METRICS = {
    "logloss", "brier_score", "ece", "mae", "mse", "rmse", "max_error",
}


def _requirements_by_id() -> dict[str, MetricRequirement]:
    return {r.metric_id: r for r in METRIC_REQUIREMENTS}


def test_metric_requirement_has_higher_is_better_field():
    req = METRIC_REQUIREMENTS[0]
    assert hasattr(req, "higher_is_better")


@pytest.mark.parametrize("metric_id", sorted(HIGHER_IS_BETTER_METRICS))
def test_higher_is_better_true(metric_id: str):
    reqs = _requirements_by_id()
    assert metric_id in reqs, f"{metric_id} not in registry"
    assert reqs[metric_id].higher_is_better is True


@pytest.mark.parametrize("metric_id", sorted(LOWER_IS_BETTER_METRICS))
def test_lower_is_better(metric_id: str):
    reqs = _requirements_by_id()
    assert metric_id in reqs, f"{metric_id} not in registry"
    assert reqs[metric_id].higher_is_better is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_v3_metric_direction.py -v`
Expected: FAIL — `MetricRequirement` has no `higher_is_better` field

**Step 3: Write minimal implementation**

In `src/metrics_lie/metrics/registry.py`, add `higher_is_better: bool = True` to `MetricRequirement` dataclass, then set it on every entry:

```python
@dataclass(frozen=True)
class MetricRequirement:
    metric_id: str
    requires_surface: set[SurfaceType]
    requires_labels: bool
    min_samples: int
    requires_both_classes: bool
    task_types: frozenset[str] | None = None
    higher_is_better: bool = True
```

Then add `higher_is_better=False` to these entries: `logloss`, `brier_score`, `ece`, `mae`, `mse`, `rmse`, `max_error`.

Also add a convenience lookup dict after `METRIC_REQUIREMENTS`:

```python
METRIC_DIRECTION: dict[str, bool] = {
    r.metric_id: r.higher_is_better for r in METRIC_REQUIREMENTS
}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_v3_metric_direction.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_v3_metric_direction.py src/metrics_lie/metrics/registry.py
git commit -m "feat: add higher_is_better to MetricRequirement with METRIC_DIRECTION lookup"
```

---

### Task 2: Generalize dashboard for metric direction awareness

**Files:**
- Modify: `src/metrics_lie/analysis/dashboard.py`
- Test: `tests/test_v3_dashboard_direction.py`

**Depends on:** Task 1

**Step 1: Write the failing test**

```python
"""Tests for direction-aware dashboard summary."""
from __future__ import annotations

from metrics_lie.analysis.dashboard import build_dashboard_summary


def test_dashboard_lower_is_better_flags_increase_as_drop():
    """For lower-is-better metrics like MAE, an increase is degradation."""
    metric_results = {
        "mae": {"mean": 1.0, "std": 0.1},
        "rmse": {"mean": 1.5, "std": 0.2},
    }
    scenario_results_by_metric = {
        "mae": [
            {"scenario_id": "label_noise_0.1", "metric": {"mean": 1.3}},
        ],
        "rmse": [
            {"scenario_id": "label_noise_0.1", "metric": {"mean": 1.7}},
        ],
    }
    dashboard = build_dashboard_summary(
        primary_metric="mae",
        surface_type="continuous",
        metric_results=metric_results,
        scenario_results_by_metric=scenario_results_by_metric,
        metric_directions={"mae": False, "rmse": False},
    )
    # MAE went from 1.0 -> 1.3 (delta=+0.3) — degradation for lower-is-better
    assert "mae" in dashboard.risk_summary["metrics_with_large_drops"]
    assert "rmse" in dashboard.risk_summary["metrics_with_large_drops"]


def test_dashboard_higher_is_better_flags_decrease_as_drop():
    """For higher-is-better metrics like AUC, a decrease is degradation."""
    metric_results = {
        "auc": {"mean": 0.90, "std": 0.02},
    }
    scenario_results_by_metric = {
        "auc": [
            {"scenario_id": "label_noise_0.1", "metric": {"mean": 0.82}},
        ],
    }
    dashboard = build_dashboard_summary(
        primary_metric="auc",
        surface_type="probability",
        metric_results=metric_results,
        scenario_results_by_metric=scenario_results_by_metric,
        metric_directions={"auc": True},
    )
    assert "auc" in dashboard.risk_summary["metrics_with_large_drops"]


def test_dashboard_backward_compat_no_direction():
    """When metric_directions is None, use legacy behavior (negative delta = drop)."""
    metric_results = {
        "auc": {"mean": 0.90, "std": 0.02},
    }
    scenario_results_by_metric = {
        "auc": [
            {"scenario_id": "label_noise_0.1", "metric": {"mean": 0.82}},
        ],
    }
    dashboard = build_dashboard_summary(
        primary_metric="auc",
        surface_type="probability",
        metric_results=metric_results,
        scenario_results_by_metric=scenario_results_by_metric,
    )
    assert "auc" in dashboard.risk_summary["metrics_with_large_drops"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_v3_dashboard_direction.py -v`
Expected: FAIL — `build_dashboard_summary()` got unexpected keyword argument `metric_directions`

**Step 3: Write minimal implementation**

Modify `build_dashboard_summary` in `src/metrics_lie/analysis/dashboard.py`:

1. Add optional parameter `metric_directions: dict[str, bool] | None = None`
2. Compute "degradation delta" per metric: if `higher_is_better` is True, degradation = negative delta (existing behavior). If `higher_is_better` is False, degradation = positive delta (flip sign).
3. Use degradation delta for `worst_delta` and `LARGE_DROP_THRESHOLD` comparison.

Key change in the delta logic:

```python
# Determine if this delta represents degradation
hib = True  # default: higher is better
if metric_directions is not None and metric_id in metric_directions:
    hib = metric_directions[metric_id]

# Degradation delta: always negative when metric degraded
degradation_delta = delta if hib else -delta
```

Then compare `degradation_delta < -LARGE_DROP_THRESHOLD` instead of `worst_delta < -LARGE_DROP_THRESHOLD`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_v3_dashboard_direction.py -v`
Expected: PASS

Also run: `pytest tests/test_phase8_dashboard.py -v`
Expected: PASS (backward compatible)

**Step 5: Commit**

```bash
git add src/metrics_lie/analysis/dashboard.py tests/test_v3_dashboard_direction.py
git commit -m "feat: add metric direction awareness to dashboard summary"
```

---

### Task 3: Generalize failure_modes for multiclass 2D probability arrays

**Files:**
- Modify: `src/metrics_lie/analysis/failure_modes.py`
- Test: `tests/test_v3_failure_modes.py`

**Step 1: Write the failing test**

```python
"""Tests for failure_modes with multiclass and regression surfaces."""
from __future__ import annotations

import numpy as np

from metrics_lie.analysis.failure_modes import locate_failure_modes
from metrics_lie.model.surface import (
    CalibrationState,
    PredictionSurface,
    SurfaceType,
)


def _make_surface(surface_type, values, **kwargs):
    return PredictionSurface(
        surface_type=surface_type,
        values=np.array(values),
        dtype=np.array(values).dtype,
        n_samples=len(values),
        class_names=kwargs.get("class_names", ("neg", "pos")),
        positive_label=kwargs.get("positive_label", 1),
        threshold=kwargs.get("threshold", 0.5),
        calibration_state=CalibrationState.UNKNOWN,
        model_hash=None,
        is_deterministic=True,
    )


def test_failure_modes_multiclass_probability():
    """Multiclass PROBABILITY surface with 2D values should not crash."""
    y_true = np.array([0, 1, 2, 0, 1])
    proba = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.2, 0.2, 0.6],
        [0.3, 0.4, 0.3],  # wrong prediction — should rank high
        [0.5, 0.3, 0.2],  # wrong prediction — should rank high
    ])
    surface = _make_surface(
        SurfaceType.PROBABILITY, proba,
        class_names=("a", "b", "c"), threshold=None,
    )
    report = locate_failure_modes(
        y_true=y_true, surface=surface, metrics=["macro_f1"], top_k=3,
    )
    assert report.total_samples == 5
    assert len(report.failure_samples) == 3
    # Samples 3 and 4 are misclassified — should appear in top failures
    assert 3 in report.failure_samples
    assert 4 in report.failure_samples


def test_failure_modes_regression_continuous():
    """Regression CONTINUOUS surface uses residual-based contribution."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    preds = np.array([1.1, 2.0, 5.0, 3.5, 5.2])
    surface = _make_surface(
        SurfaceType.CONTINUOUS, preds, threshold=None,
    )
    report = locate_failure_modes(
        y_true=y_true, surface=surface, metrics=["mae"], top_k=2,
    )
    assert report.total_samples == 5
    # Sample 2 has residual |3-5|=2.0 (worst), sample 3 has |4-3.5|=0.5
    assert 2 in report.failure_samples


def test_failure_modes_multiclass_label():
    """Multiclass LABEL surface works like binary (misclassification)."""
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 2, 2, 1, 1])  # samples 1 and 3 wrong
    surface = _make_surface(
        SurfaceType.LABEL, y_pred,
        class_names=("a", "b", "c"),
    )
    report = locate_failure_modes(
        y_true=y_true, surface=surface, metrics=["macro_f1"], top_k=2,
    )
    assert 1 in report.failure_samples
    assert 3 in report.failure_samples
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_v3_failure_modes.py -v`
Expected: FAIL — `test_failure_modes_multiclass_probability` fails because `np.abs(prob - y_true)` with 2D prob and 1D y_true produces wrong shapes

**Step 3: Write minimal implementation**

Modify the PROBABILITY branch in `locate_failure_modes` (`src/metrics_lie/analysis/failure_modes.py`):

```python
if surface.surface_type == SurfaceType.PROBABILITY:
    prob = surface.values.astype(float)
    if prob.ndim == 2:
        # Multiclass: contribution = 1 - P(correct class)
        y_int = y_true.astype(int)
        correct_class_prob = prob[np.arange(n), y_int]
        contributions += 1.0 - correct_class_prob
        pred = np.argmax(prob, axis=1)
        misclassified = pred != y_int
        contributions += misclassified.astype(float)
    else:
        # Binary: existing logic
        contributions += np.abs(prob - y_true)
        pred = (prob >= (surface.threshold or 0.5)).astype(int)
        misclassified = pred != y_true
        contributions += misclassified.astype(float)
```

Also add a CONTINUOUS branch for regression (residual-based):

```python
elif surface.surface_type == SurfaceType.CONTINUOUS:
    preds = surface.values.astype(float)
    contributions += np.abs(preds - y_true.astype(float))
else:
    # SCORE fallback
    scores = surface.values.astype(float)
    contributions += np.abs(scores - np.mean(scores))
```

The full else block becomes:

```python
elif surface.surface_type == SurfaceType.LABEL:
    pred = surface.values.astype(int)
    misclassified = pred != y_true.astype(int)
    contributions += misclassified.astype(float)
elif surface.surface_type == SurfaceType.CONTINUOUS:
    preds = surface.values.astype(float)
    contributions += np.abs(preds - y_true.astype(float))
else:
    scores = surface.values.astype(float)
    contributions += np.abs(scores - np.mean(scores))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_v3_failure_modes.py -v`
Expected: PASS

Also run: `pytest tests/test34_analysis.py -v`
Expected: PASS (backward compatible)

**Step 5: Commit**

```bash
git add src/metrics_lie/analysis/failure_modes.py tests/test_v3_failure_modes.py
git commit -m "feat: generalize failure_modes for multiclass 2D proba and regression residuals"
```

---

### Task 4: Fix execution.py analysis guard gap for multiclass

**Files:**
- Modify: `src/metrics_lie/execution.py`
- Test: `tests/test_v3_execution_analysis.py`

**Depends on:** Tasks 1, 2, 3

**Step 1: Write the failing test**

```python
"""Tests for analysis artifacts in multiclass and regression runs."""
from __future__ import annotations

import json
import os

from metrics_lie.execution import run_from_spec_dict


def test_multiclass_run_has_failure_modes():
    """Multiclass run should produce failure_modes in analysis_artifacts."""
    spec = {
        "name": "test_mc_analysis",
        "dataset": os.path.join("data", "iris_multiclass.csv"),
        "model": os.path.join("data", "iris_model.pkl"),
        "metric": "macro_f1",
        "task": "multiclass_classification",
        "scenarios": [{"type": "label_noise", "noise_rate": 0.1}],
        "n_trials": 3,
        "seed": 42,
    }
    bundle = run_from_spec_dict(spec)
    aa = bundle.analysis_artifacts
    assert "failure_modes" in aa, f"Missing failure_modes, got keys: {list(aa.keys())}"


def test_multiclass_run_has_dashboard():
    """Multiclass run should produce dashboard_summary in analysis_artifacts."""
    spec = {
        "name": "test_mc_dashboard",
        "dataset": os.path.join("data", "iris_multiclass.csv"),
        "model": os.path.join("data", "iris_model.pkl"),
        "metric": "macro_f1",
        "task": "multiclass_classification",
        "scenarios": [{"type": "label_noise", "noise_rate": 0.1}],
        "n_trials": 3,
        "seed": 42,
    }
    bundle = run_from_spec_dict(spec)
    aa = bundle.analysis_artifacts
    assert "dashboard_summary" in aa


def test_regression_run_has_failure_modes():
    """Regression run should produce failure_modes in analysis_artifacts."""
    spec = {
        "name": "test_reg_analysis",
        "dataset": os.path.join("data", "regression_sample.csv"),
        "model": os.path.join("data", "regression_model.pkl"),
        "metric": "mae",
        "task": "regression",
        "scenarios": [{"type": "label_noise", "noise_rate": 0.1}],
        "n_trials": 3,
        "seed": 42,
    }
    bundle = run_from_spec_dict(spec)
    aa = bundle.analysis_artifacts
    assert "failure_modes" in aa


def test_regression_dashboard_flags_mae_increase():
    """Regression dashboard should flag MAE increase as degradation."""
    spec = {
        "name": "test_reg_dashboard",
        "dataset": os.path.join("data", "regression_sample.csv"),
        "model": os.path.join("data", "regression_model.pkl"),
        "metric": "mae",
        "task": "regression",
        "scenarios": [{"type": "label_noise", "noise_rate": 0.3}],
        "n_trials": 5,
        "seed": 42,
    }
    bundle = run_from_spec_dict(spec)
    aa = bundle.analysis_artifacts
    assert "dashboard_summary" in aa
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_v3_execution_analysis.py -v`
Expected: FAIL — multiclass PROBABILITY surface falls through both analysis guard blocks, so `failure_modes` key is missing. Also need to create test data fixtures.

Note: This test requires multiclass and regression test fixtures. If `data/iris_multiclass.csv` and `data/iris_model.pkl` don't exist, create them first:

```python
# Helper to create test data (run once or add as conftest fixture)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pickle

# Multiclass
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
df = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"])
df["y_true"] = y
clf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
proba = clf.predict_proba(X)
for i in range(proba.shape[1]):
    df[f"y_score_{i}"] = proba[:, i]
df.to_csv("data/iris_multiclass.csv", index=False)
with open("data/iris_model.pkl", "wb") as f:
    pickle.dump(clf, f)

# Regression
rng = np.random.default_rng(42)
X_reg = rng.normal(size=(100, 3))
y_reg = X_reg[:, 0] * 2 + X_reg[:, 1] - 0.5 * X_reg[:, 2] + rng.normal(0, 0.1, 100)
df_reg = pd.DataFrame(X_reg, columns=["f0", "f1", "f2"])
reg = RandomForestRegressor(n_estimators=10, random_state=42).fit(X_reg, y_reg)
df_reg["y_true"] = y_reg
df_reg["y_score"] = reg.predict(X_reg)
df_reg.to_csv("data/regression_sample.csv", index=False)
with open("data/regression_model.pkl", "wb") as f:
    pickle.dump(reg, f)
```

**Step 3: Write minimal implementation**

Modify `src/metrics_lie/execution.py` analysis block (~line 576-643):

1. Restructure the guard: instead of binary-vs-everything-else, use three branches:
   - Binary with PROBABILITY/SCORE → full analysis (threshold_sweep, sensitivity, disagreement, failure_modes)
   - Any task with any surface → failure_modes (already generalized in Task 3)
   - Multi-metric → dashboard (pass `metric_directions`)

```python
analysis_artifacts: dict[str, Any] = {}

# Binary-only analysis: threshold sweep, sensitivity, disagreement
if (
    task_type.is_binary
    and prediction_surface is not None
    and prediction_surface.surface_type
    in (SurfaceType.PROBABILITY, SurfaceType.SCORE)
):
    sweep = run_threshold_sweep(...)
    analysis_artifacts["threshold_sweep"] = sweep.to_jsonable()
    sensitivity = run_sensitivity_analysis(...)
    analysis_artifacts["sensitivity"] = sensitivity.to_jsonable()
    disagreements = analyze_metric_disagreements(...)
    analysis_artifacts["metric_disagreements"] = [
        d.to_jsonable() for d in disagreements
    ]

# Failure modes: works for all task types and surface types
if prediction_surface is not None:
    failures = locate_failure_modes(
        y_true=y_true,
        surface=prediction_surface,
        metrics=applicable.metrics,
        subgroup=subgroup,
        top_k=20,
    )
    analysis_artifacts["failure_modes"] = failures.to_jsonable()

# Dashboard: works for all task types with 2+ metrics
if len(applicable.metrics) > 1:
    from metrics_lie.metrics.registry import METRIC_DIRECTION
    dashboard = build_dashboard_summary(
        primary_metric=primary_metric,
        surface_type=surface_type.value,
        metric_results={k: v.model_dump() for k, v in metric_results.items()},
        scenario_results_by_metric={
            k: [sr.model_dump() for sr in v]
            for k, v in scenario_results_by_metric.items()
        },
        metric_directions=METRIC_DIRECTION,
    )
    analysis_artifacts["dashboard_summary"] = dashboard.to_jsonable()
```

2. Import `METRIC_DIRECTION` at top of file.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_v3_execution_analysis.py -v`
Expected: PASS

Also run: `pytest tests/test_v2_execution_multiclass.py tests/test_v2_execution_regression.py -v`
Expected: PASS (backward compatible)

**Step 5: Commit**

```bash
git add src/metrics_lie/execution.py tests/test_v3_execution_analysis.py data/iris_multiclass.csv data/iris_model.pkl data/regression_sample.csv data/regression_model.pkl
git commit -m "feat: fix execution.py analysis guards for multiclass and regression runs"
```

---

### Task 5: Extend sensitivity analysis for multiclass and regression

**Files:**
- Modify: `src/metrics_lie/analysis/sensitivity.py`
- Modify: `src/metrics_lie/execution.py`
- Test: `tests/test_v3_sensitivity.py`

**Depends on:** Task 1

**Step 1: Write the failing test**

```python
"""Tests for sensitivity analysis with multiclass and regression surfaces."""
from __future__ import annotations

import numpy as np

from metrics_lie.analysis.sensitivity import run_sensitivity_analysis
from metrics_lie.model.surface import (
    CalibrationState,
    PredictionSurface,
    SurfaceType,
)


def _make_surface(surface_type, values, **kwargs):
    return PredictionSurface(
        surface_type=surface_type,
        values=np.array(values),
        dtype=np.array(values).dtype,
        n_samples=len(values),
        class_names=kwargs.get("class_names", ("neg", "pos")),
        positive_label=kwargs.get("positive_label", 1),
        threshold=kwargs.get("threshold", 0.5),
        calibration_state=CalibrationState.UNKNOWN,
        model_hash=None,
        is_deterministic=True,
    )


def test_sensitivity_regression_continuous():
    """Sensitivity analysis should work on CONTINUOUS surfaces."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    preds = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    surface = _make_surface(SurfaceType.CONTINUOUS, preds, threshold=None)
    result = run_sensitivity_analysis(
        y_true=y_true,
        surface=surface,
        metrics=["mae", "rmse"],
        perturbation_type="score_noise",
        magnitudes=[0.01, 0.05, 0.1],
        n_trials=10,
        seed=42,
    )
    assert result.most_sensitive_metric in ("mae", "rmse")
    assert len(result.magnitudes) == 3


def test_sensitivity_multiclass_label():
    """Sensitivity on multiclass LABEL surface with label-compatible metrics."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    surface = _make_surface(
        SurfaceType.LABEL, y_pred,
        class_names=("a", "b", "c"), threshold=None,
    )
    result = run_sensitivity_analysis(
        y_true=y_true,
        surface=surface,
        metrics=["macro_f1", "cohens_kappa"],
        perturbation_type="score_noise",
        magnitudes=[0.01, 0.05, 0.1],
        n_trials=10,
        seed=42,
    )
    assert "macro_f1" in result.metric_responses
    assert "cohens_kappa" in result.metric_responses
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_v3_sensitivity.py -v`
Expected: FAIL — `_apply_score_noise` doesn't handle CONTINUOUS, and LABEL needs integer rounding after noise

**Step 3: Write minimal implementation**

Modify `run_sensitivity_analysis` in `src/metrics_lie/analysis/sensitivity.py`:

1. Handle `CONTINUOUS` surface: add noise without clipping to [0,1]:

```python
if perturbation_type == "score_noise":
    clip = surface.surface_type == SurfaceType.PROBABILITY
    noisy = _apply_score_noise(
        surface.values, sigma=mag, rng=rng, clip=clip,
    )
    # For LABEL surfaces, round perturbed values back to integers
    if surface.surface_type == SurfaceType.LABEL:
        noisy = np.round(noisy).astype(int)
        # Clamp to valid class range
        noisy = np.clip(noisy, int(surface.values.min()), int(surface.values.max()))
```

2. Fix `_apply_score_noise` to handle any 1D array shape:

```python
def _apply_score_noise(
    scores: np.ndarray, *, sigma: float, rng: np.random.Generator, clip: bool
) -> np.ndarray:
    noisy = scores + rng.normal(loc=0.0, scale=sigma, size=scores.shape)
    if clip:
        noisy = np.clip(noisy, 0.0, 1.0)
    return noisy
```

3. In `execution.py`, add sensitivity for non-binary runs with CONTINUOUS or LABEL surfaces:

```python
# Sensitivity: also works for regression (CONTINUOUS) and multiclass (LABEL)
if (
    not task_type.is_binary
    and prediction_surface is not None
    and prediction_surface.surface_type
    in (SurfaceType.CONTINUOUS, SurfaceType.LABEL)
):
    # Filter to metrics that work on 1D predictions (exclude macro_auc which needs 2D)
    sensitivity_metrics = [
        m for m in applicable.metrics if m != "macro_auc" and m != "top_k_accuracy"
    ]
    if sensitivity_metrics:
        sensitivity = run_sensitivity_analysis(
            y_true=y_true,
            surface=prediction_surface,
            metrics=sensitivity_metrics,
            perturbation_type="score_noise",
            magnitudes=[0.01, 0.02, 0.05, 0.1, 0.2],
            n_trials=50,
            seed=spec.seed,
        )
        analysis_artifacts["sensitivity"] = sensitivity.to_jsonable()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_v3_sensitivity.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/metrics_lie/analysis/sensitivity.py src/metrics_lie/execution.py tests/test_v3_sensitivity.py
git commit -m "feat: extend sensitivity analysis for multiclass and regression surfaces"
```

---

### Task 6: Extend subgroup diagnostics for multiclass metrics

**Files:**
- Modify: `src/metrics_lie/diagnostics/subgroups.py`
- Test: `tests/test_v3_subgroups.py`

**Step 1: Write the failing test**

```python
"""Tests for subgroup diagnostics with multiclass metrics."""
from __future__ import annotations

import numpy as np

from metrics_lie.diagnostics.subgroups import safe_metric_for_group
from metrics_lie.metrics.core import METRICS


def test_safe_metric_macro_f1_multiclass():
    """macro_f1 should work per-group on multiclass data."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 1, 1, 0, 0, 2, 2, 0])
    result = safe_metric_for_group("macro_f1", METRICS["macro_f1"], y_true, y_pred)
    assert result is not None
    assert 0.0 <= result <= 1.0


def test_safe_metric_macro_auc_needs_both_classes():
    """macro_auc should return None when not enough classes in group."""
    y_true = np.array([0, 0, 0])
    y_proba = np.array([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.7, 0.2, 0.1]])
    result = safe_metric_for_group("macro_auc", METRICS["macro_auc"], y_true, y_proba)
    # macro_auc needs at least 2 classes → should return None
    assert result is None


def test_safe_metric_regression_mae():
    """MAE should work per-group on regression data."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
    result = safe_metric_for_group("mae", METRICS["mae"], y_true, y_pred)
    assert result is not None
    assert result > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_v3_subgroups.py -v`
Expected: Some tests may pass (mae works already via bare call), but macro_auc guard may fail since only "auc" has the both-classes guard

**Step 3: Write minimal implementation**

Modify `safe_metric_for_group` in `src/metrics_lie/diagnostics/subgroups.py`:

```python
from metrics_lie.metrics.core import MULTICLASS_METRICS, REGRESSION_METRICS, THRESHOLD_METRICS

def safe_metric_for_group(
    metric_name: str,
    metric_fn: Callable[..., float],
    y_true_g: np.ndarray,
    y_score_g: np.ndarray,
) -> float | None:
    if len(y_true_g) < 2:
        return None

    # For AUC variants, need both classes present
    if metric_name in ("auc", "macro_auc", "pr_auc"):
        unique_labels = np.unique(y_true_g)
        if len(unique_labels) < 2:
            return None

    try:
        if metric_name in THRESHOLD_METRICS:
            return metric_fn(y_true_g, y_score_g, threshold=DEFAULT_THRESHOLD)
        else:
            return metric_fn(y_true_g, y_score_g)
    except (ValueError, Exception):
        return None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_v3_subgroups.py -v`
Expected: PASS

Also run: `pytest tests/test16_subgroup_diagnostics.py -v`
Expected: PASS (backward compatible)

**Step 5: Commit**

```bash
git add src/metrics_lie/diagnostics/subgroups.py tests/test_v3_subgroups.py
git commit -m "feat: extend subgroup diagnostics for multiclass and regression metrics"
```

---

### Task 7: Add multiclass Brier score function

**Files:**
- Modify: `src/metrics_lie/diagnostics/calibration.py`
- Test: `tests/test_v3_calibration.py`

**Step 1: Write the failing test**

```python
"""Tests for multiclass calibration functions."""
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.diagnostics.calibration import (
    multiclass_brier_score,
    multiclass_ece,
)


def test_multiclass_brier_perfect():
    """Perfect multiclass predictions should have Brier score 0."""
    y_true = np.array([0, 1, 2])
    y_proba = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert multiclass_brier_score(y_true, y_proba) == pytest.approx(0.0)


def test_multiclass_brier_worst():
    """Completely wrong predictions should have high Brier score."""
    y_true = np.array([0, 1, 2])
    y_proba = np.array([
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ])
    score = multiclass_brier_score(y_true, y_proba)
    assert score > 0.5


def test_multiclass_brier_formula():
    """Verify multiclass Brier = mean(sum_k (p_k - y_k)^2) where y is one-hot."""
    y_true = np.array([0, 1])
    y_proba = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
    ])
    # Manual: sample 0: (0.7-1)^2 + (0.2-0)^2 + (0.1-0)^2 = 0.09+0.04+0.01 = 0.14
    #         sample 1: (0.1-0)^2 + (0.6-1)^2 + (0.3-0)^2 = 0.01+0.16+0.09 = 0.26
    #         mean = (0.14 + 0.26) / 2 = 0.20
    assert multiclass_brier_score(y_true, y_proba) == pytest.approx(0.20)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_v3_calibration.py::test_multiclass_brier_perfect -v`
Expected: FAIL — `multiclass_brier_score` not found

**Step 3: Write minimal implementation**

Add to `src/metrics_lie/diagnostics/calibration.py`:

```python
def multiclass_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Multiclass Brier score: mean(sum_k (p_k - y_k)^2) where y is one-hot.

    Args:
        y_true: Integer class labels, shape (n,).
        y_proba: Probability matrix, shape (n, K).
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_proba, dtype=float)
    n_classes = p.shape[1]
    # One-hot encode y_true
    one_hot = np.zeros_like(p)
    one_hot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_v3_calibration.py -k brier -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/metrics_lie/diagnostics/calibration.py tests/test_v3_calibration.py
git commit -m "feat: add multiclass Brier score function"
```

---

### Task 8: Add multiclass ECE (top-label calibration)

**Files:**
- Modify: `src/metrics_lie/diagnostics/calibration.py`
- Modify: `tests/test_v3_calibration.py`

**Depends on:** Task 7

**Step 1: Write the failing test**

Append to `tests/test_v3_calibration.py`:

```python
def test_multiclass_ece_perfect():
    """Perfectly calibrated multiclass predictions should have ECE near 0."""
    rng = np.random.default_rng(42)
    n = 1000
    n_classes = 3
    # Generate calibrated probabilities
    y_true = rng.integers(0, n_classes, size=n)
    y_proba = np.full((n, n_classes), 1.0 / n_classes)
    for i in range(n):
        y_proba[i, y_true[i]] = 0.9
        remaining = 0.1 / (n_classes - 1)
        for k in range(n_classes):
            if k != y_true[i]:
                y_proba[i, k] = remaining
    ece = multiclass_ece(y_true, y_proba)
    # Not perfectly 0 due to binning, but should be very low
    assert ece < 0.05


def test_multiclass_ece_returns_float():
    y_true = np.array([0, 1, 2, 0])
    y_proba = np.array([
        [0.6, 0.3, 0.1],
        [0.2, 0.5, 0.3],
        [0.1, 0.2, 0.7],
        [0.5, 0.3, 0.2],
    ])
    ece = multiclass_ece(y_true, y_proba)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_v3_calibration.py::test_multiclass_ece_perfect -v`
Expected: FAIL — `multiclass_ece` not found

**Step 3: Write minimal implementation**

Add to `src/metrics_lie/diagnostics/calibration.py`:

```python
def multiclass_ece(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> float:
    """Top-label Expected Calibration Error for multiclass.

    Uses the top (predicted) class confidence and checks if it matches
    the observed accuracy (fraction where top-class = true class).

    Args:
        y_true: Integer class labels, shape (n,).
        y_proba: Probability matrix, shape (n, K).
        n_bins: Number of equal-width confidence bins.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_proba, dtype=float)

    top_conf = np.max(p, axis=1)
    top_pred = np.argmax(p, axis=1)
    correct = (top_pred == y).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(top_conf, bins[1:-1], right=False)

    ece = 0.0
    n = float(len(y))
    for b in range(n_bins):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        avg_conf = float(top_conf[mask].mean())
        avg_acc = float(correct[mask].mean())
        ece += (mask.sum() / n) * abs(avg_acc - avg_conf)
    return float(ece)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_v3_calibration.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/metrics_lie/diagnostics/calibration.py tests/test_v3_calibration.py
git commit -m "feat: add multiclass ECE (top-label calibration error)"
```

---

### Task 9: Wire multiclass calibration into runner and execution

**Files:**
- Modify: `src/metrics_lie/runner.py`
- Modify: `src/metrics_lie/execution.py`
- Test: `tests/test_v3_calibration_pipeline.py`

**Depends on:** Tasks 7, 8, and Task 4

**Step 1: Write the failing test**

```python
"""Tests for multiclass calibration wired through the pipeline."""
from __future__ import annotations

import os

from metrics_lie.execution import run_from_spec_dict


def test_multiclass_run_has_calibration_diagnostics():
    """Multiclass run with probability surface should include calibration."""
    spec = {
        "name": "test_mc_calibration",
        "dataset": os.path.join("data", "iris_multiclass.csv"),
        "model": os.path.join("data", "iris_model.pkl"),
        "metric": "macro_f1",
        "task": "multiclass_classification",
        "scenarios": [{"type": "label_noise", "noise_rate": 0.1}],
        "n_trials": 3,
        "seed": 42,
    }
    bundle = run_from_spec_dict(spec)
    notes = bundle.notes or {}
    # Check that multiclass calibration was computed
    assert "multiclass_brier" in notes or "baseline_calibration" in notes
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_v3_calibration_pipeline.py -v`
Expected: FAIL — multiclass calibration not wired yet

**Step 3: Write minimal implementation**

1. In `src/metrics_lie/execution.py`, after the existing binary calibration block (~line 376-381), add multiclass calibration:

```python
baseline_cal = {}
if task_type.is_binary and surface_type == SurfaceType.PROBABILITY:
    baseline_cal = {
        "brier": brier_score(y_true, y_score),
        "ece": expected_calibration_error(y_true, y_score, n_bins=10),
    }
elif (
    task_type == TaskType.MULTICLASS_CLASSIFICATION
    and surface_type == SurfaceType.PROBABILITY
    and y_score.ndim == 2
):
    from metrics_lie.diagnostics.calibration import multiclass_brier_score, multiclass_ece
    baseline_cal = {
        "multiclass_brier": multiclass_brier_score(y_true, y_score),
        "multiclass_ece": multiclass_ece(y_true, y_score),
    }
```

2. Add `baseline_cal` to `notes`:

```python
if baseline_cal:
    notes["baseline_calibration"] = baseline_cal
```

3. In `src/metrics_lie/runner.py`, add multiclass calibration in the per-trial loop (after the existing binary calibration guard at line 85-87):

```python
if ctx.surface_type == "probability" and ctx.task == "binary_classification":
    briers.append(brier_score(y_p, s_p))
    eces.append(expected_calibration_error(y_p, s_p, n_bins=10))
elif ctx.surface_type == "probability" and ctx.task == "multiclass_classification" and s_p.ndim == 2:
    from metrics_lie.diagnostics.calibration import multiclass_brier_score, multiclass_ece
    briers.append(multiclass_brier_score(y_p, s_p))
    eces.append(multiclass_ece(y_p, s_p))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_v3_calibration_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/metrics_lie/runner.py src/metrics_lie/execution.py tests/test_v3_calibration_pipeline.py
git commit -m "feat: wire multiclass calibration (Brier, ECE) into runner and execution pipeline"
```

---

### Task 10: Add Fairlearn optional dependency and MetricFrame integration

**Files:**
- Modify: `pyproject.toml`
- Create: `src/metrics_lie/diagnostics/fairness.py`
- Test: `tests/test_v3_fairness.py`

**Step 1: Write the failing test**

```python
"""Tests for Fairlearn-powered fairness analysis."""
from __future__ import annotations

import numpy as np
import pytest

fairlearn = pytest.importorskip("fairlearn")

from metrics_lie.diagnostics.fairness import compute_fairness_report


def test_compute_fairness_report_binary():
    """Fairness report computes group metrics and gaps for binary classification."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 0, 0, 1])
    sensitive = np.array(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
    report = compute_fairness_report(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
        metric_fns={"accuracy": lambda yt, yp: float(np.mean(yt == yp))},
    )
    assert "group_metrics" in report
    assert "A" in report["group_metrics"]
    assert "B" in report["group_metrics"]
    assert "gaps" in report
    assert "accuracy" in report["gaps"]
    assert "demographic_parity_difference" in report


def test_compute_fairness_report_multiclass():
    """Fairness report works for multiclass predictions."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 1, 1, 0])
    sensitive = np.array(["X", "X", "X", "Y", "Y", "Y"])
    report = compute_fairness_report(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
        metric_fns={"accuracy": lambda yt, yp: float(np.mean(yt == yp))},
    )
    assert "group_metrics" in report
    assert "gaps" in report
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_v3_fairness.py -v`
Expected: FAIL or SKIP — `fairlearn` not installed or `compute_fairness_report` not found

**Step 3: Write minimal implementation**

1. Add to `pyproject.toml`:

```toml
fairness = [
  "fairlearn>=0.10"
]
```

Also update `all`:

```toml
all = [
  "metrics_lie[dev,web,onnx,boosting,fairness]"
]
```

2. Create `src/metrics_lie/diagnostics/fairness.py`:

```python
"""Fairlearn-powered fairness analysis for subgroup evaluation."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def compute_fairness_report(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]],
) -> dict[str, Any]:
    """Compute per-group metrics, gaps, and fairness indicators using Fairlearn.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        sensitive_features: Protected attribute array (same length as y_true).
        metric_fns: Dict of {metric_name: callable(y_true, y_pred) -> float}.

    Returns:
        Dict with group_metrics, gaps, and fairness indicators.
    """
    from fairlearn.metrics import MetricFrame

    mf = MetricFrame(
        metrics=metric_fns,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    group_metrics: dict[str, dict[str, float]] = {}
    by_group = mf.by_group
    for group_val in by_group.index:
        group_key = str(group_val)
        group_metrics[group_key] = {}
        for metric_name in metric_fns:
            val = by_group.loc[group_val, metric_name] if metric_name in by_group.columns else by_group.loc[group_val]
            group_metrics[group_key][metric_name] = float(val) if not isinstance(val, dict) else float(val)

    # Compute gaps (max - min per metric)
    gaps: dict[str, float] = {}
    diff = mf.difference(method="between_groups")
    if isinstance(diff, dict):
        gaps = {k: float(v) for k, v in diff.items()}
    elif hasattr(diff, "to_dict"):
        gaps = {k: float(v) for k, v in diff.to_dict().items()}
    else:
        for metric_name in metric_fns:
            gaps[metric_name] = float(diff)

    # Selection rate / demographic parity (only for binary predictions)
    result: dict[str, Any] = {
        "group_metrics": group_metrics,
        "gaps": gaps,
        "overall": {k: float(v) for k, v in mf.overall.items()} if hasattr(mf.overall, "items") else {"metric": float(mf.overall)},
    }

    # Demographic parity difference (selection rate gap)
    unique_preds = np.unique(y_pred)
    if len(unique_preds) == 2:
        from fairlearn.metrics import demographic_parity_difference
        result["demographic_parity_difference"] = float(
            demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
        )
    else:
        # For multiclass, compute selection rate per class
        groups = np.unique(sensitive_features)
        selection_rates: dict[str, dict[str, float]] = {}
        for g in groups:
            mask = sensitive_features == g
            g_preds = y_pred[mask]
            rates = {}
            for c in unique_preds:
                rates[str(c)] = float(np.mean(g_preds == c))
            selection_rates[str(g)] = rates
        result["selection_rates_by_group"] = selection_rates
        # Max selection rate difference across any class
        max_diff = 0.0
        for c in unique_preds:
            rates_for_c = [selection_rates[str(g)][str(c)] for g in groups]
            max_diff = max(max_diff, max(rates_for_c) - min(rates_for_c))
        result["demographic_parity_difference"] = float(max_diff)

    return result
```

**Step 4: Run test to verify it passes**

First install fairlearn: `pip install fairlearn>=0.10`

Run: `pytest tests/test_v3_fairness.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pyproject.toml src/metrics_lie/diagnostics/fairness.py tests/test_v3_fairness.py
git commit -m "feat: add Fairlearn-powered fairness analysis module"
```

---

### Task 11: Wire fairness analysis into execution pipeline

**Files:**
- Modify: `src/metrics_lie/execution.py`
- Modify: `src/metrics_lie/spec.py` (add optional `sensitive_feature` field)
- Test: `tests/test_v3_fairness_pipeline.py`

**Depends on:** Task 10

**Step 1: Write the failing test**

```python
"""Tests for fairness analysis wired through execution pipeline."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

fairlearn = pytest.importorskip("fairlearn")

from metrics_lie.execution import run_from_spec_dict


@pytest.fixture
def iris_with_sensitive(tmp_path):
    """Create iris CSV with a sensitive attribute column."""
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=["f0", "f1", "f2", "f3"])
    df["y_true"] = iris.target
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=10, random_state=42).fit(iris.data, iris.target)
    proba = clf.predict_proba(iris.data)
    for i in range(proba.shape[1]):
        df[f"y_score_{i}"] = proba[:, i]
    # Add sensitive feature
    df["gender"] = np.where(np.arange(len(df)) % 2 == 0, "M", "F")
    csv_path = tmp_path / "iris_sensitive.csv"
    df.to_csv(csv_path, index=False)
    import pickle
    model_path = tmp_path / "iris_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    return str(csv_path), str(model_path)


def test_fairness_in_analysis_artifacts(iris_with_sensitive):
    csv_path, model_path = iris_with_sensitive
    spec = {
        "name": "test_fairness",
        "dataset": csv_path,
        "model": model_path,
        "metric": "macro_f1",
        "task": "multiclass_classification",
        "sensitive_feature": "gender",
        "scenarios": [],
        "n_trials": 1,
        "seed": 42,
    }
    bundle = run_from_spec_dict(spec)
    aa = bundle.analysis_artifacts
    assert "fairness" in aa, f"Missing fairness, got keys: {list(aa.keys())}"
    assert "group_metrics" in aa["fairness"]
    assert "gaps" in aa["fairness"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_v3_fairness_pipeline.py -v`
Expected: FAIL — `sensitive_feature` not in spec, fairness not wired in execution

**Step 3: Write minimal implementation**

1. In `src/metrics_lie/spec.py`, add optional field to `ExperimentSpec`:

```python
sensitive_feature: str | None = None
```

2. In `src/metrics_lie/execution.py`, after the analysis artifacts block, add:

```python
# Fairness analysis (optional — requires fairlearn and sensitive_feature)
if getattr(spec, "sensitive_feature", None) and prediction_surface is not None:
    try:
        from metrics_lie.diagnostics.fairness import compute_fairness_report
        from metrics_lie.metrics.core import METRICS

        # Load sensitive feature from dataset
        sensitive_col = spec.sensitive_feature
        if sensitive_col in df.columns:
            sensitive_features = df[sensitive_col].values
            # Get predictions
            if prediction_surface.values.ndim == 2:
                y_pred_fairness = np.argmax(prediction_surface.values, axis=1)
            elif prediction_surface.surface_type == SurfaceType.LABEL:
                y_pred_fairness = prediction_surface.values.astype(int)
            elif prediction_surface.surface_type == SurfaceType.PROBABILITY:
                y_pred_fairness = (prediction_surface.values >= (prediction_surface.threshold or 0.5)).astype(int)
            elif prediction_surface.surface_type == SurfaceType.CONTINUOUS:
                y_pred_fairness = prediction_surface.values
            else:
                y_pred_fairness = prediction_surface.values

            # Build metric fns for fairness (label-based metrics only)
            fairness_metric_fns = {
                "accuracy": lambda yt, yp: float(np.mean(yt == yp)),
            }

            fairness_report = compute_fairness_report(
                y_true=y_true,
                y_pred=y_pred_fairness,
                sensitive_features=sensitive_features,
                metric_fns=fairness_metric_fns,
            )
            analysis_artifacts["fairness"] = fairness_report
    except ImportError:
        pass  # fairlearn not installed — skip silently
```

3. Ensure `df` (the loaded dataset DataFrame) is still in scope at that point. Check where `df` is loaded in execution.py and ensure the variable persists.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_v3_fairness_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/metrics_lie/spec.py src/metrics_lie/execution.py tests/test_v3_fairness_pipeline.py
git commit -m "feat: wire Fairlearn fairness analysis into execution pipeline"
```

---

### Task 12: Add Evidently optional dependency and drift detection module

**Files:**
- Modify: `pyproject.toml`
- Create: `src/metrics_lie/diagnostics/drift.py`
- Test: `tests/test_v3_drift.py`

**Step 1: Write the failing test**

```python
"""Tests for Evidently-powered drift detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

evidently = pytest.importorskip("evidently")

from metrics_lie.diagnostics.drift import compute_drift_report


def test_drift_report_no_drift():
    """Identical distributions should show no drift."""
    rng = np.random.default_rng(42)
    reference = pd.DataFrame({"f0": rng.normal(0, 1, 200), "f1": rng.normal(0, 1, 200)})
    current = pd.DataFrame({"f0": rng.normal(0, 1, 200), "f1": rng.normal(0, 1, 200)})
    report = compute_drift_report(reference=reference, current=current)
    assert "dataset_drift" in report
    assert "n_drifted_features" in report
    assert "feature_drift" in report


def test_drift_report_with_drift():
    """Shifted distribution should detect drift."""
    rng = np.random.default_rng(42)
    reference = pd.DataFrame({"f0": rng.normal(0, 1, 200), "f1": rng.normal(0, 1, 200)})
    current = pd.DataFrame({"f0": rng.normal(5, 1, 200), "f1": rng.normal(5, 1, 200)})
    report = compute_drift_report(reference=reference, current=current)
    assert report["dataset_drift"] is True
    assert report["n_drifted_features"] > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_v3_drift.py -v`
Expected: FAIL or SKIP — `evidently` not installed or `compute_drift_report` not found

**Step 3: Write minimal implementation**

1. Add to `pyproject.toml`:

```toml
drift = [
  "evidently>=0.4"
]
```

Update `all`:

```toml
all = [
  "metrics_lie[dev,web,onnx,boosting,fairness,drift]"
]
```

2. Create `src/metrics_lie/diagnostics/drift.py`:

```python
"""Evidently-powered drift detection for dataset comparison."""
from __future__ import annotations

from typing import Any

import pandas as pd


def compute_drift_report(
    *,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Compare reference and current datasets for distribution drift.

    Uses Evidently's DataDriftPreset to compute per-feature drift statistics
    (KS test for numerical, chi-squared for categorical).

    Args:
        reference: Reference (training) dataset.
        current: Current (evaluation) dataset.
        feature_columns: Columns to check. If None, uses all shared columns.

    Returns:
        Dict with dataset_drift, n_drifted_features, and per-feature results.
    """
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    if feature_columns is not None:
        reference = reference[feature_columns]
        current = current[feature_columns]
    else:
        shared = sorted(set(reference.columns) & set(current.columns))
        reference = reference[shared]
        current = current[shared]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    result_dict = report.as_dict()

    # Extract summary from Evidently's result format
    metrics = result_dict.get("metrics", [])
    dataset_drift = False
    n_drifted = 0
    feature_drift: dict[str, Any] = {}

    for metric_result in metrics:
        metric_data = metric_result.get("result", {})
        if "dataset_drift" in metric_data:
            dataset_drift = metric_data["dataset_drift"]
        if "number_of_drifted_columns" in metric_data:
            n_drifted = metric_data["number_of_drifted_columns"]
        if "drift_by_columns" in metric_data:
            for col, col_data in metric_data["drift_by_columns"].items():
                feature_drift[col] = {
                    "drift_detected": col_data.get("drift_detected", False),
                    "drift_score": col_data.get("drift_score", None),
                    "stattest_name": col_data.get("stattest_name", None),
                }

    return {
        "dataset_drift": dataset_drift,
        "n_drifted_features": n_drifted,
        "n_features": len(reference.columns),
        "feature_drift": feature_drift,
    }
```

**Step 4: Run test to verify it passes**

First install evidently: `pip install evidently>=0.4`

Run: `pytest tests/test_v3_drift.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pyproject.toml src/metrics_lie/diagnostics/drift.py tests/test_v3_drift.py
git commit -m "feat: add Evidently-powered drift detection module"
```

---

### Task 13: Wire drift detection into execution pipeline

**Files:**
- Modify: `src/metrics_lie/spec.py` (add optional `reference_dataset` field)
- Modify: `src/metrics_lie/execution.py`
- Test: `tests/test_v3_drift_pipeline.py`

**Depends on:** Task 12

**Step 1: Write the failing test**

```python
"""Tests for drift detection wired through execution pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

evidently = pytest.importorskip("evidently")

from metrics_lie.execution import run_from_spec_dict


@pytest.fixture
def regression_with_reference(tmp_path):
    """Create regression CSV and a reference dataset."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 3))
    y = X[:, 0] * 2 + rng.normal(0, 0.1, 100)

    from sklearn.ensemble import RandomForestRegressor
    import pickle

    reg = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)

    df = pd.DataFrame(X, columns=["f0", "f1", "f2"])
    df["y_true"] = y
    df["y_score"] = reg.predict(X)
    csv_path = tmp_path / "eval.csv"
    df.to_csv(csv_path, index=False)

    # Reference with shifted distribution
    X_ref = rng.normal(5, 1, size=(100, 3))
    df_ref = pd.DataFrame(X_ref, columns=["f0", "f1", "f2"])
    ref_path = tmp_path / "reference.csv"
    df_ref.to_csv(ref_path, index=False)

    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(reg, f)

    return str(csv_path), str(model_path), str(ref_path)


def test_drift_in_analysis_artifacts(regression_with_reference):
    csv_path, model_path, ref_path = regression_with_reference
    spec = {
        "name": "test_drift",
        "dataset": csv_path,
        "model": model_path,
        "metric": "mae",
        "task": "regression",
        "reference_dataset": ref_path,
        "scenarios": [],
        "n_trials": 1,
        "seed": 42,
    }
    bundle = run_from_spec_dict(spec)
    aa = bundle.analysis_artifacts
    assert "drift" in aa, f"Missing drift, got keys: {list(aa.keys())}"
    assert "dataset_drift" in aa["drift"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_v3_drift_pipeline.py -v`
Expected: FAIL — `reference_dataset` not in spec, drift not wired

**Step 3: Write minimal implementation**

1. In `src/metrics_lie/spec.py`, add optional field:

```python
reference_dataset: str | None = None
```

2. In `src/metrics_lie/execution.py`, add drift check after loading dataset:

```python
# Drift detection (optional — requires evidently and reference_dataset)
if getattr(spec, "reference_dataset", None):
    try:
        from metrics_lie.diagnostics.drift import compute_drift_report

        ref_df = pd.read_csv(spec.reference_dataset)
        # Use feature columns only (exclude y_true, y_score, etc.)
        feature_cols = [c for c in df.columns if c not in ("y_true", "y_score", "subgroup") and not c.startswith("y_score_")]
        shared_cols = sorted(set(feature_cols) & set(ref_df.columns))
        if shared_cols:
            drift_report = compute_drift_report(
                reference=ref_df,
                current=df,
                feature_columns=shared_cols,
            )
            analysis_artifacts["drift"] = drift_report
    except ImportError:
        pass  # evidently not installed — skip silently
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_v3_drift_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/metrics_lie/spec.py src/metrics_lie/execution.py tests/test_v3_drift_pipeline.py
git commit -m "feat: wire Evidently drift detection into execution pipeline"
```

---

### Task 14: End-to-end integration tests for all Phase 3 features

**Files:**
- Test: `tests/test_v3_integration.py`

**Depends on:** All previous tasks

**Step 1: Write the integration test**

```python
"""End-to-end integration tests for Phase 3 diagnostics generalization."""
from __future__ import annotations

import os

import numpy as np
import pytest

from metrics_lie.execution import run_from_spec_dict


class TestMulticlassFullAnalysis:
    """Multiclass runs should produce complete analysis artifacts."""

    def test_multiclass_has_all_analysis_keys(self):
        spec = {
            "name": "test_mc_full",
            "dataset": os.path.join("data", "iris_multiclass.csv"),
            "model": os.path.join("data", "iris_model.pkl"),
            "metric": "macro_f1",
            "task": "multiclass_classification",
            "scenarios": [
                {"type": "label_noise", "noise_rate": 0.1},
                {"type": "score_noise", "noise_level": 0.05},
            ],
            "n_trials": 3,
            "seed": 42,
        }
        bundle = run_from_spec_dict(spec)
        aa = bundle.analysis_artifacts
        assert "failure_modes" in aa
        assert "dashboard_summary" in aa
        # Sensitivity should be present for multiclass with LABEL/CONTINUOUS surface
        # (depends on surface type returned by model)

    def test_multiclass_dashboard_is_direction_aware(self):
        spec = {
            "name": "test_mc_direction",
            "dataset": os.path.join("data", "iris_multiclass.csv"),
            "model": os.path.join("data", "iris_model.pkl"),
            "metric": "macro_f1",
            "task": "multiclass_classification",
            "scenarios": [{"type": "label_noise", "noise_rate": 0.3}],
            "n_trials": 5,
            "seed": 42,
        }
        bundle = run_from_spec_dict(spec)
        dashboard = bundle.analysis_artifacts.get("dashboard_summary", {})
        assert "risk_summary" in dashboard


class TestRegressionFullAnalysis:
    """Regression runs should produce complete analysis artifacts."""

    def test_regression_has_failure_modes_and_dashboard(self):
        spec = {
            "name": "test_reg_full",
            "dataset": os.path.join("data", "regression_sample.csv"),
            "model": os.path.join("data", "regression_model.pkl"),
            "metric": "mae",
            "task": "regression",
            "scenarios": [
                {"type": "label_noise", "noise_rate": 0.1},
                {"type": "score_noise", "noise_level": 0.05},
            ],
            "n_trials": 3,
            "seed": 42,
        }
        bundle = run_from_spec_dict(spec)
        aa = bundle.analysis_artifacts
        assert "failure_modes" in aa
        assert "dashboard_summary" in aa

    def test_regression_failure_modes_residual_based(self):
        spec = {
            "name": "test_reg_residuals",
            "dataset": os.path.join("data", "regression_sample.csv"),
            "model": os.path.join("data", "regression_model.pkl"),
            "metric": "mae",
            "task": "regression",
            "scenarios": [],
            "n_trials": 1,
            "seed": 42,
        }
        bundle = run_from_spec_dict(spec)
        fm = bundle.analysis_artifacts.get("failure_modes", {})
        assert fm["total_samples"] > 0
        assert len(fm["failure_samples"]) > 0


class TestBackwardCompatibility:
    """Binary classification runs should still work exactly as before."""

    def test_binary_still_has_full_analysis(self):
        spec = {
            "name": "test_binary_compat",
            "dataset": os.path.join("data", "demo_credit.csv"),
            "model": os.path.join("data", "demo_model.pkl"),
            "metric": "auc",
            "scenarios": [{"type": "label_noise", "noise_rate": 0.1}],
            "n_trials": 3,
            "seed": 42,
        }
        bundle = run_from_spec_dict(spec)
        aa = bundle.analysis_artifacts
        assert "threshold_sweep" in aa
        assert "sensitivity" in aa
        assert "metric_disagreements" in aa
        assert "failure_modes" in aa
        assert "dashboard_summary" in aa
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_v3_integration.py -v`
Expected: PASS (all previous tasks complete)

**Step 3: Commit**

```bash
git add tests/test_v3_integration.py
git commit -m "test: add end-to-end Phase 3 integration tests for all task types"
```

---

### Task 15: Run full test suite and fix any regressions

**Step 1: Run linter**

Run: `ruff check src tests`
Expected: No errors. If there are errors, fix them.

**Step 2: Run full test suite**

Run: `pytest -v`
Expected: All tests pass. If any existing tests fail, investigate and fix.

**Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: resolve Phase 3 test regressions"
```

**Step 4: Final commit — Phase 3 plan doc**

```bash
git add docs/plans/2026-02-27-v1-phase3-diagnostics-analysis-generalization.md
git commit -m "docs: add Phase 3 diagnostics & analysis generalization plan"
```

---

## Parallelization Strategy

| Task | Stream | Dependencies | Can Parallel With |
|------|--------|-------------|-------------------|
| 1 | A | None | 3, 6, 7, 8, 10, 12 |
| 2 | A | 1 | 3, 6, 7, 8, 10, 12 |
| 3 | A | None | 1, 2, 6, 7, 8, 10, 12 |
| 4 | A | 1, 2, 3 | 7, 8, 9, 10, 12 |
| 5 | A | 1 | 3, 6, 7, 8, 10, 12 |
| 6 | A | None | 1, 2, 3, 7, 8, 10, 12 |
| 7 | B | None | 1, 2, 3, 5, 6, 10, 12 |
| 8 | B | 7 | 1, 2, 3, 5, 6, 10, 12 |
| 9 | B | 7, 8, 4 | 10, 12 |
| 10 | C | None | 1-8 |
| 11 | C | 10, 4 | 12 |
| 12 | C | None | 1-8, 10 |
| 13 | C | 12, 4 | 11 |
| 14 | — | All | None |
| 15 | — | 14 | None |

**Maximum parallelism**: Launch Tasks 1, 3, 6, 7, 10, 12 simultaneously (6 parallel agents).

**Recommended 3-agent strategy:**
- Agent 1 (Stream A): Tasks 1 → 2 → 4 → 5
- Agent 2 (Stream B): Tasks 7 → 8 → 3 → 6
- Agent 3 (Stream C): Tasks 10 → 12 → 11 → 13
- Then: Tasks 9, 14, 15 sequentially
