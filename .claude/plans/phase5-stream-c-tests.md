# Plan: Phase 5 Stream C — Backend Tests (Multi-Task Web API)

> **Execution model:** This plan was written by Opus for execution by Sonnet.
> Run: `claude --model claude-sonnet-4-6` then say "Execute plan .claude/plans/phase5-stream-c-tests.md"

## Goal

Add comprehensive backend tests verifying all Phase 5 multi-task changes: contract fields, model validation, bundle transform, presets filtering, and engine bridge routing. All tests must pass with `pytest tests/ -x -q`.

## Context

- Stream A added: task_type to contracts, multi-format model validation, task-aware bundle transform, preset filtering, engine bridge multi-task routing
- Stream B added: frontend TypeScript types and components (no backend test coverage needed)
- Existing test `test_milestone4_bundle_transform.py` tests binary-only transform — we extend it
- Test naming: `test_phase5_*.py` in `tests/` directory
- All backend web tests use `pytest.importorskip("fastapi")` guard
- Tests should NOT require a running server — test functions directly

## Prerequisites

- [ ] Stream A and Stream B branches are merged to main (or current branch has all Stream A changes)
- [ ] Read each file referenced before writing tests

## Tasks

### Task C1: Test contracts — new fields have defaults

**File:** Create `tests/test_phase5_contracts.py`

**Code:**

```python
"""Tests for Phase 5 contract additions — task_type and task-specific fields."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from web.backend.app.contracts import (
    ExperimentSummary,
    ModelUploadResponse,
    ResultSummary,
    SupportedFormat,
)


def test_experiment_summary_task_type_default():
    """ExperimentSummary defaults to binary_classification."""
    summary = ExperimentSummary(
        id="exp1",
        name="test",
        metric_id="auc",
        stress_suite_id="default",
        status="created",
        created_at="2026-01-01T00:00:00+00:00",
    )
    assert summary.task_type == "binary_classification"


def test_experiment_summary_task_type_explicit():
    summary = ExperimentSummary(
        id="exp1",
        name="test",
        metric_id="mae",
        stress_suite_id="default",
        task_type="regression",
        status="created",
        created_at="2026-01-01T00:00:00+00:00",
    )
    assert summary.task_type == "regression"


def test_result_summary_task_specific_fields_default_none():
    """All task-specific fields default to None."""
    result = ResultSummary(
        experiment_id="exp1",
        run_id="run1",
        headline_score=0.9,
        generated_at="2026-01-01T00:00:00+00:00",
    )
    assert result.task_type == "binary_classification"
    assert result.confusion_matrix is None
    assert result.class_names is None
    assert result.per_class_metrics is None
    assert result.residual_stats is None
    assert result.ranking_metrics is None


def test_result_summary_confusion_matrix_round_trip():
    """Confusion matrix round-trips through the contract."""
    cm = [[50, 10], [5, 35]]
    result = ResultSummary(
        experiment_id="exp1",
        run_id="run1",
        headline_score=0.85,
        task_type="binary_classification",
        confusion_matrix=cm,
        class_names=["0", "1"],
        generated_at="2026-01-01T00:00:00+00:00",
    )
    assert result.confusion_matrix == cm
    assert result.class_names == ["0", "1"]


def test_result_summary_per_class_metrics():
    result = ResultSummary(
        experiment_id="exp1",
        run_id="run1",
        headline_score=0.8,
        task_type="multiclass_classification",
        per_class_metrics={
            "0": {"precision": 0.9, "recall": 0.85, "f1": 0.87, "support": 50},
            "1": {"precision": 0.8, "recall": 0.75, "f1": 0.77, "support": 40},
            "2": {"precision": 0.7, "recall": 0.65, "f1": 0.67, "support": 30},
        },
        generated_at="2026-01-01T00:00:00+00:00",
    )
    assert len(result.per_class_metrics) == 3
    assert result.per_class_metrics["0"]["precision"] == 0.9


def test_result_summary_residual_stats():
    result = ResultSummary(
        experiment_id="exp1",
        run_id="run1",
        headline_score=0.5,
        task_type="regression",
        residual_stats={
            "mean": 0.01, "std": 0.5, "min": -2.0, "max": 1.8,
            "median": 0.0, "mae": 0.3, "rmse": 0.5,
        },
        generated_at="2026-01-01T00:00:00+00:00",
    )
    assert result.residual_stats["mae"] == pytest.approx(0.3)
    assert result.residual_stats["rmse"] == pytest.approx(0.5)


def test_model_upload_response_task_type():
    resp = ModelUploadResponse(
        model_id="abc123",
        original_filename="model.pkl",
        model_class="sklearn.linear_model.LogisticRegression",
        task_type="multiclass_classification",
        n_classes=5,
        capabilities={"predict": True, "predict_proba": True},
        file_size_bytes=1024,
    )
    assert resp.task_type == "multiclass_classification"
    assert resp.n_classes == 5


def test_supported_format_contract():
    fmt = SupportedFormat(
        format_id="pickle",
        name="sklearn Pickle",
        extensions=[".pkl", ".joblib"],
        task_types=["binary_classification", "regression"],
    )
    assert fmt.format_id == "pickle"
    assert ".pkl" in fmt.extensions
```

**Verification:** `pytest tests/test_phase5_contracts.py -v`
Expected: All 8 tests pass.

### Task C2: Test model validation — multi-task, multi-format

**File:** Create `tests/test_phase5_model_validation.py`

**Code:**

```python
"""Tests for Phase 5 multi-task model validation."""
from __future__ import annotations

import pickle

import numpy as np
import pytest

pytest.importorskip("fastapi")

from web.backend.app.model_validation import (
    ACCEPTED_EXTENSIONS,
    ModelValidationResult,
    validate_model,
    validate_sklearn_pickle,
)


class _FakeClassifier:
    """Fake binary classifier with predict_proba."""
    n_features_in_ = 3

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)) * 0.6, np.ones(len(X)) * 0.4])


class _FakeMulticlassClassifier:
    """Fake multiclass classifier with 4 classes."""
    n_features_in_ = 3

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([
            np.ones(n) * 0.7,
            np.ones(n) * 0.1,
            np.ones(n) * 0.1,
            np.ones(n) * 0.1,
        ])


class _FakeRegressor:
    """Fake regressor with predict only."""
    n_features_in_ = 3

    def predict(self, X):
        return np.zeros(len(X))


class _FakeBadModel:
    """Model with no predict."""
    pass


def _pickle_model(model) -> bytes:
    return pickle.dumps(model)


# --- Extension acceptance ---

def test_accepted_extensions_include_all_formats():
    assert ".pkl" in ACCEPTED_EXTENSIONS
    assert ".onnx" in ACCEPTED_EXTENSIONS
    assert ".ubj" in ACCEPTED_EXTENSIONS
    assert ".lgb" in ACCEPTED_EXTENSIONS
    assert ".cbm" in ACCEPTED_EXTENSIONS


def test_unsupported_extension_rejected():
    result = validate_model(b"data", ".txt", "binary_classification")
    assert not result.valid
    assert "Unsupported" in result.error


# --- Pickle binary ---

def test_binary_classifier_valid():
    raw = _pickle_model(_FakeClassifier())
    result = validate_model(raw, ".pkl", "binary_classification")
    assert result.valid
    assert result.task_type == "binary_classification"
    assert result.n_classes == 2


def test_multiclass_model_fails_as_binary():
    raw = _pickle_model(_FakeMulticlassClassifier())
    result = validate_model(raw, ".pkl", "binary_classification")
    assert not result.valid
    assert "2 classes" in result.error


# --- Pickle multiclass ---

def test_multiclass_classifier_valid():
    raw = _pickle_model(_FakeMulticlassClassifier())
    result = validate_model(raw, ".pkl", "multiclass_classification")
    assert result.valid
    assert result.task_type == "multiclass_classification"
    assert result.n_classes == 4


def test_binary_model_fails_as_multiclass():
    raw = _pickle_model(_FakeClassifier())
    result = validate_model(raw, ".pkl", "multiclass_classification")
    assert not result.valid
    assert "3+ classes" in result.error


# --- Pickle regression ---

def test_regression_model_valid():
    raw = _pickle_model(_FakeRegressor())
    result = validate_model(raw, ".pkl", "regression")
    assert result.valid
    assert result.task_type == "regression"
    assert result.n_classes is None


def test_regression_doesnt_require_proba():
    """Regression models don't need predict_proba."""
    raw = _pickle_model(_FakeRegressor())
    result = validate_model(raw, ".pkl", "regression")
    assert result.valid
    # Regressor has predict but not predict_proba
    assert result.capabilities["predict"] is True


# --- Validation errors ---

def test_no_predict_rejected():
    raw = _pickle_model(_FakeBadModel())
    result = validate_model(raw, ".pkl", "binary_classification")
    assert not result.valid
    assert "predict()" in result.error


def test_invalid_pickle_bytes():
    result = validate_model(b"not a pickle", ".pkl", "binary_classification")
    assert not result.valid
    assert "Invalid pickle" in result.error


# --- Boosting format passthrough ---

def test_boosting_format_accepted():
    """Boosting formats are accepted if non-empty (validated at inference time)."""
    result = validate_model(b"model data", ".ubj", "binary_classification")
    assert result.valid
    assert result.model_class == "xgboost_model"


def test_lightgbm_format_accepted():
    result = validate_model(b"model data", ".lgb", "regression")
    assert result.valid
    assert result.task_type == "regression"


# --- Backward compatibility ---

def test_validate_sklearn_pickle_alias():
    """Legacy validate_sklearn_pickle still works."""
    raw = _pickle_model(_FakeClassifier())
    result = validate_sklearn_pickle(raw)
    assert result.valid
    assert result.task_type == "binary_classification"
```

**Verification:** `pytest tests/test_phase5_model_validation.py -v`
Expected: All 14 tests pass.

### Task C3: Test bundle transform — multi-task extraction

**File:** Create `tests/test_phase5_bundle_transform.py`

**Code:**

```python
"""Tests for Phase 5 task-aware bundle transform."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from metrics_lie.schema import MetricSummary, ResultBundle, ScenarioResult

from web.backend.app.bundle_transform import bundle_to_result_summary


def _metric(mean: float = 0.85) -> MetricSummary:
    return MetricSummary(mean=mean, std=0.02, q05=0.81, q50=0.85, q95=0.89, n=200)


def _bundle(
    *,
    task_type: str = "binary_classification",
    baseline_mean: float = 0.90,
    scenarios: list[ScenarioResult] | None = None,
    baseline_diagnostics: dict | None = None,
    task_specific: dict | None = None,
    analysis_artifacts: dict | None = None,
) -> ResultBundle:
    if scenarios is None:
        scenarios = [
            ScenarioResult(scenario_id="label_noise", params={"p": 0.1}, metric=_metric(0.85)),
        ]
    notes: dict = {"phase": "test"}
    if baseline_diagnostics is not None:
        notes["baseline_diagnostics"] = baseline_diagnostics
    if task_specific is not None:
        notes["task_specific"] = task_specific
    return ResultBundle(
        run_id="TESTRUN01",
        experiment_name="test",
        metric_name="auc",
        task_type=task_type,
        baseline=_metric(baseline_mean),
        scenarios=scenarios,
        notes=notes,
        analysis_artifacts=analysis_artifacts or {},
        applicable_metrics=["auc"],
        created_at="2026-01-15T12:00:00+00:00",
    )


# --- task_type passthrough ---

def test_task_type_binary():
    result = bundle_to_result_summary(_bundle(task_type="binary_classification"), "e", "r")
    assert result.task_type == "binary_classification"


def test_task_type_regression():
    result = bundle_to_result_summary(_bundle(task_type="regression"), "e", "r")
    assert result.task_type == "regression"


def test_task_type_multiclass():
    result = bundle_to_result_summary(_bundle(task_type="multiclass_classification"), "e", "r")
    assert result.task_type == "multiclass_classification"


# --- confusion matrix ---

def test_confusion_matrix_extracted():
    ts = {"confusion_matrix": [[45, 5], [10, 40]], "class_names": ["0", "1"]}
    result = bundle_to_result_summary(_bundle(task_specific=ts), "e", "r")
    assert result.confusion_matrix == [[45, 5], [10, 40]]
    assert result.class_names == ["0", "1"]


def test_confusion_matrix_none_when_absent():
    result = bundle_to_result_summary(_bundle(), "e", "r")
    assert result.confusion_matrix is None


# --- per-class metrics ---

def test_per_class_metrics_extracted():
    ts = {
        "per_class_metrics": {
            "0": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 50},
            "1": {"precision": 0.7, "recall": 0.6, "f1": 0.65, "support": 30},
            "2": {"precision": 0.8, "recall": 0.9, "f1": 0.85, "support": 40},
        }
    }
    result = bundle_to_result_summary(
        _bundle(task_type="multiclass_classification", task_specific=ts), "e", "r"
    )
    assert result.per_class_metrics is not None
    assert len(result.per_class_metrics) == 3
    assert result.per_class_metrics["0"]["precision"] == 0.9


# --- residual stats ---

def test_residual_stats_extracted():
    ts = {
        "residual_stats": {
            "mean": 0.01, "std": 0.5, "min": -2.0, "max": 1.8,
            "median": 0.0, "mae": 0.3, "rmse": 0.5,
        }
    }
    result = bundle_to_result_summary(
        _bundle(task_type="regression", task_specific=ts), "e", "r"
    )
    assert result.residual_stats is not None
    assert result.residual_stats["mae"] == pytest.approx(0.3)
    assert result.residual_stats["rmse"] == pytest.approx(0.5)


def test_residual_stats_none_for_classification():
    result = bundle_to_result_summary(_bundle(task_type="binary_classification"), "e", "r")
    assert result.residual_stats is None


# --- severity classification ---

def test_severity_high_for_large_delta():
    """Delta >= 0.1 should be 'high'."""
    bundle = _bundle(
        baseline_mean=0.90,
        scenarios=[ScenarioResult(scenario_id="test", params={}, metric=_metric(0.78))],
    )
    result = bundle_to_result_summary(bundle, "e", "r")
    assert result.scenario_results[0].severity == "high"


def test_severity_med_for_moderate_delta():
    """Delta 0.05-0.1 should be 'med'."""
    bundle = _bundle(
        baseline_mean=0.90,
        scenarios=[ScenarioResult(scenario_id="test", params={}, metric=_metric(0.84))],
    )
    result = bundle_to_result_summary(bundle, "e", "r")
    assert result.scenario_results[0].severity == "med"


def test_severity_low_for_small_delta():
    """Delta 0.01-0.05 should be 'low'."""
    bundle = _bundle(
        baseline_mean=0.90,
        scenarios=[ScenarioResult(scenario_id="test", params={}, metric=_metric(0.88))],
    )
    result = bundle_to_result_summary(bundle, "e", "r")
    assert result.scenario_results[0].severity == "low"


def test_severity_none_for_tiny_delta():
    """Delta < 0.01 should be None."""
    bundle = _bundle(
        baseline_mean=0.90,
        scenarios=[ScenarioResult(scenario_id="test", params={}, metric=_metric(0.899))],
    )
    result = bundle_to_result_summary(bundle, "e", "r")
    assert result.scenario_results[0].severity is None


# --- component scores multi-task ---

def test_multiclass_calibration_components():
    diag = {"multiclass_brier": 0.15, "multiclass_ece": 0.08}
    result = bundle_to_result_summary(
        _bundle(task_type="multiclass_classification", baseline_diagnostics=diag), "e", "r"
    )
    names = [c.name for c in result.component_scores]
    assert "multiclass_brier" in names
    assert "multiclass_ece" in names


def test_regression_component_scores():
    ts = {"residual_stats": {"mae": 0.3, "rmse": 0.5, "mean": 0, "std": 0.5, "min": -1, "max": 1, "median": 0}}
    result = bundle_to_result_summary(
        _bundle(task_type="regression", task_specific=ts), "e", "r"
    )
    names = [c.name for c in result.component_scores]
    assert "mae" in names
    assert "rmse" in names


# --- flags multi-task ---

def test_high_ece_flag_still_works():
    result = bundle_to_result_summary(
        _bundle(baseline_diagnostics={"ece": 0.15}), "e", "r"
    )
    assert any(f.code == "high_ece" for f in result.flags)


def test_multiclass_ece_flag():
    result = bundle_to_result_summary(
        _bundle(
            task_type="multiclass_classification",
            baseline_diagnostics={"multiclass_ece": 0.15},
        ),
        "e", "r",
    )
    assert any(f.code == "high_multiclass_ece" for f in result.flags)


def test_residual_outlier_flag():
    ts = {"residual_stats": {"mean": 0, "std": 1.0, "min": -5.0, "max": 4.0, "median": 0, "mae": 0.8, "rmse": 1.0}}
    result = bundle_to_result_summary(
        _bundle(task_type="regression", task_specific=ts), "e", "r"
    )
    assert any(f.code == "high_residual_outliers" for f in result.flags)


def test_no_residual_flag_when_within_bounds():
    ts = {"residual_stats": {"mean": 0, "std": 1.0, "min": -2.0, "max": 2.0, "median": 0, "mae": 0.5, "rmse": 1.0}}
    result = bundle_to_result_summary(
        _bundle(task_type="regression", task_specific=ts), "e", "r"
    )
    assert not any(f.code == "high_residual_outliers" for f in result.flags)


# --- backward compat: existing test from milestone4 should still pass pattern ---

def test_existing_binary_transform_unchanged():
    """Binary bundles without task_specific still work (backward compat)."""
    bundle = _bundle(
        baseline_diagnostics={"brier": 0.12, "ece": 0.08},
    )
    result = bundle_to_result_summary(bundle, "exp_1", "run_1")
    assert result.headline_score == pytest.approx(0.90)
    assert result.task_type == "binary_classification"
    names = [c.name for c in result.component_scores]
    assert "brier_score" in names
    assert "ece_score" in names
    assert len(result.flags) == 0  # ECE 0.08 < 0.1
```

**Verification:** `pytest tests/test_phase5_bundle_transform.py -v`
Expected: All 20 tests pass.

### Task C4: Test preset filtering by task type

**File:** Create `tests/test_phase5_presets.py`

**Code:**

```python
"""Tests for Phase 5 preset task-type filtering."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from web.backend.app.storage import METRIC_PRESETS


def test_all_presets_have_task_types_field():
    """Every preset must have a task_types list."""
    for p in METRIC_PRESETS:
        assert "task_types" in p, f"Preset {p['id']} missing task_types"
        assert isinstance(p["task_types"], list)
        assert len(p["task_types"]) > 0


def test_binary_presets_include_auc():
    binary = [p for p in METRIC_PRESETS if "binary_classification" in p["task_types"]]
    ids = [p["id"] for p in binary]
    assert "auc" in ids
    assert "brier_score" in ids


def test_regression_presets_include_mae():
    regression = [p for p in METRIC_PRESETS if "regression" in p["task_types"]]
    ids = [p["id"] for p in regression]
    assert "mae" in ids
    assert "mse" in ids
    assert "rmse" in ids
    assert "r_squared" in ids


def test_multiclass_presets_include_weighted_f1():
    multiclass = [p for p in METRIC_PRESETS if "multiclass_classification" in p["task_types"]]
    ids = [p["id"] for p in multiclass]
    assert "weighted_f1" in ids
    assert "macro_f1" in ids
    assert "cohens_kappa" in ids


def test_auc_not_in_regression():
    regression = [p for p in METRIC_PRESETS if "regression" in p["task_types"]]
    ids = [p["id"] for p in regression]
    assert "auc" not in ids


def test_mae_not_in_binary():
    binary = [p for p in METRIC_PRESETS if "binary_classification" in p["task_types"]]
    ids = [p["id"] for p in binary]
    assert "mae" not in ids


def test_shared_metrics_appear_in_both_classification_types():
    """accuracy, f1, precision, recall should be in both binary and multiclass."""
    shared = ["accuracy", "f1", "precision", "recall"]
    binary = {p["id"] for p in METRIC_PRESETS if "binary_classification" in p["task_types"]}
    multi = {p["id"] for p in METRIC_PRESETS if "multiclass_classification" in p["task_types"]}
    for m in shared:
        assert m in binary, f"{m} missing from binary"
        assert m in multi, f"{m} missing from multiclass"


def test_filter_simulation():
    """Simulate the API filter: only return presets matching task_type."""
    task_type = "regression"
    filtered = [p for p in METRIC_PRESETS if task_type in p.get("task_types", [])]
    assert all("regression" in p["task_types"] for p in filtered)
    assert len(filtered) >= 4  # mae, mse, rmse, r_squared at minimum
```

**Verification:** `pytest tests/test_phase5_presets.py -v`
Expected: All 8 tests pass.

### Task C5: Test engine bridge task routing

**File:** Create `tests/test_phase5_engine_bridge.py`

**Code:**

```python
"""Tests for Phase 5 engine bridge multi-task routing."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from web.backend.app.engine_bridge import _get_default_scenarios


# --- scenario routing by task type ---

def test_binary_scenarios_include_class_imbalance():
    scenarios = _get_default_scenarios("default", "binary_classification")
    ids = [s["id"] for s in scenarios]
    assert "class_imbalance" in ids
    assert "label_noise" in ids
    assert "score_noise" in ids


def test_regression_scenarios_exclude_class_imbalance():
    scenarios = _get_default_scenarios("default", "regression")
    ids = [s["id"] for s in scenarios]
    assert "class_imbalance" not in ids
    assert "label_noise" in ids
    assert "score_noise" in ids


def test_ranking_scenarios_exclude_class_imbalance():
    scenarios = _get_default_scenarios("default", "ranking")
    ids = [s["id"] for s in scenarios]
    assert "class_imbalance" not in ids


def test_multiclass_scenarios_include_class_imbalance():
    scenarios = _get_default_scenarios("default", "multiclass_classification")
    ids = [s["id"] for s in scenarios]
    assert "class_imbalance" in ids


def test_default_task_type_is_binary():
    """When no task_type given, defaults to binary (has class_imbalance)."""
    scenarios = _get_default_scenarios("default")
    ids = [s["id"] for s in scenarios]
    assert "class_imbalance" in ids
```

**Verification:** `pytest tests/test_phase5_engine_bridge.py -v`
Expected: All 5 tests pass.

### Task C6: Run full test suite and fix issues

**Steps:**

1. Run full suite: `pytest tests/ -x -q --tb=short 2>&1 | tail -20`
2. If `test_milestone4_bundle_transform.py` fails because `ResultSummary` now requires `task_type` or the transform function signature changed, check:
   - The old tests should still pass because `bundle_to_result_summary` still accepts the same arguments
   - If the old bundle helper doesn't set `task_type` on `ResultBundle`, that's fine — the transform reads it via `getattr(bundle, "task_type", "binary_classification")`
3. Fix any import errors or assertion mismatches.
4. Run lint: `ruff check tests/test_phase5_*.py --fix`
5. Run full suite again: `pytest tests/ -x -q --tb=short`

**Acceptance:** All tests pass, including both old (`test_milestone4_*`) and new (`test_phase5_*`) tests.

### Task C7: Commit test files

**Steps:**

1. Run: `ruff check tests/test_phase5_*.py`
2. Run: `pytest tests/test_phase5_*.py -v` (just the new tests)
3. Run: `pytest tests/ -x -q --tb=short` (full suite)
4. Commit:

```bash
git add tests/test_phase5_contracts.py tests/test_phase5_model_validation.py tests/test_phase5_bundle_transform.py tests/test_phase5_presets.py tests/test_phase5_engine_bridge.py
git commit -m "test: Phase 5 Stream C — multi-task web backend tests

- Contract tests: task_type defaults, confusion_matrix, per_class_metrics, residual_stats
- Model validation tests: binary/multiclass/regression pickle, ONNX, boosting formats
- Bundle transform tests: task-specific extraction, severity classification, flags
- Preset tests: task-type filtering, metric coverage per task type
- Engine bridge tests: scenario routing by task type"
```

**Acceptance:** All 55+ tests pass, lint clean, single commit.

## Boundaries

**DO:**
- Follow steps exactly as written
- Test the *actual* functions from Stream A, not mocks
- Run the full test suite to catch regressions

**DO NOT:**
- Modify any source files (contracts.py, bundle_transform.py, etc.) — only create test files
- Add new dependencies
- Create integration tests that require a running server (use direct function calls)

## Escalation Triggers

Stop and flag for Opus review if:
- Stream A code changes aren't merged yet (imports will fail)
- `ResultBundle` doesn't have `task_type` field
- The `_get_default_scenarios` function signature doesn't accept `task_type`
- More than 3 test failures in the full suite

When escalating, write to `.claude/plans/phase5-stream-c-blockers.md`.

## Verification

After all tasks complete:
- [ ] `ruff check tests/test_phase5_*.py` passes
- [ ] `pytest tests/test_phase5_*.py -v` — all new tests pass
- [ ] `pytest tests/ -x -q` — full suite passes (no regressions)
- [ ] No source files modified — only test files created
- [ ] Single commit with all 5 test files
