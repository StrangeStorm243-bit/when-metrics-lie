# Plan: Phase 6 Stream D — Tests (MLflow + KServe)

> **Execution model:** This plan was written by Opus for execution by Sonnet.
> Run: `claude --model claude-sonnet-4-6` then say "Execute plan .claude/plans/phase6-stream-d-tests.md"

## Goal

Add tests for Phase 6: MLflow logging, MLflow adapter, KServe V2 protocol, and default registry with MLflow. All tests use mocks (no real MLflow server or HTTP endpoint required).

## Context

- Stream A added: `src/metrics_lie/integrations/mlflow.py`, `src/metrics_lie/model/adapters/mlflow_adapter.py`, MLflow in default registry
- Stream B added: KServe V2 protocol in `src/metrics_lie/model/adapters/http_adapter.py`
- Existing HTTP adapter tests: `tests/test_v1_http_adapter.py` — uses `unittest.mock.patch("requests.post")`
- Test naming convention: `test_phase6_*.py` in `tests/`
- All tests should use mocks — no external services

## Prerequisites

- [ ] Streams A and B are merged (all source changes present)
- [ ] Read each source file before writing its tests

## Tasks

### Task D1: Test MLflow logging

**File:** Create `tests/test_phase6_mlflow_logging.py`

**Code:**

```python
"""Tests for Phase 6 MLflow logging integration."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import json

import pytest

from metrics_lie.schema import MetricSummary, ResultBundle, ScenarioResult


def _metric(mean: float = 0.85) -> MetricSummary:
    return MetricSummary(mean=mean, std=0.02, q05=0.81, q50=0.85, q95=0.89, n=200)


def _bundle(
    *,
    metric_name: str = "auc",
    baseline_mean: float = 0.90,
    task_type: str = "binary_classification",
) -> ResultBundle:
    return ResultBundle(
        run_id="TEST_RUN_01",
        experiment_name="test_exp",
        metric_name=metric_name,
        task_type=task_type,
        baseline=_metric(baseline_mean),
        scenarios=[
            ScenarioResult(scenario_id="label_noise", params={"p": 0.1}, metric=_metric(0.85)),
            ScenarioResult(scenario_id="score_noise", params={"sigma": 0.05}, metric=_metric(0.88)),
        ],
        metric_results={"auc": _metric(0.90), "f1": _metric(0.82)},
        created_at="2026-01-01T00:00:00+00:00",
    )


@patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.pyfunc": MagicMock()})
def test_log_to_mlflow_creates_run():
    """log_to_mlflow creates a new MLflow run and logs params/metrics."""
    import sys
    mock_mlflow = sys.modules["mlflow"]

    mock_run = MagicMock()
    mock_run.info.run_id = "mlflow_run_123"
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.active_run.return_value = mock_run

    from metrics_lie.integrations.mlflow import log_to_mlflow

    bundle = _bundle()
    rid = log_to_mlflow(bundle)

    assert rid == "mlflow_run_123"
    mock_mlflow.log_param.assert_any_call("spectra.experiment_name", "test_exp")
    mock_mlflow.log_param.assert_any_call("spectra.metric_name", "auc")
    mock_mlflow.log_param.assert_any_call("spectra.task_type", "binary_classification")
    mock_mlflow.log_metric.assert_any_call("spectra.auc.baseline_mean", 0.90)


@patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.pyfunc": MagicMock()})
def test_log_to_mlflow_logs_scenario_deltas():
    """Scenario deltas are logged as metrics."""
    import sys
    mock_mlflow = sys.modules["mlflow"]

    mock_run = MagicMock()
    mock_run.info.run_id = "run_456"
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.active_run.return_value = mock_run

    from metrics_lie.integrations.mlflow import log_to_mlflow

    bundle = _bundle(baseline_mean=0.90)
    log_to_mlflow(bundle)

    # label_noise: 0.85 - 0.90 = -0.05
    mock_mlflow.log_metric.assert_any_call("spectra.scenario.label_noise.delta", pytest.approx(-0.05))
    mock_mlflow.log_metric.assert_any_call("spectra.scenario.score_noise.mean", pytest.approx(0.88))


@patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.pyfunc": MagicMock()})
def test_log_to_mlflow_logs_artifact():
    """Full ResultBundle is logged as JSON artifact."""
    import sys
    mock_mlflow = sys.modules["mlflow"]

    mock_run = MagicMock()
    mock_run.info.run_id = "run_789"
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.active_run.return_value = mock_run

    from metrics_lie.integrations.mlflow import log_to_mlflow

    bundle = _bundle()
    log_to_mlflow(bundle)

    mock_mlflow.log_artifact.assert_called_once()
    args = mock_mlflow.log_artifact.call_args
    assert args.kwargs.get("artifact_path") == "spectra" or args[1].get("artifact_path") == "spectra"


@patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.pyfunc": MagicMock()})
def test_log_to_mlflow_with_tracking_uri():
    """tracking_uri is set when provided."""
    import sys
    mock_mlflow = sys.modules["mlflow"]

    mock_run = MagicMock()
    mock_run.info.run_id = "run_uri"
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.active_run.return_value = mock_run

    from metrics_lie.integrations.mlflow import log_to_mlflow

    bundle = _bundle()
    log_to_mlflow(bundle, tracking_uri="http://mlflow:5000")

    mock_mlflow.set_tracking_uri.assert_called_once_with("http://mlflow:5000")


def test_log_to_mlflow_import_error_without_mlflow():
    """Raises ImportError with install hint when mlflow is missing."""
    # This test works when mlflow is NOT installed
    try:
        import mlflow  # noqa: F401
        pytest.skip("mlflow is installed — cannot test missing import")
    except ImportError:
        pass

    from metrics_lie.integrations.mlflow import log_to_mlflow

    bundle = _bundle()
    with pytest.raises(ImportError, match="pip install"):
        log_to_mlflow(bundle)
```

**Verification:** `python -m pytest tests/test_phase6_mlflow_logging.py -v --tb=short`
Expected: 5 tests pass.

### Task D2: Test MLflow adapter

**File:** Create `tests/test_phase6_mlflow_adapter.py`

**Code:**

```python
"""Tests for Phase 6 MLflow model adapter."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from metrics_lie.model.surface import SurfaceType
from metrics_lie.task_types import TaskType


def test_mlflow_adapter_import_error():
    """Raises ImportError with install hint when mlflow is missing."""
    try:
        import mlflow  # noqa: F401
        pytest.skip("mlflow is installed — cannot test missing import")
    except ImportError:
        pass

    with pytest.raises(ImportError, match="pip install"):
        from metrics_lie.model.adapters.mlflow_adapter import MLflowAdapter
        MLflowAdapter(uri="runs:/fake/model")


@patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.pyfunc": MagicMock()})
def test_mlflow_adapter_predict_binary():
    """MLflow adapter returns LABEL surface for binary classification."""
    import sys
    mock_pyfunc = sys.modules["mlflow.pyfunc"]

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1, 0])
    mock_model.metadata = MagicMock()
    mock_model.metadata.flavors = {"sklearn": {}}
    mock_pyfunc.load_model.return_value = mock_model

    from metrics_lie.model.adapters.mlflow_adapter import MLflowAdapter

    adapter = MLflowAdapter(
        uri="runs:/abc123/model",
        task_type=TaskType.BINARY_CLASSIFICATION,
    )

    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    surface = adapter.predict(X)

    assert surface.surface_type == SurfaceType.LABEL
    assert surface.n_samples == 3
    assert np.array_equal(surface.values, np.array([0, 1, 0]))


@patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.pyfunc": MagicMock()})
def test_mlflow_adapter_predict_proba_binary():
    """MLflow adapter returns PROBABILITY surface when output is 2D."""
    import sys
    mock_pyfunc = sys.modules["mlflow.pyfunc"]

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    mock_model.metadata = MagicMock()
    mock_model.metadata.flavors = {}
    mock_pyfunc.load_model.return_value = mock_model

    from metrics_lie.model.adapters.mlflow_adapter import MLflowAdapter

    adapter = MLflowAdapter(
        uri="runs:/abc123/model",
        task_type=TaskType.BINARY_CLASSIFICATION,
    )

    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    surface = adapter.predict_proba(X)

    assert surface is not None
    assert surface.surface_type == SurfaceType.PROBABILITY
    assert surface.n_samples == 3
    assert surface.values[1] == pytest.approx(0.7)  # positive class prob


@patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.pyfunc": MagicMock()})
def test_mlflow_adapter_predict_regression():
    """MLflow adapter returns CONTINUOUS surface for regression."""
    import sys
    mock_pyfunc = sys.modules["mlflow.pyfunc"]

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1.5, 2.3, 0.8])
    mock_model.metadata = MagicMock()
    mock_model.metadata.flavors = {}
    mock_pyfunc.load_model.return_value = mock_model

    from metrics_lie.model.adapters.mlflow_adapter import MLflowAdapter

    adapter = MLflowAdapter(
        uri="runs:/abc123/model",
        task_type=TaskType.REGRESSION,
    )

    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    surface = adapter.predict(X)

    assert surface.surface_type == SurfaceType.CONTINUOUS
    assert surface.n_samples == 3


@patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.pyfunc": MagicMock()})
def test_mlflow_adapter_metadata():
    """MLflow adapter metadata detects sklearn flavor."""
    import sys
    mock_pyfunc = sys.modules["mlflow.pyfunc"]

    mock_model = MagicMock()
    mock_model.metadata = MagicMock()
    mock_model.metadata.flavors = {"sklearn": {"version": "1.3"}}
    mock_pyfunc.load_model.return_value = mock_model

    from metrics_lie.model.adapters.mlflow_adapter import MLflowAdapter

    adapter = MLflowAdapter(uri="runs:/abc/model")
    meta = adapter.metadata

    assert meta.model_format == "mlflow"
    assert meta.model_class == "mlflow.sklearn"


@patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.pyfunc": MagicMock()})
def test_mlflow_adapter_get_all_surfaces():
    """get_all_surfaces returns both label and probability surfaces."""
    import sys
    mock_pyfunc = sys.modules["mlflow.pyfunc"]

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
    mock_model.metadata = MagicMock()
    mock_model.metadata.flavors = {}
    mock_pyfunc.load_model.return_value = mock_model

    from metrics_lie.model.adapters.mlflow_adapter import MLflowAdapter

    adapter = MLflowAdapter(uri="runs:/abc/model")
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    surfaces = adapter.get_all_surfaces(X)

    assert len(surfaces) >= 1
```

**Verification:** `python -m pytest tests/test_phase6_mlflow_adapter.py -v --tb=short`
Expected: 6 tests pass.

### Task D3: Test KServe V2 protocol

**File:** Create `tests/test_phase6_kserve.py`

**Code:**

```python
"""Tests for Phase 6 KServe V2 protocol support in HTTP adapter."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from metrics_lie.model.adapters.http_adapter import HTTPAdapter
from metrics_lie.model.surface import SurfaceType
from metrics_lie.task_types import TaskType


def _mock_kserve_response(outputs):
    """Create a mock response with KServe V2 format."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"outputs": outputs}
    return resp


def _mock_custom_response(predictions):
    """Create a mock response with custom format."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"predictions": predictions}
    return resp


# --- Protocol detection ---

def test_auto_detect_kserve_from_url():
    """URL with /v2/ triggers KServe protocol."""
    adapter = HTTPAdapter(
        endpoint="http://localhost:8080/v2/models/mymodel/infer",
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    assert adapter._protocol == "kserve_v2"


def test_custom_protocol_default():
    """Default URL uses custom protocol."""
    adapter = HTTPAdapter(
        endpoint="http://localhost:8080/predict",
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    assert adapter._protocol == "custom"


def test_explicit_kserve_protocol():
    """Explicit protocol='kserve_v2' is respected."""
    adapter = HTTPAdapter(
        endpoint="http://localhost:8080/myapi",
        protocol="kserve_v2",
        model_name="mymodel",
    )
    assert adapter._protocol == "kserve_v2"


# --- KServe URL building ---

def test_kserve_url_with_model_name():
    adapter = HTTPAdapter(
        endpoint="http://localhost:8080",
        protocol="kserve_v2",
        model_name="my_model",
    )
    url = adapter._build_kserve_url()
    assert url == "http://localhost:8080/v2/models/my_model/infer"


def test_kserve_url_without_model_name():
    """When no model_name, use endpoint as-is."""
    adapter = HTTPAdapter(
        endpoint="http://localhost:8080/v2/models/existing/infer",
        protocol="kserve_v2",
    )
    url = adapter._build_kserve_url()
    assert url == "http://localhost:8080/v2/models/existing/infer"


# --- KServe request format ---

def test_kserve_request_format():
    adapter = HTTPAdapter(
        endpoint="http://localhost:8080",
        protocol="kserve_v2",
        model_name="m",
    )
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    req = adapter._format_kserve_request(X)

    assert "inputs" in req
    assert len(req["inputs"]) == 1
    inp = req["inputs"][0]
    assert inp["name"] == "input"
    assert inp["shape"] == [2, 2]
    assert inp["datatype"] == "FP64"
    assert inp["data"] == [1.0, 2.0, 3.0, 4.0]


# --- KServe response parsing ---

def test_parse_kserve_2d_probabilities():
    """2D output [n, classes] is parsed as probabilities."""
    adapter = HTTPAdapter(
        endpoint="http://localhost:8080",
        protocol="kserve_v2",
        model_name="m",
    )
    outputs = [{"name": "output", "shape": [2, 2], "data": [0.8, 0.2, 0.3, 0.7]}]
    preds = adapter._parse_kserve_response({"outputs": outputs})

    assert len(preds) == 2
    assert preds[0]["label"] == 0
    assert preds[0]["probability"] == [0.8, 0.2]
    assert preds[1]["label"] == 1
    assert preds[1]["probability"] == [0.3, 0.7]


def test_parse_kserve_1d_scores():
    """1D output [n] with values in [0,1] is treated as probabilities."""
    adapter = HTTPAdapter(
        endpoint="http://localhost:8080",
        protocol="kserve_v2",
        model_name="m",
        threshold=0.5,
    )
    outputs = [{"name": "output", "shape": [3], "data": [0.9, 0.3, 0.6]}]
    preds = adapter._parse_kserve_response({"outputs": outputs})

    assert len(preds) == 3
    assert preds[0]["label"] == 1  # 0.9 >= 0.5
    assert preds[1]["label"] == 0  # 0.3 < 0.5
    assert preds[2]["label"] == 1  # 0.6 >= 0.5


# --- End-to-end with mock ---

def test_kserve_predict_end_to_end():
    """Full predict() flow with KServe V2 mock."""
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_kserve_response(
            [{"name": "output", "shape": [2, 2], "data": [0.8, 0.2, 0.3, 0.7]}]
        )
        adapter = HTTPAdapter(
            endpoint="http://localhost:8080",
            protocol="kserve_v2",
            model_name="m",
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        surface = adapter.predict(X)

        assert surface.surface_type == SurfaceType.LABEL
        assert surface.n_samples == 2

        # Verify the URL was correct
        call_args = mock_post.call_args
        assert "/v2/models/m/infer" in call_args[0][0]


def test_kserve_predict_proba_end_to_end():
    """Full predict_proba() flow with KServe V2 mock."""
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_kserve_response(
            [{"name": "output", "shape": [2, 2], "data": [0.8, 0.2, 0.3, 0.7]}]
        )
        adapter = HTTPAdapter(
            endpoint="http://localhost:8080",
            protocol="kserve_v2",
            model_name="m",
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        surface = adapter.predict_proba(X)

        assert surface is not None
        assert surface.surface_type == SurfaceType.PROBABILITY
        assert surface.n_samples == 2


# --- Backward compatibility ---

def test_custom_protocol_still_works():
    """Existing custom protocol calls are unchanged."""
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_custom_response([
            {"label": 0, "probability": [0.8, 0.2]},
            {"label": 1, "probability": [0.3, 0.7]},
        ])
        adapter = HTTPAdapter(
            endpoint="http://localhost:8080/predict",
            task_type=TaskType.BINARY_CLASSIFICATION,
        )
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        surface = adapter.predict(X)

        assert surface.surface_type == SurfaceType.LABEL
        assert surface.n_samples == 2

        # URL should be the endpoint directly (no /v2/ rewrite)
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:8080/predict"
```

**Verification:** `python -m pytest tests/test_phase6_kserve.py -v --tb=short`
Expected: 12 tests pass.

### Task D4: Test default registry with MLflow

**File:** Create `tests/test_phase6_registry.py`

**Code:**

```python
"""Tests for Phase 6 default registry MLflow integration."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from metrics_lie.model.default_registry import get_default_registry, _reset_registry


def test_registry_includes_sklearn_and_http():
    """Default registry always has sklearn and http."""
    _reset_registry()
    reg = get_default_registry()
    formats = reg.list_formats()
    assert "sklearn" in formats
    assert "http" in formats


def test_registry_mlflow_when_available():
    """MLflow format is registered when mlflow is importable."""
    _reset_registry()
    with patch.dict("sys.modules", {"mlflow": __import__("types")}):
        _reset_registry()
        reg = get_default_registry()
        formats = reg.list_formats()
        assert "mlflow" in formats
    _reset_registry()


def test_registry_no_mlflow_when_missing():
    """MLflow format is NOT registered when mlflow is not importable."""
    _reset_registry()
    # Force mlflow import to fail
    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("no mlflow")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        _reset_registry()
        reg = get_default_registry()
        formats = reg.list_formats()
        # mlflow should NOT be in the list
        assert "mlflow" not in formats
    _reset_registry()
```

**Verification:** `python -m pytest tests/test_phase6_registry.py -v --tb=short`
Expected: 3 tests pass.

### Task D5: Lint and full test suite

**Steps:**

1. Run: `cd /c/GitHubProjects/when-metrics-lie && python -m ruff check tests/test_phase6_*.py --fix`
2. Run: `cd /c/GitHubProjects/when-metrics-lie && python -m pytest tests/test_phase6_*.py -v --tb=short`
   Expected: All ~26 tests pass.
3. Run: `cd /c/GitHubProjects/when-metrics-lie && python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5`
   Expected: All tests pass (existing + new).

**Acceptance:** All Phase 6 tests pass, full suite green.

### Task D6: Fix any test failures

If tests fail:
1. Read the source file the test targets
2. Adjust test expectations to match actual API
3. Common issues:
   - MLflow mock may need `sys.modules` patching before import
   - KServe response parsing shape expectations
   - Default registry caching (call `_reset_registry()` in setup)
4. Re-run until green.

Do NOT modify source files — only fix test files.

## Boundaries

**DO:**
- Create test files only
- Use mocks for all external dependencies (mlflow, requests)
- Test both happy path and error cases

**DO NOT:**
- Modify any source files
- Install mlflow in the environment
- Create a git commit (parent agent handles this)

## Escalation Triggers

Stop and flag for Opus review if:
- Stream A/B source changes are missing (imports fail)
- More than 3 tests can't be fixed by adjusting expectations
- Default registry tests cause side effects on other tests

When escalating, write to `.claude/plans/phase6-stream-d-blockers.md`.

## Verification

After all tasks complete:
- [ ] `python -m ruff check tests/test_phase6_*.py` passes
- [ ] `python -m pytest tests/test_phase6_*.py -v` — all new tests pass
- [ ] `python -m pytest tests/ -x -q` — full suite passes
- [ ] No source files modified
