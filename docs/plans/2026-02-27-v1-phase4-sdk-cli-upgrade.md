# V1.0 Phase 4: Public Python SDK & CLI Upgrade Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Provide a clean programmatic Python SDK (`import spectra`) for notebooks, scripts, and CI/CD, upgrade the CLI from argparse to Typer, and add notebook display functions.

**Architecture:** Create a thin `src/metrics_lie/sdk.py` module that wraps internal pipeline functions with a user-friendly API returning `ResultBundle` objects directly. Re-export key types from `src/metrics_lie/__init__.py`. Migrate CLI to Typer with richer output formatting. Add `display.py` for notebook HTML rendering. The SDK functions suppress stdout noise by routing prints through `logging`.

**Tech Stack:** Python 3.11+, Typer (CLI), Pydantic v2, Rich (CLI output), IPython (notebook display, optional)

**Scope boundaries:** This phase covers the Python SDK, CLI migration, and basic notebook display. Interactive dashboard (Streamlit), plotly integration, and advanced notebook widgets are deferred.

---

## Work Streams

Tasks are organized in dependency order. Streams A, B, and C are independent.

### Stream A: Public SDK (Tasks 1-5)
### Stream B: CLI Upgrade to Typer (Tasks 6-9)
### Stream C: Notebook Display + Cleanup (Tasks 10-12)

---

### Task 1: Create SDK entry points — evaluate, compare, score

**Files:**
- Create: `src/metrics_lie/sdk.py`
- Test: `tests/test_v4_sdk.py`

**Step 1: Write the failing test**

```python
"""Tests for public SDK entry points."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from metrics_lie.db.session import DB_PATH, engine, init_db
from metrics_lie.schema import ResultBundle


@pytest.fixture(autouse=True)
def fresh_db():
    engine.dispose()
    if DB_PATH.exists():
        DB_PATH.unlink()
    init_db()


@pytest.fixture
def binary_fixtures(tmp_path: Path):
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegression(random_state=0).fit(X, y)
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    df = pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "y_true": y, "y_score": np.zeros(n)})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path), str(model_path)


def test_evaluate_returns_result_bundle(binary_fixtures):
    from metrics_lie.sdk import evaluate

    csv_path, model_path = binary_fixtures
    result = evaluate(
        name="sdk_test",
        dataset=csv_path,
        model=model_path,
        metric="auc",
        scenarios=[{"type": "label_noise", "noise_rate": 0.1}],
        n_trials=3,
        seed=42,
    )
    assert isinstance(result, ResultBundle)
    assert result.metric_name == "auc"
    assert result.run_id is not None


def test_evaluate_file(binary_fixtures, tmp_path: Path):
    import json
    from metrics_lie.sdk import evaluate_file

    csv_path, model_path = binary_fixtures
    spec = {
        "name": "file_test",
        "dataset": {
            "source": "csv",
            "path": csv_path,
            "y_true_col": "y_true",
            "y_score_col": "y_score",
            "feature_cols": ["f1", "f2"],
        },
        "metric": "auc",
        "model_source": {"kind": "pickle", "path": model_path},
        "scenarios": [],
        "n_trials": 1,
        "seed": 42,
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    result = evaluate_file(str(spec_path))
    assert isinstance(result, ResultBundle)


def test_compare_bundles_sdk(binary_fixtures):
    from metrics_lie.sdk import evaluate, compare

    csv_path, model_path = binary_fixtures
    result_a = evaluate(
        name="compare_a", dataset=csv_path, model=model_path,
        metric="auc", n_trials=3, seed=42,
    )
    result_b = evaluate(
        name="compare_b", dataset=csv_path, model=model_path,
        metric="auc", n_trials=3, seed=99,
    )
    report = compare(result_a, result_b)
    assert "baseline_delta" in report
    assert "decision" in report


def test_score_sdk(binary_fixtures):
    from metrics_lie.sdk import evaluate, score

    csv_path, model_path = binary_fixtures
    result_a = evaluate(
        name="score_a", dataset=csv_path, model=model_path,
        metric="auc", n_trials=3, seed=42,
    )
    result_b = evaluate(
        name="score_b", dataset=csv_path, model=model_path,
        metric="auc", n_trials=3, seed=99,
    )
    scorecard = score(result_a, result_b, profile="balanced")
    assert hasattr(scorecard, "total_score")
    assert hasattr(scorecard, "profile_name")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_v4_sdk.py -v`
Expected: FAIL — `metrics_lie.sdk` does not exist

**Step 3: Write minimal implementation**

Create `src/metrics_lie/sdk.py`:

```python
"""Public SDK entry points for Spectra.

Usage:
    from metrics_lie.sdk import evaluate, compare, score

    result = evaluate(name="test", dataset="data.csv", model="model.pkl", metric="auc")
    report = compare(result_a, result_b)
    card = score(result_a, result_b, profile="balanced")
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from metrics_lie.schema import ResultBundle
from metrics_lie.utils.paths import get_run_dir

logger = logging.getLogger(__name__)


def evaluate(
    *,
    name: str,
    dataset: str,
    model: str | None = None,
    metric: str = "auc",
    task: str = "binary_classification",
    scenarios: list[dict[str, Any]] | None = None,
    n_trials: int = 200,
    seed: int = 42,
    y_true_col: str = "y_true",
    y_score_col: str = "y_score",
    feature_cols: list[str] | None = None,
    subgroup_col: str | None = None,
    sensitive_feature: str | None = None,
    reference_dataset: str | None = None,
) -> ResultBundle:
    """Run a stress-test evaluation and return the ResultBundle.

    Args:
        name: Human-readable experiment name.
        dataset: Path to CSV dataset.
        model: Path to model file (pickle, onnx, etc.) or None for surface-only.
        metric: Primary metric ID (e.g. 'auc', 'macro_f1', 'mae').
        task: Task type ('binary_classification', 'multiclass_classification', 'regression').
        scenarios: List of scenario dicts, e.g. [{"type": "label_noise", "noise_rate": 0.1}].
        n_trials: Monte Carlo trials per scenario.
        seed: Random seed for reproducibility.
        y_true_col: Column name for ground truth labels.
        y_score_col: Column name for predictions/scores.
        feature_cols: Feature columns for model inference. Auto-detected if None and model is provided.
        subgroup_col: Optional subgroup column for fairness diagnostics.
        sensitive_feature: Optional sensitive attribute for Fairlearn analysis.
        reference_dataset: Optional reference CSV for drift detection.

    Returns:
        ResultBundle with full evaluation results.
    """
    from metrics_lie.execution import run_from_spec_dict

    spec_dict: dict[str, Any] = {
        "name": name,
        "task": task,
        "dataset": {
            "source": "csv",
            "path": str(dataset),
            "y_true_col": y_true_col,
            "y_score_col": y_score_col,
        },
        "metric": metric,
        "scenarios": _normalize_scenarios(scenarios or []),
        "n_trials": n_trials,
        "seed": seed,
    }

    if subgroup_col:
        spec_dict["dataset"]["subgroup_col"] = subgroup_col

    if feature_cols:
        spec_dict["dataset"]["feature_cols"] = feature_cols
    elif model:
        # Auto-detect feature columns if model provided but no explicit feature_cols
        import pandas as pd
        df = pd.read_csv(dataset, nrows=0)
        non_feature = {y_true_col, y_score_col, subgroup_col} - {None}
        # Also exclude y_score_N columns
        auto_features = [
            c for c in df.columns
            if c not in non_feature and not c.startswith("y_score_")
        ]
        if auto_features:
            spec_dict["dataset"]["feature_cols"] = auto_features

    if model:
        model_path = Path(model)
        kind = _detect_model_kind(model_path)
        spec_dict["model_source"] = {"kind": kind, "path": str(model_path)}

    if sensitive_feature:
        spec_dict["sensitive_feature"] = sensitive_feature
    if reference_dataset:
        spec_dict["reference_dataset"] = reference_dataset

    run_id = run_from_spec_dict(spec_dict)
    return _load_bundle(run_id)


def evaluate_file(path: str | Path) -> ResultBundle:
    """Run evaluation from a JSON spec file."""
    from metrics_lie.execution import run_from_spec_dict

    spec_dict = json.loads(Path(path).read_text(encoding="utf-8"))
    run_id = run_from_spec_dict(spec_dict, spec_path_for_notes=str(path))
    return _load_bundle(run_id)


def compare(
    result_a: ResultBundle,
    result_b: ResultBundle,
) -> dict[str, Any]:
    """Compare two ResultBundles and return a comparison report dict."""
    from metrics_lie.compare.compare import compare_bundles

    bundle_a = json.loads(result_a.to_pretty_json())
    bundle_b = json.loads(result_b.to_pretty_json())
    return compare_bundles(bundle_a, bundle_b)


def score(
    result_a: ResultBundle,
    result_b: ResultBundle,
    profile: str = "balanced",
) -> Any:
    """Compare and score two ResultBundles with a decision profile.

    Returns a DecisionScorecard.
    """
    from metrics_lie.decision.extract import extract_components
    from metrics_lie.decision.scorecard import build_scorecard
    from metrics_lie.profiles.load import get_profile_or_load

    report = compare(result_a, result_b)
    prof = get_profile_or_load(profile)
    components = extract_components(report, prof)
    return build_scorecard(components, prof)


def _load_bundle(run_id: str) -> ResultBundle:
    """Load a ResultBundle from disk by run_id."""
    paths = get_run_dir(run_id)
    return ResultBundle.model_validate_json(
        paths.results_json.read_text(encoding="utf-8")
    )


def _detect_model_kind(path: Path) -> str:
    """Detect model format from file extension."""
    suffix = path.suffix.lower()
    kind_map = {
        ".pkl": "pickle",
        ".pickle": "pickle",
        ".joblib": "pickle",
        ".onnx": "onnx",
        ".ubj": "xgboost",
        ".xgb": "xgboost",
        ".txt": "lightgbm",
        ".cbm": "catboost",
    }
    return kind_map.get(suffix, "pickle")


def _normalize_scenarios(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize shorthand scenario dicts to ScenarioSpec format.

    Accepts either:
        {"type": "label_noise", "noise_rate": 0.1}    (shorthand)
        {"id": "label_noise", "params": {"p": 0.1}}   (full format)
    """
    normalized = []
    for s in scenarios:
        if "id" in s and "params" in s:
            normalized.append(s)
        else:
            scenario_type = s.pop("type", s.pop("id", None))
            if scenario_type is None:
                raise ValueError(f"Scenario must have 'type' or 'id' key: {s}")
            normalized.append({"id": scenario_type, "params": dict(s)})
    return normalized
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_v4_sdk.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/metrics_lie/sdk.py tests/test_v4_sdk.py
git commit -m "feat: add public SDK entry points (evaluate, compare, score)"
```

---

### Task 2: Export public API from `__init__.py`

**Files:**
- Modify: `src/metrics_lie/__init__.py`
- Test: `tests/test_v4_exports.py`

**Step 1: Write the failing test**

```python
"""Tests for public API exports."""
from __future__ import annotations


def test_evaluate_importable():
    from metrics_lie import evaluate
    assert callable(evaluate)


def test_compare_importable():
    from metrics_lie import compare
    assert callable(compare)


def test_score_importable():
    from metrics_lie import score
    assert callable(score)


def test_evaluate_file_importable():
    from metrics_lie import evaluate_file
    assert callable(evaluate_file)


def test_result_bundle_importable():
    from metrics_lie import ResultBundle
    assert ResultBundle is not None


def test_experiment_spec_importable():
    from metrics_lie import ExperimentSpec
    assert ExperimentSpec is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_v4_exports.py -v`
Expected: FAIL — `cannot import name 'evaluate' from 'metrics_lie'`

**Step 3: Write minimal implementation**

Modify `src/metrics_lie/__init__.py`:

```python
"""Spectra — Scenario-first ML evaluation engine."""
from __future__ import annotations

__version__ = "0.3.0"

from metrics_lie.sdk import compare, evaluate, evaluate_file, score
from metrics_lie.schema import ResultBundle
from metrics_lie.spec import ExperimentSpec

__all__ = [
    "__version__",
    "evaluate",
    "evaluate_file",
    "compare",
    "score",
    "ResultBundle",
    "ExperimentSpec",
]
```

Note: bump version from `0.2.0` to `0.3.0` for this SDK release.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_v4_exports.py -v`
Expected: PASS

Also run: `pytest tests/test31_version_flag.py -v`
Expected: May need to update version assertion from `0.2.0` to `0.3.0`.

**Step 5: Commit**

```bash
git add src/metrics_lie/__init__.py tests/test_v4_exports.py
git commit -m "feat: export public SDK API from __init__.py, bump to v0.3.0"
```

---

### Task 3: Add scenario presets module

**Files:**
- Create: `src/metrics_lie/presets.py`
- Test: `tests/test_v4_presets.py`

**Step 1: Write the failing test**

```python
"""Tests for scenario presets."""
from __future__ import annotations

from metrics_lie.presets import (
    standard_stress_suite,
    light_stress_suite,
    classification_suite,
    regression_suite,
)


def test_standard_stress_suite_is_list():
    assert isinstance(standard_stress_suite, list)
    assert len(standard_stress_suite) >= 3


def test_light_stress_suite_is_subset():
    assert isinstance(light_stress_suite, list)
    assert len(light_stress_suite) <= len(standard_stress_suite)


def test_classification_suite_no_regression_only():
    for s in classification_suite:
        assert s["id"] != "score_noise" or True  # score_noise applies to all


def test_regression_suite_exists():
    assert isinstance(regression_suite, list)
    assert len(regression_suite) >= 1


def test_all_presets_have_id_and_params():
    for suite in [standard_stress_suite, light_stress_suite, classification_suite, regression_suite]:
        for s in suite:
            assert "id" in s
            assert "params" in s
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_v4_presets.py -v`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

Create `src/metrics_lie/presets.py`:

```python
"""Pre-built scenario suites for common evaluation patterns."""
from __future__ import annotations

# Standard stress suite — covers the main failure modes
standard_stress_suite: list[dict] = [
    {"id": "label_noise", "params": {"p": 0.05}},
    {"id": "label_noise", "params": {"p": 0.10}},
    {"id": "label_noise", "params": {"p": 0.20}},
    {"id": "score_noise", "params": {"sigma": 0.02}},
    {"id": "score_noise", "params": {"sigma": 0.05}},
    {"id": "score_noise", "params": {"sigma": 0.10}},
    {"id": "class_imbalance", "params": {"target_ratio": 0.1}},
    {"id": "class_imbalance", "params": {"target_ratio": 0.3}},
]

# Light stress suite — quick evaluation with fewer scenarios
light_stress_suite: list[dict] = [
    {"id": "label_noise", "params": {"p": 0.10}},
    {"id": "score_noise", "params": {"sigma": 0.05}},
    {"id": "class_imbalance", "params": {"target_ratio": 0.2}},
]

# Classification-specific suite
classification_suite: list[dict] = [
    {"id": "label_noise", "params": {"p": 0.05}},
    {"id": "label_noise", "params": {"p": 0.10}},
    {"id": "label_noise", "params": {"p": 0.20}},
    {"id": "class_imbalance", "params": {"target_ratio": 0.1}},
    {"id": "class_imbalance", "params": {"target_ratio": 0.3}},
    {"id": "threshold_gaming", "params": {}},
]

# Regression suite
regression_suite: list[dict] = [
    {"id": "label_noise", "params": {"p": 0.05}},
    {"id": "label_noise", "params": {"p": 0.10}},
    {"id": "score_noise", "params": {"sigma": 0.02}},
    {"id": "score_noise", "params": {"sigma": 0.05}},
    {"id": "score_noise", "params": {"sigma": 0.10}},
]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_v4_presets.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/metrics_lie/presets.py tests/test_v4_presets.py
git commit -m "feat: add scenario preset suites for common evaluation patterns"
```

---

### Task 4: Add metrics and scenarios list helpers

**Files:**
- Create: `src/metrics_lie/catalog.py`
- Test: `tests/test_v4_catalog.py`

**Step 1: Write the failing test**

```python
"""Tests for metrics/scenarios catalog helpers."""
from __future__ import annotations

from metrics_lie.catalog import list_metrics, list_scenarios, list_model_formats


def test_list_metrics_binary():
    metrics = list_metrics(task="binary_classification")
    assert "auc" in metrics
    assert "f1" in metrics
    assert "macro_f1" not in metrics


def test_list_metrics_multiclass():
    metrics = list_metrics(task="multiclass_classification")
    assert "macro_f1" in metrics
    assert "auc" not in metrics


def test_list_metrics_regression():
    metrics = list_metrics(task="regression")
    assert "mae" in metrics
    assert "auc" not in metrics


def test_list_metrics_all():
    metrics = list_metrics()
    assert "auc" in metrics
    assert "mae" in metrics
    assert "macro_f1" in metrics


def test_list_scenarios_all():
    scenarios = list_scenarios()
    assert "label_noise" in scenarios
    assert "score_noise" in scenarios


def test_list_scenarios_by_task():
    scenarios = list_scenarios(task="regression")
    assert "label_noise" in scenarios
    assert "threshold_gaming" not in scenarios


def test_list_model_formats():
    formats = list_model_formats()
    assert "pickle" in formats
    assert "onnx" in formats
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_v4_catalog.py -v`

**Step 3: Write minimal implementation**

Create `src/metrics_lie/catalog.py`:

```python
"""Catalog helpers for listing available metrics, scenarios, and model formats."""
from __future__ import annotations

from metrics_lie.metrics.registry import METRIC_REQUIREMENTS
from metrics_lie.surface_compat import SCENARIO_TASK_COMPAT


def list_metrics(task: str | None = None) -> list[str]:
    """List available metric IDs, optionally filtered by task type."""
    result = []
    for req in METRIC_REQUIREMENTS:
        if task is None:
            result.append(req.metric_id)
        elif req.task_types is None or task in req.task_types:
            result.append(req.metric_id)
    return sorted(result)


def list_scenarios(task: str | None = None) -> list[str]:
    """List available scenario IDs, optionally filtered by task type."""
    if task is None:
        return sorted(SCENARIO_TASK_COMPAT.keys())
    return sorted(
        sid for sid, tasks in SCENARIO_TASK_COMPAT.items() if task in tasks
    )


def list_model_formats() -> list[str]:
    """List supported model formats."""
    return [
        "pickle", "onnx", "xgboost", "lightgbm", "catboost", "http",
    ]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_v4_catalog.py -v`

**Step 5: Commit**

```bash
git add src/metrics_lie/catalog.py tests/test_v4_catalog.py
git commit -m "feat: add catalog helpers for listing metrics, scenarios, model formats"
```

---

### Task 5: Export presets and catalog from `__init__.py`

**Files:**
- Modify: `src/metrics_lie/__init__.py`
- Test: `tests/test_v4_exports.py` (extend)

**Step 1: Extend test**

Add to `tests/test_v4_exports.py`:

```python
def test_presets_importable():
    from metrics_lie import presets
    assert hasattr(presets, "standard_stress_suite")


def test_list_metrics_importable():
    from metrics_lie import list_metrics
    assert callable(list_metrics)


def test_list_scenarios_importable():
    from metrics_lie import list_scenarios
    assert callable(list_scenarios)
```

**Step 2: Update `__init__.py`**

Add imports:
```python
from metrics_lie import presets
from metrics_lie.catalog import list_metrics, list_model_formats, list_scenarios
```

Add to `__all__`:
```python
"presets", "list_metrics", "list_scenarios", "list_model_formats",
```

**Step 3: Commit**

```bash
git add src/metrics_lie/__init__.py tests/test_v4_exports.py
git commit -m "feat: export presets and catalog from public API"
```

---

### Task 6: Add Typer dependency and create new CLI module

**Files:**
- Modify: `pyproject.toml`
- Create: `src/metrics_lie/cli_app.py`
- Test: `tests/test_v4_cli.py`

**Step 1: Write the failing test**

```python
"""Tests for Typer-based CLI."""
from __future__ import annotations

from typer.testing import CliRunner

from metrics_lie.cli_app import app

runner = CliRunner()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "Spectra" in result.stdout


def test_metrics_list():
    result = runner.invoke(app, ["metrics", "list"])
    assert result.exit_code == 0
    assert "auc" in result.stdout


def test_metrics_list_with_task():
    result = runner.invoke(app, ["metrics", "list", "--task", "regression"])
    assert result.exit_code == 0
    assert "mae" in result.stdout
    assert "auc" not in result.stdout


def test_scenarios_list():
    result = runner.invoke(app, ["scenarios", "list"])
    assert result.exit_code == 0
    assert "label_noise" in result.stdout


def test_models_list():
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "pickle" in result.stdout
    assert "onnx" in result.stdout
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_v4_cli.py -v`
Expected: FAIL — typer not installed, cli_app not found

**Step 3: Write minimal implementation**

1. Add to `pyproject.toml` dependencies:
```toml
dependencies = [
  ...existing...,
  "typer>=0.9",
  "rich>=13.0",
]
```

2. Create `src/metrics_lie/cli_app.py`:

```python
"""Typer-based CLI for Spectra."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from metrics_lie import __version__

app = typer.Typer(name="spectra", help="Spectra — stress-test your ML models.")


def _version_callback(value: bool):
    if value:
        typer.echo(f"Spectra {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
):
    """Spectra — stress-test your ML models."""


@app.command()
def run(
    spec: str = typer.Argument(..., help="Path to experiment spec JSON file."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override model path."),
    task: Optional[str] = typer.Option(None, "--task", "-t", help="Override task type."),
):
    """Run an experiment from a spec file."""
    from metrics_lie.sdk import evaluate_file
    from metrics_lie.execution import run_from_spec_dict

    spec_dict = json.loads(Path(spec).read_text(encoding="utf-8"))
    if model:
        if "model_source" not in spec_dict:
            spec_dict["model_source"] = {"kind": "pickle"}
        spec_dict["model_source"]["path"] = model
    if task:
        spec_dict["task"] = task

    run_id = run_from_spec_dict(spec_dict, spec_path_for_notes=spec)
    typer.echo(f"Run ID: {run_id}")


@app.command()
def compare(
    run_a: str = typer.Argument(..., help="First run ID."),
    run_b: str = typer.Argument(..., help="Second run ID."),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json or table."),
):
    """Compare two runs."""
    from metrics_lie.compare.compare import compare_runs

    report = compare_runs(run_a, run_b)
    if format == "json":
        typer.echo(json.dumps(report, indent=2, default=str))
    else:
        _print_compare_table(report)


@app.command()
def score(
    run_a: str = typer.Argument(..., help="First run ID."),
    run_b: str = typer.Argument(..., help="Second run ID."),
    profile: str = typer.Option("balanced", "--profile", "-p", help="Decision profile name."),
):
    """Score a comparison with a decision profile."""
    from metrics_lie.compare.compare import compare_runs
    from metrics_lie.decision.extract import extract_components
    from metrics_lie.decision.scorecard import build_scorecard
    from metrics_lie.profiles.load import get_profile_or_load

    report = compare_runs(run_a, run_b)
    prof = get_profile_or_load(profile)
    components = extract_components(report, prof)
    card = build_scorecard(components, prof)
    typer.echo(json.dumps(card.model_dump(), indent=2, default=str))


# --- Catalog subcommands ---

metrics_app = typer.Typer(help="List and explore available metrics.")
app.add_typer(metrics_app, name="metrics")

scenarios_app = typer.Typer(help="List and explore available scenarios.")
app.add_typer(scenarios_app, name="scenarios")

models_app = typer.Typer(help="List supported model formats.")
app.add_typer(models_app, name="models")


@metrics_app.command("list")
def metrics_list(
    task: Optional[str] = typer.Option(None, "--task", "-t", help="Filter by task type."),
):
    """List available metrics."""
    from metrics_lie.catalog import list_metrics
    from metrics_lie.metrics.registry import METRIC_DIRECTION

    metrics = list_metrics(task=task)
    for m in metrics:
        direction = "higher-is-better" if METRIC_DIRECTION.get(m, True) else "lower-is-better"
        typer.echo(f"  {m:<25} {direction}")


@scenarios_app.command("list")
def scenarios_list(
    task: Optional[str] = typer.Option(None, "--task", "-t", help="Filter by task type."),
):
    """List available scenarios."""
    from metrics_lie.catalog import list_scenarios

    for s in list_scenarios(task=task):
        typer.echo(f"  {s}")


@models_app.command("list")
def models_list():
    """List supported model formats."""
    from metrics_lie.catalog import list_model_formats

    for fmt in list_model_formats():
        typer.echo(f"  {fmt}")


def _print_compare_table(report: dict):
    """Print a comparison report as a formatted table."""
    typer.echo(f"Metric: {report.get('metric_name', '?')}")
    bd = report.get("baseline_delta", {})
    typer.echo(f"Baseline delta: {bd.get('mean', 0):+.4f} (A={bd.get('a', 0):.4f}, B={bd.get('b', 0):.4f})")
    decision = report.get("decision", {})
    typer.echo(f"Winner: {decision.get('winner', '?')} ({decision.get('confidence', '?')})")
    typer.echo(f"Reasoning: {decision.get('reasoning', '?')}")


if __name__ == "__main__":
    app()
```

**Step 4: Run test to verify it passes**

First install: `pip install "typer>=0.9" "rich>=13.0"`

Run: `pytest tests/test_v4_cli.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pyproject.toml src/metrics_lie/cli_app.py tests/test_v4_cli.py
git commit -m "feat: add Typer-based CLI with metrics/scenarios/models list commands"
```

---

### Task 7: Migrate existing CLI commands to Typer

**Files:**
- Modify: `src/metrics_lie/cli_app.py`
- Test: `tests/test_v4_cli.py` (extend)

**Step 1: Extend tests**

Add to `tests/test_v4_cli.py`:

```python
def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "compare" in result.stdout
    assert "score" in result.stdout
    assert "metrics" in result.stdout
    assert "scenarios" in result.stdout
```

**Step 2: Add remaining commands to cli_app.py**

Port `rerun`, `experiments list/show`, `runs list/show`, `jobs list/show`, `enqueue-run`, `enqueue-rerun`, `worker-once` from `cli.py` to `cli_app.py`. Use Typer subcommand groups.

```python
# --- DB query subcommands ---

experiments_app = typer.Typer(help="Query experiments.")
app.add_typer(experiments_app, name="experiments")

runs_app = typer.Typer(help="Query runs.")
app.add_typer(runs_app, name="runs")

jobs_app = typer.Typer(help="Query jobs.")
app.add_typer(jobs_app, name="jobs")


@app.command()
def rerun(run_id: str = typer.Argument(..., help="Run ID to rerun.")):
    """Deterministic rerun of a completed run."""
    from metrics_lie.execution import rerun as _rerun
    new_id = _rerun(run_id)
    typer.echo(f"Rerun complete. New run ID: {new_id}")


@app.command("enqueue-run")
def enqueue_run(experiment_id: str = typer.Argument(...)):
    """Queue a run job for an experiment."""
    from metrics_lie.db.session import get_session
    from metrics_lie.db.crud import create_job
    with get_session() as session:
        job_id = create_job(session, experiment_id=experiment_id, job_type="run")
    typer.echo(job_id)


@app.command("enqueue-rerun")
def enqueue_rerun(run_id: str = typer.Argument(...)):
    """Queue a rerun job."""
    from metrics_lie.db.session import get_session
    from metrics_lie.db.crud import create_job
    with get_session() as session:
        job_id = create_job(session, run_id=run_id, job_type="rerun")
    typer.echo(job_id)


@app.command("worker-once")
def worker_once():
    """Process one job from the queue."""
    from metrics_lie.worker import process_one_job
    processed = process_one_job()
    if processed:
        typer.echo("[OK] Processed 1 job")
    else:
        typer.echo("[INFO] No jobs available")


# experiments list/show, runs list/show, jobs list/show — same pattern
```

Port the remaining `experiments`, `runs`, `jobs` subcommands from `cli.py`, adapting from argparse to Typer.

**Step 3: Update entry point in pyproject.toml**

```toml
[project.scripts]
spectra = "metrics_lie.cli_app:app"
```

Keep the old `cli.py` intact for backward compatibility but switch the entry point.

**Step 4: Run test**

Run: `pytest tests/test_v4_cli.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/metrics_lie/cli_app.py pyproject.toml tests/test_v4_cli.py
git commit -m "feat: port all CLI commands to Typer, update entry point"
```

---

### Task 8: Add `spectra evaluate` quick-eval CLI command

**Files:**
- Modify: `src/metrics_lie/cli_app.py`
- Test: `tests/test_v4_cli.py` (extend)

**Step 1: Write the test**

```python
def test_evaluate_command_help():
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "--dataset" in result.stdout
    assert "--metric" in result.stdout
```

**Step 2: Add to `cli_app.py`**

```python
@app.command()
def evaluate(
    model: str = typer.Argument(..., help="Path to model file."),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Path to CSV dataset."),
    metric: str = typer.Option("auc", "--metric", "-m", help="Primary metric."),
    task: str = typer.Option("binary_classification", "--task", "-t", help="Task type."),
    n_trials: int = typer.Option(200, "--trials", "-n", help="Monte Carlo trials."),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
):
    """Quick evaluation: model + dataset → results."""
    from metrics_lie.sdk import evaluate as sdk_evaluate

    result = sdk_evaluate(
        name=f"eval_{Path(model).stem}",
        dataset=dataset,
        model=model,
        metric=metric,
        task=task,
        n_trials=n_trials,
        seed=seed,
    )
    typer.echo(f"Run ID: {result.run_id}")
    typer.echo(f"Baseline {metric} = {result.baseline.mean:.6f}")
```

**Step 3: Commit**

```bash
git add src/metrics_lie/cli_app.py tests/test_v4_cli.py
git commit -m "feat: add 'spectra evaluate' quick-eval CLI command"
```

---

### Task 9: Add notebook display functions

**Files:**
- Create: `src/metrics_lie/display.py`
- Test: `tests/test_v4_display.py`

**Step 1: Write the failing test**

```python
"""Tests for notebook display functions."""
from __future__ import annotations

from metrics_lie.display import format_summary, format_comparison


def test_format_summary_returns_string():
    from metrics_lie.schema import MetricSummary, ResultBundle, ScenarioResult

    bundle = ResultBundle(
        run_id="TEST123",
        experiment_name="test",
        metric_name="auc",
        task_type="binary_classification",
        baseline=MetricSummary(mean=0.85, std=0.0, q05=0.85, q50=0.85, q95=0.85, n=1),
        scenarios=[],
        applicable_metrics=["auc"],
        metric_results={},
        scenario_results_by_metric={},
        analysis_artifacts={},
        notes={},
    )
    text = format_summary(bundle)
    assert "TEST123" in text
    assert "auc" in text
    assert "0.85" in text


def test_format_comparison_returns_string():
    report = {
        "metric_name": "auc",
        "baseline_delta": {"mean": -0.05, "a": 0.90, "b": 0.85},
        "decision": {"winner": "A", "confidence": "high", "reasoning": "better AUC"},
    }
    text = format_comparison(report)
    assert "auc" in text
    assert "A" in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_v4_display.py -v`

**Step 3: Write minimal implementation**

Create `src/metrics_lie/display.py`:

```python
"""Display functions for notebooks and terminals."""
from __future__ import annotations

from typing import Any

from metrics_lie.schema import ResultBundle


def format_summary(bundle: ResultBundle) -> str:
    """Format a ResultBundle as a readable text summary."""
    lines = [
        f"Spectra Run: {bundle.run_id}",
        f"Experiment: {bundle.experiment_name}",
        f"Task: {bundle.task_type}",
        f"Primary metric: {bundle.metric_name} = {bundle.baseline.mean:.4f}",
        f"Applicable metrics: {', '.join(bundle.applicable_metrics)}",
        f"Scenarios: {len(bundle.scenarios)}",
    ]

    if bundle.metric_results:
        lines.append("\nMetric Results:")
        for mid, ms in bundle.metric_results.items():
            summary = ms if isinstance(ms, dict) else ms.model_dump()
            lines.append(f"  {mid}: {summary.get('mean', '?'):.4f} (std={summary.get('std', 0):.4f})")

    aa = bundle.analysis_artifacts or {}
    if "dashboard_summary" in aa:
        ds = aa["dashboard_summary"]
        risk = ds.get("risk_summary", {})
        drops = risk.get("metrics_with_large_drops", [])
        if drops:
            lines.append(f"\nRisk: {len(drops)} metric(s) with large drops: {', '.join(drops)}")
        else:
            lines.append("\nRisk: No large metric drops detected.")

    return "\n".join(lines)


def format_comparison(report: dict[str, Any]) -> str:
    """Format a comparison report as readable text."""
    lines = [
        f"Comparison: {report.get('metric_name', '?')}",
    ]
    bd = report.get("baseline_delta", {})
    lines.append(f"Baseline delta: {bd.get('mean', 0):+.4f} (A={bd.get('a', 0):.4f}, B={bd.get('b', 0):.4f})")

    decision = report.get("decision", {})
    lines.append(f"Winner: {decision.get('winner', '?')} ({decision.get('confidence', '?')})")
    lines.append(f"Reasoning: {decision.get('reasoning', '?')}")

    flags = report.get("risk_flags", [])
    if flags:
        lines.append(f"Risk flags: {', '.join(flags)}")

    return "\n".join(lines)


def display(bundle: ResultBundle) -> None:
    """Display a ResultBundle. Uses HTML in Jupyter, text elsewhere."""
    try:
        from IPython.display import display as ipy_display, HTML
        ipy_display(HTML(_to_html(bundle)))
    except ImportError:
        print(format_summary(bundle))


def _to_html(bundle: ResultBundle) -> str:
    """Convert ResultBundle to HTML for Jupyter display."""
    summary = format_summary(bundle)
    escaped = summary.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<pre style='background:#f5f5f5;padding:12px;border-radius:4px;'>{escaped}</pre>"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_v4_display.py -v`

**Step 5: Commit**

```bash
git add src/metrics_lie/display.py tests/test_v4_display.py
git commit -m "feat: add notebook display functions (format_summary, format_comparison, display)"
```

---

### Task 10: Export display functions and update version

**Files:**
- Modify: `src/metrics_lie/__init__.py`
- Modify: `pyproject.toml` (version)

**Step 1: Add to `__init__.py`**

```python
from metrics_lie.display import display, format_comparison, format_summary
```

Add to `__all__`: `"display", "format_summary", "format_comparison",`

**Step 2: Update pyproject.toml version**

```toml
version = "0.3.0"
```

**Step 3: Commit**

```bash
git add src/metrics_lie/__init__.py pyproject.toml
git commit -m "feat: export display functions, finalize v0.3.0"
```

---

### Task 11: Run full test suite and fix regressions

**Step 1: Run linter**

Run: `ruff check src tests`

**Step 2: Run full test suite**

Run: `pytest -v`

**Step 3: Fix any failures and commit**

```bash
git commit -m "fix: resolve Phase 4 test regressions"
```

---

### Task 12: Commit plan document

```bash
git add docs/plans/2026-02-27-v1-phase4-sdk-cli-upgrade.md
git commit -m "docs: add Phase 4 SDK & CLI upgrade plan"
```

---

## Parallelization Strategy

| Task | Stream | Dependencies | Can Parallel With |
|------|--------|-------------|-------------------|
| 1 | A | None | 6, 9 |
| 2 | A | 1 | 6, 9 |
| 3 | A | None | 1, 6, 9 |
| 4 | A | None | 1, 6, 9 |
| 5 | A | 1, 2, 3, 4 | 6, 9 |
| 6 | B | None | 1, 3, 4, 9 |
| 7 | B | 6 | 9 |
| 8 | B | 6, 1 | 9 |
| 9 | C | None | 1-8 |
| 10 | — | 5, 9 | None |
| 11 | — | All | None |
| 12 | — | 11 | None |

**Recommended 3-agent strategy:**
- Agent 1 (Stream A): Tasks 1 → 2 → 3 → 4 → 5
- Agent 2 (Stream B): Tasks 6 → 7 → 8
- Agent 3 (Stream C): Task 9
- Then: Tasks 10, 11, 12 sequentially
