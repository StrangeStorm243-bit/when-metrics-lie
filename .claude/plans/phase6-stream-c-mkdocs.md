# Plan: Phase 6 Stream C — MkDocs Documentation Site

> **Execution model:** This plan was written by Opus for execution by Sonnet.
> Run: `claude --model claude-sonnet-4-6` then say "Execute plan .claude/plans/phase6-stream-c-mkdocs.md"

## Goal

Set up a MkDocs Material documentation site with auto-generated API reference, getting-started guide, CLI reference, concepts overview, integration guides, and contributing guide. ~10 pages.

## Context

- No existing MkDocs setup — this creates everything from scratch
- Project root: `C:\GitHubProjects\when-metrics-lie`
- Public SDK API in `src/metrics_lie/__init__.py`: evaluate, evaluate_file, compare, score, log_to_mlflow, ResultBundle, ExperimentSpec
- CLI entry point: `spectra` command via Typer in `src/metrics_lie/cli_app.py`
- CLI commands: run, evaluate, compare, score, rerun, enqueue-run, worker-once, experiments, runs, jobs
- Supported model formats: sklearn (.pkl, .joblib), ONNX (.onnx), XGBoost (.ubj, .xgb), LightGBM (.lgb), CatBoost (.cbm), HTTP endpoints, MLflow (new in Phase 6)
- Task types: binary_classification, multiclass_classification, regression, ranking
- 4 scenarios: label_noise, score_noise, class_imbalance, threshold_gaming
- Decision profiles: balanced, risk_averse, performance_focused

## Prerequisites

- [ ] Read `src/metrics_lie/__init__.py` for the public API
- [ ] Read `src/metrics_lie/cli_app.py` for CLI commands (first 50 lines for structure)

## Tasks

### Task C1: Create mkdocs.yml and add docs dependency group

**Files:**
- Create: `mkdocs.yml`
- Modify: `pyproject.toml`

**Steps:**

1. Create `mkdocs.yml` at project root:

```yaml
site_name: Spectra Documentation
site_description: Scenario-first ML evaluation engine — stress-test models to find where metrics lie
site_url: https://strangestorm243-bit.github.io/when-metrics-lie/
repo_url: https://github.com/StrangeStorm243-bit/when-metrics-lie
repo_name: StrangeStorm243-bit/when-metrics-lie

theme:
  name: material
  palette:
    - scheme: default
      primary: deep purple
      accent: amber
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: deep purple
      accent: amber
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - content.code.copy
    - search.suggest

nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - Concepts: concepts.md
  - CLI Reference: cli-reference.md
  - API Reference:
    - SDK: api-reference/sdk.md
    - ExperimentSpec: api-reference/spec.md
    - ResultBundle: api-reference/result-bundle.md
  - Integrations:
    - Model Formats: integrations/model-formats.md
    - MLflow: integrations/mlflow.md
  - Contributing: contributing.md

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [src]
          options:
            show_source: false
            show_root_heading: true
            heading_level: 3

markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
  - admonition
  - pymdownx.details
  - toc:
      permalink: true
```

2. Add `docs` dependency group to `pyproject.toml`. After the `mlflow` group (or `drift` if Stream A hasn't run yet), add:

```toml
docs = [
  "mkdocs-material>=9.0",
  "mkdocstrings[python]>=0.24"
]
```

3. Update the `all` group to include `docs`:

Add `docs` to the `all` extras list (append `,docs` before the closing bracket).

4. Run: `cd /c/GitHubProjects/when-metrics-lie && python -c "import tomllib; t = tomllib.loads(open('pyproject.toml').read()); print(list(t['project']['optional-dependencies'].keys()))"`
   Expected: List includes 'docs'

**Acceptance:** `mkdocs.yml` exists at project root. `docs` dependency group in pyproject.toml.

### Task C2: Create docs directory structure and index page

**Files:**
- Create: `docs/index.md`

**Steps:**

1. Create the docs directory and index page:

```markdown
# Spectra

**Scenario-first ML evaluation engine** — stress-test models to find where metrics lie.

Don't trust your metrics. Prove them.

## What is Spectra?

Spectra runs your ML model through realistic failure scenarios (noisy labels, score perturbation, class imbalance, threshold gaming) and reports where your metrics disagree, break down, or mislead.

Instead of a single accuracy number, you get:

- **Baseline vs. stressed metrics** across Monte Carlo trials
- **Metric disagreement analysis** showing which metrics tell different stories
- **Calibration and gaming detection** catching optimistic thresholds
- **Decision framework** with weighted scoring for model comparison

## Quick Example

```python
import metrics_lie as spectra

result = spectra.evaluate(
    name="my_model_eval",
    dataset="data.csv",
    model="model.pkl",
    metric="auc",
    task="binary_classification",
)

spectra.display(result)
```

## Supported Task Types

| Task Type | Metrics | Scenarios |
|-----------|---------|-----------|
| Binary Classification | AUC, F1, Brier, ECE, PR-AUC, MCC, ... | All 4 |
| Multiclass Classification | Macro F1, Weighted F1, Cohen's Kappa, ... | label_noise, score_noise, class_imbalance |
| Regression | MAE, MSE, RMSE, R-squared, MAPE, ... | label_noise, score_noise |
| Ranking | NDCG, MAP, MRR, Hit Rate | label_noise, score_noise |

## Supported Model Formats

| Format | Extensions | Install |
|--------|-----------|---------|
| sklearn | `.pkl`, `.joblib` | Core (included) |
| ONNX | `.onnx` | `pip install metrics_lie[onnx]` |
| XGBoost | `.ubj`, `.xgb` | `pip install metrics_lie[boosting]` |
| LightGBM | `.lgb` | `pip install metrics_lie[boosting]` |
| CatBoost | `.cbm` | `pip install metrics_lie[boosting]` |
| HTTP Endpoint | — | Core (included) |
| MLflow Registry | — | `pip install metrics_lie[mlflow]` |

## Next Steps

- [Getting Started](getting-started.md) — install and run your first evaluation
- [CLI Reference](cli-reference.md) — all commands
- [API Reference](api-reference/sdk.md) — Python SDK
```

2. Run: `ls docs/index.md` — Expected: file exists

**Acceptance:** Index page with overview, quick example, task type table, model format table.

### Task C3: Create getting-started guide

**Files:**
- Create: `docs/getting-started.md`

**Steps:**

1. Create the guide:

```markdown
# Getting Started

Install Spectra, run your first evaluation, and compare two models in under 5 minutes.

## Installation

```bash
# Core (sklearn models, tabular data)
pip install metrics_lie

# With ONNX support
pip install metrics_lie[onnx]

# With boosting (XGBoost, LightGBM, CatBoost)
pip install metrics_lie[boosting]

# Everything
pip install metrics_lie[all]
```

## Your First Evaluation

### 1. Prepare your data

Spectra needs a CSV with at least two columns:

| y_true | y_score | group |
|--------|---------|-------|
| 1 | 0.87 | A |
| 0 | 0.23 | B |
| 1 | 0.65 | A |

- `y_true`: Ground truth labels
- `y_score`: Model predictions (probabilities, scores, or labels)
- `group` (optional): Subgroup column for fairness analysis

### 2. Run from Python

```python
import metrics_lie as spectra

result = spectra.evaluate(
    name="my_first_eval",
    dataset="data.csv",
    metric="auc",
    task="binary_classification",
    y_true_col="y_true",
    y_score_col="y_score",
)

# Display results
spectra.display(result)
```

### 3. Run from CLI

```bash
# Create a spec file
cat > spec.json << 'EOF'
{
  "name": "my_first_eval",
  "task": "binary_classification",
  "dataset": {
    "source": "csv",
    "path": "data.csv",
    "y_true_col": "y_true",
    "y_score_col": "y_score"
  },
  "metric": "auc",
  "scenarios": [
    {"id": "label_noise", "params": {"p": 0.1}},
    {"id": "score_noise", "params": {"sigma": 0.05}}
  ],
  "n_trials": 200,
  "seed": 42
}
EOF

spectra run spec.json
```

### 4. With a trained model

If you have a trained sklearn model, Spectra can generate predictions itself:

```python
result = spectra.evaluate(
    name="with_model",
    dataset="data.csv",
    model="model.pkl",
    metric="auc",
    y_true_col="y_true",
    y_score_col="y_score",
)
```

## Comparing Models

```python
result_a = spectra.evaluate(name="model_a", dataset="data.csv", model="model_a.pkl", metric="auc")
result_b = spectra.evaluate(name="model_b", dataset="data.csv", model="model_b.pkl", metric="auc")

report = spectra.compare(result_a, result_b)
spectra.format_comparison(report)
```

Or from CLI:

```bash
spectra compare <run_id_a> <run_id_b>
spectra score <run_id_a> <run_id_b> --profile balanced
```

## Regression Evaluation

```python
result = spectra.evaluate(
    name="regression_eval",
    dataset="housing.csv",
    model="regressor.pkl",
    metric="mae",
    task="regression",
    y_true_col="price",
    y_score_col="predicted_price",
)
```

## What's Next

- [Concepts](concepts.md) — understand scenarios, metrics, and decision profiles
- [CLI Reference](cli-reference.md) — all available commands
- [Model Formats](integrations/model-formats.md) — ONNX, XGBoost, HTTP endpoints, and more
```

**Acceptance:** Complete getting-started guide with install, Python, CLI, comparison, and regression examples.

### Task C4: Create CLI reference

**Files:**
- Create: `docs/cli-reference.md`

**Steps:**

1. Create the reference:

```markdown
# CLI Reference

Spectra's CLI is available via the `spectra` command after installation.

## Core Commands

### `spectra run <spec.json>`

Run an experiment from a JSON spec file.

```bash
spectra run experiments/binary_eval.json
```

### `spectra evaluate`

Quick evaluation without a spec file.

```bash
spectra evaluate --name quick_test \
  --dataset data.csv \
  --metric auc \
  --task binary_classification
```

### `spectra compare <run_a> <run_b>`

Compare two runs side-by-side.

```bash
spectra compare abc123 def456
```

Output includes baseline delta, per-scenario deltas, regression flags, and a winner recommendation.

### `spectra score <run_a> <run_b>`

Compare with decision profile scoring.

```bash
spectra score abc123 def456 --profile balanced
spectra score abc123 def456 --profile risk_averse
spectra score abc123 def456 --profile performance_focused
```

Profiles weight different components (calibration, robustness, fairness, etc.) differently.

### `spectra rerun <run_id>`

Deterministically rerun an experiment.

```bash
spectra rerun abc123
```

## Job Queue Commands

### `spectra enqueue-run <experiment_id>`

Queue a run for background processing.

### `spectra worker-once`

Process one job from the queue.

## Query Commands

### `spectra experiments list`

List experiments.

```bash
spectra experiments list --limit 10
```

### `spectra experiments show <id>`

Show experiment details.

### `spectra runs list`

List runs.

```bash
spectra runs list --limit 10 --status completed
spectra runs list --experiment <experiment_id>
```

### `spectra runs show <id>`

Show run details.

### `spectra jobs list`

List queued jobs.

```bash
spectra jobs list --limit 10 --status pending
```

## Decision Profiles

Three built-in profiles for `spectra score`:

| Profile | Focus |
|---------|-------|
| `balanced` | Equal weight across all components |
| `risk_averse` | Prioritizes calibration and robustness |
| `performance_focused` | Prioritizes headline metric performance |

Custom profiles: pass a JSON file path instead of a profile name.
```

**Acceptance:** CLI reference covering all commands with examples.

### Task C5: Create API reference pages

**Files:**
- Create: `docs/api-reference/sdk.md`
- Create: `docs/api-reference/spec.md`
- Create: `docs/api-reference/result-bundle.md`

**Steps:**

1. Create `docs/api-reference/sdk.md`:

```markdown
# SDK Reference

The Spectra Python SDK is the primary interface for programmatic use.

```python
import metrics_lie as spectra
```

## Core Functions

::: metrics_lie.sdk.evaluate

::: metrics_lie.sdk.evaluate_file

::: metrics_lie.sdk.compare

::: metrics_lie.sdk.score

## MLflow Integration

::: metrics_lie.sdk.log_to_mlflow

## Display Functions

::: metrics_lie.display.display

::: metrics_lie.display.format_summary

::: metrics_lie.display.format_comparison

## Catalog

::: metrics_lie.catalog.list_metrics

::: metrics_lie.catalog.list_scenarios

::: metrics_lie.catalog.list_model_formats
```

2. Create `docs/api-reference/spec.md`:

```markdown
# ExperimentSpec Reference

The `ExperimentSpec` defines what Spectra evaluates. It can be passed as a JSON file to the CLI or constructed programmatically.

## JSON Format

```json
{
  "name": "experiment_name",
  "task": "binary_classification",
  "dataset": {
    "source": "csv",
    "path": "data.csv",
    "y_true_col": "y_true",
    "y_score_col": "y_score",
    "subgroup_col": "group",
    "feature_cols": ["feat_1", "feat_2"]
  },
  "model_source": {
    "kind": "pickle",
    "path": "model.pkl",
    "threshold": 0.5
  },
  "metric": "auc",
  "scenarios": [
    {"id": "label_noise", "params": {"p": 0.1}},
    {"id": "score_noise", "params": {"sigma": 0.05}},
    {"id": "class_imbalance", "params": {"target_pos_rate": 0.2}},
    {"id": "threshold_gaming", "params": {"delta_threshold": 0.05}}
  ],
  "n_trials": 200,
  "seed": 42
}
```

## Fields

### Top-Level

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | Yes | — | Experiment name |
| `task` | string | No | `binary_classification` | Task type |
| `metric` | string | Yes | — | Primary metric ID |
| `n_trials` | int | No | `200` | Monte Carlo trial count |
| `seed` | int | No | `42` | Random seed |

### Task Types

| Value | Description |
|-------|-------------|
| `binary_classification` | Two-class classification |
| `multiclass_classification` | Three or more classes |
| `regression` | Continuous target |
| `ranking` | Ordered relevance |

### Dataset

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | Yes | `"csv"` |
| `path` | string | Yes | Path to CSV file |
| `y_true_col` | string | Yes | Ground truth column |
| `y_score_col` | string | Yes | Prediction column |
| `subgroup_col` | string | No | Subgroup column for fairness |
| `feature_cols` | list[string] | No | Feature columns (for model inference) |

### Model Source

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `kind` | string | Yes | `pickle`, `onnx`, `xgboost`, `lightgbm`, `catboost`, `http`, `mlflow` |
| `path` | string | Depends | File path (for file-based formats) |
| `endpoint` | string | Depends | URL (for `http` kind) |
| `uri` | string | Depends | MLflow URI (for `mlflow` kind) |
| `threshold` | float | No | Decision threshold (default: 0.5) |

### Scenarios

| Scenario ID | Params | Applicable Tasks |
|-------------|--------|------------------|
| `label_noise` | `p`: flip rate (0.0-0.5) | All |
| `score_noise` | `sigma`: noise std (0.0-0.5) | All |
| `class_imbalance` | `target_pos_rate`, `max_remove_frac` | Classification |
| `threshold_gaming` | `delta_threshold` (-0.5 to +0.5) | Binary only |
```

3. Create `docs/api-reference/result-bundle.md`:

```markdown
# ResultBundle Reference

The `ResultBundle` is the output of every Spectra evaluation. It contains baseline metrics, scenario results, diagnostics, and analysis artifacts.

## Structure

```python
from metrics_lie import ResultBundle

# ResultBundle fields
result.run_id              # Unique run identifier
result.experiment_name     # Experiment name from spec
result.metric_name         # Primary metric
result.task_type           # Task type string
result.baseline            # MetricSummary for baseline
result.scenarios           # List[ScenarioResult]
result.metric_results      # Dict[str, MetricSummary] for all metrics
result.applicable_metrics  # List of applicable metric names
result.analysis_artifacts  # Dict with threshold_sweep, sensitivity, etc.
result.notes               # Dict with diagnostics, task_specific data
result.created_at          # ISO timestamp
```

## MetricSummary

Statistics across Monte Carlo trials:

| Field | Type | Description |
|-------|------|-------------|
| `mean` | float | Mean across trials |
| `std` | float | Standard deviation |
| `q05` | float | 5th percentile |
| `q50` | float | Median |
| `q95` | float | 95th percentile |
| `n` | int | Number of trials |

## ScenarioResult

| Field | Type | Description |
|-------|------|-------------|
| `scenario_id` | string | Scenario identifier |
| `params` | dict | Scenario parameters |
| `metric` | MetricSummary | Metric stats under this scenario |
| `diagnostics` | dict | Scenario-specific diagnostics |

## Serialization

```python
# To JSON
json_str = result.to_pretty_json()

# From JSON
result = ResultBundle.model_validate_json(json_str)
```

## Display

```python
import metrics_lie as spectra

# Rich terminal display
spectra.display(result)

# Plain text summary
print(spectra.format_summary(result))
```
```

**Acceptance:** Three API reference pages. SDK page uses mkdocstrings `::: module.function` directives for auto-generation.

### Task C6: Create concepts and integration pages

**Files:**
- Create: `docs/concepts.md`
- Create: `docs/integrations/model-formats.md`
- Create: `docs/integrations/mlflow.md`

**Steps:**

1. Create `docs/concepts.md`:

```markdown
# Concepts

## Scenarios

Scenarios simulate realistic failure modes. Each scenario perturbs the data or model behavior, then re-evaluates metrics across multiple Monte Carlo trials.

### Label Noise
Randomly flips a fraction of ground truth labels. Simulates annotation errors.

- **Parameter**: `p` — flip probability (0.0 to 0.5)
- **Use case**: How robust is your model to mislabeled training/test data?

### Score Noise
Adds Gaussian noise to model predictions. Simulates prediction instability.

- **Parameter**: `sigma` — noise standard deviation (0.0 to 0.5)
- **Use case**: How stable are your metrics when predictions are slightly perturbed?

### Class Imbalance
Removes samples to shift the class distribution. Simulates deployment drift.

- **Parameters**: `target_pos_rate`, `max_remove_frac`
- **Use case**: Does your model's performance hold under different class ratios?

### Threshold Gaming
Shifts the decision threshold. Detects threshold-optimized metrics.

- **Parameter**: `delta_threshold` (-0.5 to +0.5)
- **Use case**: Is your reported accuracy inflated by a cherry-picked threshold?

## Metrics

Spectra computes metrics across all Monte Carlo trials, producing distributions (mean, std, quantiles) rather than point estimates.

### By Task Type

**Binary Classification**: AUC, F1, Precision, Recall, Accuracy, Log Loss, Brier Score, ECE, MCC, PR-AUC

**Multiclass**: Macro F1, Weighted F1, Macro/Weighted Precision & Recall, Multiclass AUC, Cohen's Kappa

**Regression**: MAE, MSE, RMSE, R-squared, MAPE, Median Absolute Error, Max Error

**Ranking**: NDCG, MAP, MRR, Hit Rate

## Decision Profiles

Decision profiles weight different evaluation components when comparing two models:

| Profile | Emphasis |
|---------|----------|
| `balanced` | Equal weight across calibration, robustness, fairness, and performance |
| `risk_averse` | Heavier weight on calibration and worst-case scenarios |
| `performance_focused` | Prioritizes headline metric improvement |

Use profiles with `spectra.score(result_a, result_b, profile="balanced")`.

## Analysis Artifacts

Beyond metrics, Spectra produces:

- **Threshold Sweep**: Metric values across decision thresholds, showing crossover points
- **Sensitivity Analysis**: Which perturbation parameters have the largest impact
- **Metric Disagreement**: Pairs of metrics that tell different stories
- **Failure Modes**: Worst-case scenario + metric combinations
- **Dashboard Summary**: Multi-metric risk overview
```

2. Create `docs/integrations/model-formats.md`:

```markdown
# Model Formats

Spectra supports multiple model formats through its adapter registry.

## sklearn (Pickle/Joblib)

The default format. Works with any scikit-learn estimator.

```python
result = spectra.evaluate(
    name="sklearn_eval",
    dataset="data.csv",
    model="model.pkl",     # or model.joblib
    metric="auc",
)
```

**Extensions**: `.pkl`, `.pickle`, `.joblib`
**Install**: Included in core

## ONNX

Cross-framework models exported to ONNX format.

```python
result = spectra.evaluate(
    name="onnx_eval",
    dataset="data.csv",
    model="model.onnx",
    metric="auc",
)
```

**Extensions**: `.onnx`
**Install**: `pip install metrics_lie[onnx]`

## XGBoost / LightGBM / CatBoost

Gradient boosting models in their native formats.

```python
result = spectra.evaluate(
    name="xgb_eval",
    dataset="data.csv",
    model="model.ubj",     # XGBoost
    metric="auc",
)
```

**Extensions**: `.ubj`, `.xgb` (XGBoost), `.lgb` (LightGBM), `.cbm` (CatBoost)
**Install**: `pip install metrics_lie[boosting]`

## HTTP Endpoints

Evaluate models served via REST API.

```json
{
  "model_source": {
    "kind": "http",
    "endpoint": "http://localhost:8080/predict"
  }
}
```

**Protocol**: POST `{"instances": [[f1, f2, ...]]}` returns `{"predictions": [{"label": 0, "probability": [0.8, 0.2]}]}`

### KServe V2 Protocol

Auto-detected when the URL contains `/v2/`:

```json
{
  "model_source": {
    "kind": "http",
    "endpoint": "http://localhost:8080",
    "model_name": "my_model",
    "protocol": "kserve_v2"
  }
}
```

**Install**: Included in core (requires `requests`)

## MLflow Registry

Load models from MLflow's model registry.

```json
{
  "model_source": {
    "kind": "mlflow",
    "uri": "runs:/abc123def/model"
  }
}
```

Supports all MLflow URI formats: `runs:/`, `models:/name/version`, `models:/name/stage`.

**Install**: `pip install metrics_lie[mlflow]`
```

3. Create `docs/integrations/mlflow.md`:

```markdown
# MLflow Integration

Spectra integrates with MLflow for model loading and result logging.

## Installation

```bash
pip install metrics_lie[mlflow]
```

## Logging Results to MLflow

After running an evaluation, log the results to MLflow tracking:

```python
import metrics_lie as spectra

result = spectra.evaluate(
    name="my_eval",
    dataset="data.csv",
    model="model.pkl",
    metric="auc",
)

# Log to MLflow
run_id = spectra.log_to_mlflow(result)
print(f"Logged to MLflow run: {run_id}")
```

### What Gets Logged

| MLflow Entity | Content |
|---------------|---------|
| **Params** | experiment_name, metric_name, task_type, spectra run_id |
| **Metrics** | Baseline mean/std/q05/q95, per-metric means, scenario deltas |
| **Artifacts** | Full ResultBundle JSON in `spectra/` artifact directory |

### Advanced Options

```python
# Log to a specific MLflow run
spectra.log_to_mlflow(result, run_id="existing_run_id")

# Log to a named experiment
spectra.log_to_mlflow(result, experiment_name="model_evaluation")

# Use a remote tracking server
spectra.log_to_mlflow(result, tracking_uri="http://mlflow.example.com:5000")
```

## Loading Models from MLflow

Evaluate a model directly from MLflow's model registry:

```python
result = spectra.evaluate(
    name="mlflow_model_eval",
    dataset="data.csv",
    metric="auc",
)
```

Or via spec file:

```json
{
  "name": "mlflow_eval",
  "task": "binary_classification",
  "dataset": {
    "source": "csv",
    "path": "data.csv",
    "y_true_col": "y_true",
    "y_score_col": "y_score"
  },
  "model_source": {
    "kind": "mlflow",
    "uri": "models:/my_model/Production"
  },
  "metric": "auc"
}
```

### Supported URIs

| URI Format | Example |
|------------|---------|
| Run artifact | `runs:/abc123def456/model` |
| Model version | `models:/my_model/3` |
| Model stage | `models:/my_model/Production` |
```

**Acceptance:** Three content pages covering concepts, model formats, and MLflow integration.

### Task C7: Create contributing guide

**Files:**
- Create: `docs/contributing.md`

**Steps:**

1. Create the guide:

```markdown
# Contributing

Spectra is extensible by design. Here's how to add new metrics, scenarios, and model adapters.

## Development Setup

```bash
git clone https://github.com/StrangeStorm243-bit/when-metrics-lie.git
cd when-metrics-lie
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest  # verify everything works
```

## Adding a Metric

Metrics are registered in `src/metrics_lie/metrics/`:

1. Add the metric function in `core.py`:

```python
def my_metric(y_true, y_pred, **kwargs):
    """Compute my custom metric."""
    return float(some_computation(y_true, y_pred))
```

2. Register it in `registry.py`:

```python
MetricRequirement(
    name="my_metric",
    fn=my_metric,
    requires_surface={SurfaceType.PROBABILITY},
    task_types={TaskType.BINARY_CLASSIFICATION},
    higher_is_better=True,
)
```

3. Add a test in `tests/`.

## Adding a Scenario

Scenarios implement the `Scenario` protocol in `src/metrics_lie/scenarios/`:

1. Create `src/metrics_lie/scenarios/my_scenario.py`:

```python
class MyScenario:
    id = "my_scenario"

    def apply(self, context):
        # Perturb context.y_true or context.y_score
        return modified_y_true, modified_y_score
```

2. Register in `registry.py`.

## Adding a Model Adapter

Adapters implement `ModelAdapterProtocol` in `src/metrics_lie/model/`:

1. Create `src/metrics_lie/model/adapters/my_adapter.py`
2. Implement: `task_type`, `metadata`, `predict()`, `predict_proba()`, `predict_raw()`, `get_all_surfaces()`
3. Register in `default_registry.py`

## Code Standards

- Python 3.11+, `from __future__ import annotations`
- Full type annotations
- `ruff check` for linting
- `pytest` for testing
- Commit messages: `feat:`, `fix:`, `test:`, `docs:`
```

**Acceptance:** Contributing guide with setup, extension patterns, and code standards.

### Task C8: Verify docs build

**Steps:**

1. Run: `cd /c/GitHubProjects/when-metrics-lie && pip install mkdocs-material "mkdocstrings[python]>=0.24" 2>&1 | tail -3`
2. Run: `cd /c/GitHubProjects/when-metrics-lie && mkdocs build --strict 2>&1 | tail -10`
   Expected: Build succeeds. Some mkdocstrings warnings are OK if mlflow is not installed, but the build should not error.
3. If build fails, check for:
   - Missing `docs/` subdirectories (create `docs/api-reference/` and `docs/integrations/` dirs)
   - YAML syntax errors in `mkdocs.yml`
   - Fix any issues and re-run

**Acceptance:** `mkdocs build` produces `site/` directory with all pages rendered.

## Boundaries

**DO:**
- Follow steps exactly
- Create all files in `docs/` directory
- Use actual project details (URLs, commands, formats)

**DO NOT:**
- Modify any Python source files
- Modify any frontend files
- Create a git commit (parent agent handles this)
- Add GitHub Pages deployment action (that's a follow-up)

## Escalation Triggers

Stop and flag for Opus review if:
- `mkdocs build` fails with errors that can't be fixed by creating missing directories
- mkdocstrings can't find the Python modules
- More than 2 build warnings about missing references

When escalating, write to `.claude/plans/phase6-stream-c-blockers.md`.

## Verification

After all tasks complete:
- [ ] `mkdocs build --strict` succeeds (or only has mlflow-related warnings)
- [ ] All 10 pages exist in `docs/`
- [ ] `mkdocs.yml` nav matches actual file structure
- [ ] No Python source files modified
