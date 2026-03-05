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
