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
