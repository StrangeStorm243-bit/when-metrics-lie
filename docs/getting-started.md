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
