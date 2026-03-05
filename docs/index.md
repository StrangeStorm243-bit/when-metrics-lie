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
