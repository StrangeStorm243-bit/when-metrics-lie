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
