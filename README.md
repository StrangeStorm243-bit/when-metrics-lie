# Spectra

[![CI](https://github.com/StrangeStorm243-bit/when-metrics-lie/actions/workflows/ci.yml/badge.svg)](https://github.com/StrangeStorm243-bit/when-metrics-lie/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

**Scenario-first ML evaluation engine.** Stress-test your models to find where metrics lie.

Spectra runs your model through realistic failure scenarios (label noise, score noise, class imbalance, threshold gaming) and shows you exactly where your metrics break down. Instead of a single accuracy number, you get a transparent stress-test report.

## Install

```bash
pip install spectra-ml
```

With web UI support:

```bash
pip install spectra-ml[web]
```

## Quick Start

### Python SDK

```python
import metrics_lie as spectra

result = spectra.evaluate(
    name="my-model-audit",
    dataset="data.csv",
    model="model.pkl",
    metric="auc",
    trust_pickle=True,
)

spectra.display(result)
```

### CLI

```bash
# Run from spec file
spectra run experiment.json

# Quick evaluation
spectra evaluate model.pkl --dataset data.csv --metric auc --trust-pickle

# Launch web UI
spectra serve
```

### Web UI (Quick Test)

```bash
pip install spectra-ml[web]
spectra serve
```

Upload your model + dataset CSV. Spectra auto-detects columns, task type, and best metric. One click to run a full stress test.

## What It Does

1. **Stress-tests metrics** across scenarios: label noise, score noise, class imbalance, threshold gaming
2. **Detects metric disagreement** — when accuracy says "great" but calibration says "broken"
3. **Runs diagnostics**: calibration analysis, subgroup gaps, sensitivity ranking, threshold sweeps
4. **Produces decision scorecards** with weighted components and transparent reasoning
5. **Compares models** with regression detection and structured comparison reports

## Supported

| Category | Options |
|----------|---------|
| **Task Types** | Binary classification, multiclass, regression, ranking |
| **Metrics** | 27 metrics: AUC, F1, precision, recall, Brier, ECE, MAE, RMSE, R2, NDCG, and more |
| **Model Formats** | sklearn pickle, ONNX, PyTorch, TensorFlow, XGBoost, LightGBM, CatBoost, MLflow |
| **Scenarios** | Label noise, score noise, class imbalance, threshold gaming |

## Architecture

```
spectra run / evaluate / serve
        |
  Core Engine (metrics_lie)
    |- Dataset Loading (CSV)
    |- Model Adapter (pickle, ONNX, PyTorch, ...)
    |- Scenario Runner (Monte Carlo trials)
    |- Metrics (27 metrics across 4 task types)
    |- Diagnostics (calibration, gaming, subgroups)
    |- Analysis (dashboard, disagreement, sensitivity)
    |- Decision Framework (scorecard, components)
    '- Artifacts (plots, reports)
```

## Development

```bash
git clone https://github.com/StrangeStorm243-bit/when-metrics-lie.git
cd when-metrics-lie
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev,web]"
pytest
```

## Documentation

Full docs: [https://strangestorm243-bit.github.io/when-metrics-lie/](https://strangestorm243-bit.github.io/when-metrics-lie/)

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
