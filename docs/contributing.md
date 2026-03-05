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
