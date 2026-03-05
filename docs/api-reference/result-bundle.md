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
