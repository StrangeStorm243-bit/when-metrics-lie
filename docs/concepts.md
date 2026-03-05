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
