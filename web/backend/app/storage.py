"""In-memory preset registries for Phase 3.1.

These will later map to Spectra core registry/CLI presets.
For now, they are simple Python data structures.
"""

# Metric presets - will map to metrics_lie.metrics.core.METRICS
METRIC_PRESETS = [
    {
        "id": "auc",
        "name": "AUC-ROC",
        "description": "Area under the ROC curve",
        "requires_surface": ["probability", "score"],
        "task_types": ["binary_classification"],
    },
    {
        "id": "pr_auc",
        "name": "PR-AUC",
        "description": "Area under the Precision-Recall curve",
        "requires_surface": ["probability", "score"],
        "task_types": ["binary_classification"],
    },
    {
        "id": "accuracy",
        "name": "Accuracy",
        "description": "Classification accuracy",
        "requires_surface": ["label", "probability"],
        "task_types": ["binary_classification", "multiclass_classification"],
    },
    {
        "id": "logloss",
        "name": "Log Loss",
        "description": "Logarithmic loss (cross-entropy)",
        "requires_surface": ["probability"],
        "task_types": ["binary_classification", "multiclass_classification"],
    },
    {
        "id": "f1",
        "name": "F1 Score",
        "description": "Harmonic mean of precision and recall",
        "requires_surface": ["label", "probability"],
        "task_types": ["binary_classification", "multiclass_classification"],
    },
    {
        "id": "precision",
        "name": "Precision",
        "description": "Positive predictive value",
        "requires_surface": ["label", "probability"],
        "task_types": ["binary_classification", "multiclass_classification"],
    },
    {
        "id": "recall",
        "name": "Recall",
        "description": "True positive rate",
        "requires_surface": ["label", "probability"],
        "task_types": ["binary_classification", "multiclass_classification"],
    },
    {
        "id": "matthews_corrcoef",
        "name": "Matthews Corrcoef",
        "description": "Matthews correlation coefficient",
        "requires_surface": ["label", "probability"],
        "task_types": ["binary_classification", "multiclass_classification"],
    },
    {
        "id": "brier_score",
        "name": "Brier Score",
        "description": "Mean squared error of probabilistic predictions",
        "requires_surface": ["probability"],
        "task_types": ["binary_classification"],
    },
    {
        "id": "ece",
        "name": "Expected Calibration Error",
        "description": "Calibration error across probability bins",
        "requires_surface": ["probability"],
        "task_types": ["binary_classification", "multiclass_classification"],
    },
    # Regression metrics
    {
        "id": "mae",
        "name": "Mean Absolute Error",
        "description": "Average absolute prediction error",
        "requires_surface": ["continuous"],
        "task_types": ["regression"],
    },
    {
        "id": "mse",
        "name": "Mean Squared Error",
        "description": "Average squared prediction error",
        "requires_surface": ["continuous"],
        "task_types": ["regression"],
    },
    {
        "id": "rmse",
        "name": "Root Mean Squared Error",
        "description": "Square root of average squared error",
        "requires_surface": ["continuous"],
        "task_types": ["regression"],
    },
    {
        "id": "r_squared",
        "name": "R-Squared",
        "description": "Coefficient of determination",
        "requires_surface": ["continuous"],
        "task_types": ["regression"],
    },
    # Multiclass-specific
    {
        "id": "weighted_f1",
        "name": "Weighted F1",
        "description": "Weighted average F1 across classes",
        "requires_surface": ["label", "probability"],
        "task_types": ["multiclass_classification"],
    },
    {
        "id": "macro_f1",
        "name": "Macro F1",
        "description": "Unweighted average F1 across classes",
        "requires_surface": ["label", "probability"],
        "task_types": ["multiclass_classification"],
    },
    {
        "id": "cohens_kappa",
        "name": "Cohen's Kappa",
        "description": "Agreement beyond chance",
        "requires_surface": ["label", "probability"],
        "task_types": ["multiclass_classification"],
    },
]

# Stress suite presets - will map to decision profiles and scenario configurations
STRESS_SUITE_PRESETS = [
    {
        "id": "balanced",
        "name": "Balanced Suite",
        "description": "Balanced evaluation with worst-case scenario aggregation",
        "scenarios": ["label_noise", "score_noise", "class_imbalance", "threshold_gaming"],
    },
    {
        "id": "risk_averse",
        "name": "Risk-Averse Suite",
        "description": "Heavy penalties for calibration and subgroup regressions",
        "scenarios": ["label_noise", "score_noise", "class_imbalance", "threshold_gaming"],
    },
    {
        "id": "performance_first",
        "name": "Performance-First Suite",
        "description": "Emphasizes mean metric improvement with relaxed constraints",
        "scenarios": ["label_noise", "score_noise"],
    },
    {
        "id": "comprehensive",
        "name": "Comprehensive Suite",
        "description": "Full scenario coverage with all diagnostics enabled",
        "scenarios": ["label_noise", "score_noise", "class_imbalance", "threshold_gaming"],
    },
]
