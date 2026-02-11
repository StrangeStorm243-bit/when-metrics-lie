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
    },
    {
        "id": "pr_auc",
        "name": "PR-AUC",
        "description": "Area under the Precision-Recall curve",
        "requires_surface": ["probability", "score"],
    },
    {
        "id": "accuracy",
        "name": "Accuracy",
        "description": "Classification accuracy",
        "requires_surface": ["label", "probability"],
    },
    {
        "id": "logloss",
        "name": "Log Loss",
        "description": "Logarithmic loss (cross-entropy)",
        "requires_surface": ["probability"],
    },
    {
        "id": "f1",
        "name": "F1 Score",
        "description": "Harmonic mean of precision and recall",
        "requires_surface": ["label", "probability"],
    },
    {
        "id": "precision",
        "name": "Precision",
        "description": "Positive predictive value",
        "requires_surface": ["label", "probability"],
    },
    {
        "id": "recall",
        "name": "Recall",
        "description": "True positive rate",
        "requires_surface": ["label", "probability"],
    },
    {
        "id": "matthews_corrcoef",
        "name": "Matthews Corrcoef",
        "description": "Matthews correlation coefficient",
        "requires_surface": ["label", "probability"],
    },
    {
        "id": "brier_score",
        "name": "Brier Score",
        "description": "Mean squared error of probabilistic predictions",
        "requires_surface": ["probability"],
    },
    {
        "id": "ece",
        "name": "Expected Calibration Error",
        "description": "Calibration error across probability bins",
        "requires_surface": ["probability"],
    },
]

# Stress suite presets - will map to decision profiles and scenario configurations
STRESS_SUITE_PRESETS = [
    {
        "id": "balanced",
        "name": "Balanced Suite",
        "description": "Balanced evaluation with worst-case scenario aggregation",
    },
    {
        "id": "risk_averse",
        "name": "Risk-Averse Suite",
        "description": "Heavy penalties for calibration and subgroup regressions",
    },
    {
        "id": "performance_first",
        "name": "Performance-First Suite",
        "description": "Emphasizes mean metric improvement with relaxed constraints",
    },
    {
        "id": "comprehensive",
        "name": "Comprehensive Suite",
        "description": "Full scenario coverage with all diagnostics enabled",
    },
]
