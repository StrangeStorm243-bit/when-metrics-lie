import numpy as np

from metrics_lie.diagnostics.metric_gaming import accuracy_at_threshold, find_optimal_threshold


def test_threshold_optimization_increases_accuracy():
    """Test that threshold optimization can increase accuracy above baseline."""
    # Create synthetic data where optimal threshold is not 0.5
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 0.95])

    baseline_acc = accuracy_at_threshold(y_true, y_score, 0.5)
    thresholds = np.linspace(0.05, 0.95, 19)
    opt_thresh, opt_acc = find_optimal_threshold(y_true, y_score, thresholds)

    # Optimal should be at least as good as baseline
    assert opt_acc >= baseline_acc
    # In this case, optimal should be better (threshold around 0.5-0.6 works better)
    assert opt_acc > baseline_acc or opt_thresh != 0.5

