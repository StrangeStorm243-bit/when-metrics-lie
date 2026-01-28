from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score


def accuracy_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    """Compute accuracy at a specific threshold."""
    y_pred = (y_score >= threshold).astype(int)
    return float(accuracy_score(y_true, y_pred))


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, float]:
    """
    Find the threshold that maximizes accuracy.
    
    Returns:
        (best_threshold, best_accuracy)
    """
    best_acc = -1.0
    best_thresh = 0.5
    
    for thresh in thresholds:
        acc = accuracy_at_threshold(y_true, y_score, thresh)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    
    return (float(best_thresh), float(best_acc))

