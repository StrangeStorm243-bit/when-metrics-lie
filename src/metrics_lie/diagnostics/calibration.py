from __future__ import annotations

import numpy as np


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Brier score for binary outcomes: mean((p - y)^2).
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_score, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    return float(np.mean((p - y) ** 2))


def multiclass_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Multiclass Brier score: mean(sum_k (p_k - y_k)^2) where y is one-hot."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_proba, dtype=float)
    one_hot = np.zeros_like(p)
    one_hot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> float:
    """
    Expected Calibration Error (ECE) using equal-width bins over [0, 1].
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_score, dtype=float)
    p = np.clip(p, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize returns 1..n_bins; subtract 1 -> 0..n_bins-1
    bin_idx = np.digitize(p, bins[1:-1], right=False)

    ece = 0.0
    n = float(p.size)
    for b in range(n_bins):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def per_class_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict[int, float]:
    """Per-class ECE for multiclass problems.

    Computes ECE for each class in a one-vs-rest manner.

    Args:
        y_true: Integer class labels (n_samples,)
        y_proba: Probability matrix (n_samples, n_classes)
        n_bins: Number of calibration bins

    Returns:
        Dict mapping class index to ECE value.
    """
    if y_proba.ndim != 2:
        raise ValueError(f"per_class_ece requires 2D probability matrix, got shape {y_proba.shape}")
    n_classes = y_proba.shape[1]
    result: dict[int, float] = {}
    for c in range(n_classes):
        binary_true = (y_true == c).astype(int)
        binary_proba = y_proba[:, c]
        result[c] = float(expected_calibration_error(binary_true, binary_proba, n_bins=n_bins))
    return result


def multiclass_ece(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> float:
    """Top-label Expected Calibration Error for multiclass."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_proba, dtype=float)
    top_conf = np.max(p, axis=1)
    top_pred = np.argmax(p, axis=1)
    correct = (top_pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(top_conf, bins[1:-1], right=False)
    ece = 0.0
    n = float(len(y))
    for b in range(n_bins):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        avg_conf = float(top_conf[mask].mean())
        avg_acc = float(correct[mask].mean())
        ece += (mask.sum() / n) * abs(avg_acc - avg_conf)
    return float(ece)
