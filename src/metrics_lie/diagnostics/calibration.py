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


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
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


