from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def metric_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # roc_auc_score expects scores (probabilities ok)
    return float(roc_auc_score(y_true, y_score))


def metric_logloss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # log_loss expects probabilities; enforce numerical safety
    eps = 1e-15
    y_score = np.clip(y_score, eps, 1 - eps)
    return float(log_loss(y_true, y_score))


def metric_accuracy(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (y_score >= threshold).astype(int)
    return float(accuracy_score(y_true, y_pred))


METRICS: Dict[str, Callable[..., float]] = {
    "auc": metric_auc,
    "logloss": metric_logloss,
    "accuracy": metric_accuracy,
}
