from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from metrics_lie.diagnostics.calibration import brier_score, expected_calibration_error


def metric_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # roc_auc_score expects scores (probabilities ok)
    return float(roc_auc_score(y_true, y_score))


def metric_logloss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # log_loss expects probabilities; enforce numerical safety
    eps = 1e-15
    y_score = np.clip(y_score, eps, 1 - eps)
    return float(log_loss(y_true, y_score))


def metric_accuracy(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> float:
    y_pred = (y_score >= threshold).astype(int)
    return float(accuracy_score(y_true, y_pred))


def metric_f1(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (y_score >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


def metric_precision(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> float:
    y_pred = (y_score >= threshold).astype(int)
    return float(precision_score(y_true, y_pred, zero_division=0))


def metric_recall(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> float:
    y_pred = (y_score >= threshold).astype(int)
    return float(recall_score(y_true, y_pred, zero_division=0))


def metric_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def metric_matthews_corrcoef(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> float:
    y_pred = (y_score >= threshold).astype(int)
    return float(matthews_corrcoef(y_true, y_pred))


def metric_brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(brier_score(y_true, y_score))


def metric_ece(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(expected_calibration_error(y_true, y_score, n_bins=10))


METRICS: Dict[str, Callable[..., float]] = {
    "auc": metric_auc,
    "logloss": metric_logloss,
    "accuracy": metric_accuracy,
    "f1": metric_f1,
    "precision": metric_precision,
    "recall": metric_recall,
    "pr_auc": metric_pr_auc,
    "matthews_corrcoef": metric_matthews_corrcoef,
    "brier_score": metric_brier_score,
    "ece": metric_ece,
}
