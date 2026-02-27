from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
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


# --- Multiclass metric functions ---


def metric_macro_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro-averaged F1. y_score = predicted class labels."""
    y_pred = y_score.astype(int) if y_score.ndim == 1 else np.argmax(y_score, axis=1)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def metric_weighted_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Weighted-average F1. y_score = predicted class labels."""
    y_pred = y_score.astype(int) if y_score.ndim == 1 else np.argmax(y_score, axis=1)
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def metric_macro_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro-averaged precision. y_score = predicted class labels."""
    y_pred = y_score.astype(int) if y_score.ndim == 1 else np.argmax(y_score, axis=1)
    return float(precision_score(y_true, y_pred, average="macro", zero_division=0))


def metric_macro_recall(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro-averaged recall. y_score = predicted class labels."""
    y_pred = y_score.astype(int) if y_score.ndim == 1 else np.argmax(y_score, axis=1)
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))


def metric_macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro-averaged AUC via One-vs-Rest. y_score = probability matrix (n, K)."""
    return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))


def metric_cohens_kappa(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Cohen's kappa coefficient. y_score = predicted class labels."""
    y_pred = y_score.astype(int) if y_score.ndim == 1 else np.argmax(y_score, axis=1)
    return float(cohen_kappa_score(y_true, y_pred))


def metric_top_k_accuracy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Top-2 accuracy. y_score = probability matrix (n, K)."""
    k = min(2, y_score.shape[1]) if y_score.ndim == 2 else 1
    if k <= 1 or y_score.ndim == 1:
        y_pred = y_score.astype(int) if y_score.ndim == 1 else np.argmax(y_score, axis=1)
        return float(accuracy_score(y_true, y_pred))
    return float(top_k_accuracy_score(y_true, y_score, k=k))


# Metric category sets (canonical source of truth).
# Threshold metrics require a decision threshold to produce binary predictions.
THRESHOLD_METRICS: set[str] = {"accuracy", "f1", "precision", "recall", "matthews_corrcoef"}
# Calibration metrics measure probability calibration quality.
CALIBRATION_METRICS: set[str] = {"brier_score", "ece"}
# Ranking metrics evaluate score ordering without a threshold.
RANKING_METRICS: set[str] = {"auc", "pr_auc", "logloss"}
# Multiclass metrics (no threshold -- use argmax or probability matrix directly).
MULTICLASS_METRICS: set[str] = {
    "macro_f1", "weighted_f1", "macro_precision", "macro_recall",
    "macro_auc", "cohens_kappa", "top_k_accuracy",
}


def compute_metric(
    metric_id: str,
    metric_fn: Callable[..., float],
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> float:
    """Call a metric function with the correct signature for its category.

    Threshold metrics receive an explicit ``threshold`` parameter.
    All other metrics are called with just (y_true, y_score).
    """
    if metric_id in THRESHOLD_METRICS:
        return float(metric_fn(y_true, y_score, threshold=threshold))
    return float(metric_fn(y_true, y_score))


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
    "macro_f1": metric_macro_f1,
    "weighted_f1": metric_weighted_f1,
    "macro_precision": metric_macro_precision,
    "macro_recall": metric_macro_recall,
    "macro_auc": metric_macro_auc,
    "cohens_kappa": metric_cohens_kappa,
    "top_k_accuracy": metric_top_k_accuracy,
}
