from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from metrics_lie.metrics.core import THRESHOLD_METRICS
from metrics_lie.surface_compat import DEFAULT_THRESHOLD


def group_indices(subgroup: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Return a dict mapping group string values to boolean masks.
    """
    unique_groups = np.unique(subgroup)
    result: Dict[str, np.ndarray] = {}
    for g in unique_groups:
        # Convert to string for stable keys
        key = str(g)
        result[key] = subgroup == g
    return result


def compute_group_sizes(subgroup: np.ndarray) -> Dict[str, int]:
    """
    Return a dict mapping group string values to their sizes.
    """
    unique_groups, counts = np.unique(subgroup, return_counts=True)
    result: Dict[str, int] = {}
    for g, cnt in zip(unique_groups, counts):
        key = str(g)
        result[key] = int(cnt)
    return result


def safe_metric_for_group(
    metric_name: str,
    metric_fn: Callable[..., float],
    y_true_g: np.ndarray,
    y_score_g: np.ndarray,
) -> float | None:
    """
    Compute metric for a group, returning None if invalid (too small, single class for AUC, etc.).
    """
    if len(y_true_g) < 2:
        return None

    # AUC variants need both classes present
    if metric_name in ("auc", "macro_auc", "pr_auc"):
        if len(np.unique(y_true_g)) < 2:
            return None

    try:
        if metric_name in THRESHOLD_METRICS:
            return metric_fn(y_true_g, y_score_g, threshold=DEFAULT_THRESHOLD)
        else:
            return metric_fn(y_true_g, y_score_g)
    except (ValueError, Exception):
        # Handle any metric-specific errors gracefully
        return None
