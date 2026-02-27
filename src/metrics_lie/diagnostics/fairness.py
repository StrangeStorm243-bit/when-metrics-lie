"""Fairlearn-powered fairness analysis for subgroup evaluation."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def compute_fairness_report(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]],
) -> dict[str, Any]:
    """Compute per-group metrics, gaps, and fairness indicators using Fairlearn MetricFrame."""
    from fairlearn.metrics import MetricFrame

    mf = MetricFrame(
        metrics=metric_fns,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    # Extract per-group metrics
    group_metrics: dict[str, dict[str, float]] = {}
    by_group = mf.by_group
    for group_val in by_group.index:
        group_key = str(group_val)
        group_metrics[group_key] = {}
        for metric_name in metric_fns:
            val = by_group.loc[group_val, metric_name]
            group_metrics[group_key][metric_name] = float(val)

    # Compute gaps (max - min per metric)
    diff = mf.difference(method="between_groups")
    gaps: dict[str, float] = {k: float(v) for k, v in diff.items()}

    # Overall metrics
    overall = mf.overall
    overall_dict: dict[str, float] = {k: float(v) for k, v in overall.items()}

    result: dict[str, Any] = {
        "group_metrics": group_metrics,
        "gaps": gaps,
        "overall": overall_dict,
    }

    # Demographic parity difference
    unique_preds = np.unique(y_pred)
    if len(unique_preds) == 2:
        try:
            from fairlearn.metrics import demographic_parity_difference
            result["demographic_parity_difference"] = float(
                demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
            )
        except Exception:
            result["demographic_parity_difference"] = None
    else:
        # For multiclass, compute max selection rate difference across classes
        groups = np.unique(sensitive_features)
        max_diff = 0.0
        for c in unique_preds:
            rates = []
            for g in groups:
                mask = sensitive_features == g
                rates.append(float(np.mean(y_pred[mask] == c)))
            max_diff = max(max_diff, max(rates) - min(rates))
        result["demographic_parity_difference"] = float(max_diff)

    return result
