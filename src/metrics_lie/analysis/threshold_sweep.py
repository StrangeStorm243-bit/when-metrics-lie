from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metrics_lie.diagnostics.calibration import brier_score, expected_calibration_error
from metrics_lie.metrics.core import (
    metric_accuracy,
    metric_f1,
    metric_logloss,
    metric_matthews_corrcoef,
    metric_precision,
    metric_pr_auc,
    metric_recall,
    metric_auc,
)
from metrics_lie.model.surface import PredictionSurface, SurfaceType


THRESHOLD_METRICS = {
    "accuracy": metric_accuracy,
    "f1": metric_f1,
    "precision": metric_precision,
    "recall": metric_recall,
    "matthews_corrcoef": metric_matthews_corrcoef,
}

SCORE_METRICS = {
    "auc": metric_auc,
    "pr_auc": metric_pr_auc,
    "logloss": metric_logloss,
    "brier_score": brier_score,
    "ece": expected_calibration_error,
}


@dataclass(frozen=True)
class ThresholdSweepResult:
    thresholds: np.ndarray
    metric_curves: dict[str, np.ndarray]
    optimal_thresholds: dict[str, float]
    crossover_points: list[dict]

    def to_jsonable(self) -> dict:
        return {
            "thresholds": self.thresholds.tolist(),
            "metric_curves": {k: v.tolist() for k, v in self.metric_curves.items()},
            "optimal_thresholds": self.optimal_thresholds,
            "crossover_points": self.crossover_points,
        }


def run_threshold_sweep(
    *,
    y_true: np.ndarray,
    surface: PredictionSurface,
    metrics: list[str],
    n_points: int = 101,
) -> ThresholdSweepResult:
    if surface.surface_type != SurfaceType.PROBABILITY:
        raise ValueError("threshold sweep requires probability surface")

    thresholds = np.linspace(0.0, 1.0, n_points)
    curves: dict[str, np.ndarray] = {}
    optimal_thresholds: dict[str, float] = {}

    for metric_id in metrics:
        if metric_id in THRESHOLD_METRICS:
            fn = THRESHOLD_METRICS[metric_id]
            vals = [fn(y_true, surface.values, threshold=t) for t in thresholds]
            curves[metric_id] = np.array(vals, dtype=float)
            best_idx = int(np.argmax(curves[metric_id]))
            optimal_thresholds[metric_id] = float(thresholds[best_idx])
        elif metric_id in SCORE_METRICS:
            if metric_id == "ece":
                v = SCORE_METRICS[metric_id](y_true, surface.values, n_bins=10)
            else:
                v = SCORE_METRICS[metric_id](y_true, surface.values)
            curves[metric_id] = np.full_like(thresholds, float(v), dtype=float)
            optimal_thresholds[metric_id] = float(surface.threshold or 0.5)

    crossover_points: list[dict] = []
    metric_ids = [m for m in metrics if m in THRESHOLD_METRICS]
    for i in range(len(metric_ids)):
        for j in range(i + 1, len(metric_ids)):
            a = metric_ids[i]
            b = metric_ids[j]
            ta = optimal_thresholds.get(a)
            tb = optimal_thresholds.get(b)
            if ta is None or tb is None:
                continue
            if abs(ta - tb) > 1e-9:
                crossover_points.append(
                    {"metric_a": a, "metric_b": b, "threshold_a": ta, "threshold_b": tb}
                )

    return ThresholdSweepResult(
        thresholds=thresholds,
        metric_curves=curves,
        optimal_thresholds=optimal_thresholds,
        crossover_points=crossover_points,
    )
