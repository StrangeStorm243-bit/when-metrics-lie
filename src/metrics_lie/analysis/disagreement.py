from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metrics_lie.metrics.core import THRESHOLD_METRICS
from metrics_lie.model.surface import PredictionSurface, SurfaceType


@dataclass(frozen=True)
class MetricDisagreementResult:
    metric_a: str
    metric_b: str
    disagreement_rate: float
    disagreement_regions: list[dict]
    sample_level_disagreements: list[int] | None

    def to_jsonable(self) -> dict:
        return {
            "metric_a": self.metric_a,
            "metric_b": self.metric_b,
            "disagreement_rate": self.disagreement_rate,
            "disagreement_regions": self.disagreement_regions,
            "sample_level_disagreements": self.sample_level_disagreements,
        }


def analyze_metric_disagreements(
    *,
    y_true: np.ndarray,
    surface: PredictionSurface,
    thresholds: dict[str, float],
    metrics: list[str],
) -> list[MetricDisagreementResult]:
    if surface.surface_type not in (SurfaceType.PROBABILITY, SurfaceType.SCORE):
        return []

    threshold_metrics = [
        m for m in metrics if m in THRESHOLD_METRICS and m in thresholds
    ]
    results: list[MetricDisagreementResult] = []

    for i in range(len(threshold_metrics)):
        for j in range(i + 1, len(threshold_metrics)):
            ma = threshold_metrics[i]
            mb = threshold_metrics[j]
            ta = thresholds[ma]
            tb = thresholds[mb]
            pred_a = (surface.values >= ta).astype(int)
            pred_b = (surface.values >= tb).astype(int)
            disagree_mask = pred_a != pred_b
            disagree_indices = np.where(disagree_mask)[0].tolist()
            disagreement_rate = (
                float(np.mean(disagree_mask)) if disagree_mask.size > 0 else 0.0
            )
            results.append(
                MetricDisagreementResult(
                    metric_a=ma,
                    metric_b=mb,
                    disagreement_rate=disagreement_rate,
                    disagreement_regions=[
                        {"threshold_a": float(ta), "threshold_b": float(tb)}
                    ],
                    sample_level_disagreements=disagree_indices,
                )
            )

    return results
