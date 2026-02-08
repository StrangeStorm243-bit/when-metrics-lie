from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from metrics_lie.model.surface import SurfaceType


MetricId = Literal[
    "accuracy",
    "auc",
    "pr_auc",
    "f1",
    "precision",
    "recall",
    "logloss",
    "brier_score",
    "ece",
    "matthews_corrcoef",
]


@dataclass(frozen=True)
class MetricRequirement:
    metric_id: MetricId
    requires_surface: set[SurfaceType]
    requires_labels: bool
    min_samples: int
    requires_both_classes: bool


METRIC_REQUIREMENTS: list[MetricRequirement] = [
    MetricRequirement(
        metric_id="accuracy",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
    ),
    MetricRequirement(
        metric_id="auc",
        requires_surface={SurfaceType.PROBABILITY, SurfaceType.SCORE},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
    ),
    MetricRequirement(
        metric_id="pr_auc",
        requires_surface={SurfaceType.PROBABILITY, SurfaceType.SCORE},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
    ),
    MetricRequirement(
        metric_id="f1",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
    ),
    MetricRequirement(
        metric_id="precision",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
    ),
    MetricRequirement(
        metric_id="recall",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
    ),
    MetricRequirement(
        metric_id="logloss",
        requires_surface={SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
    ),
    MetricRequirement(
        metric_id="brier_score",
        requires_surface={SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
    ),
    MetricRequirement(
        metric_id="ece",
        requires_surface={SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
    ),
    MetricRequirement(
        metric_id="matthews_corrcoef",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
    ),
]
