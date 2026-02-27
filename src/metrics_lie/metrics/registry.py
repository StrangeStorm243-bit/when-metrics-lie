from __future__ import annotations

from dataclasses import dataclass

from metrics_lie.model.surface import SurfaceType


@dataclass(frozen=True)
class MetricRequirement:
    metric_id: str
    requires_surface: set[SurfaceType]
    requires_labels: bool
    min_samples: int
    requires_both_classes: bool
    task_types: frozenset[str] | None = None
    higher_is_better: bool = True


# --- Binary classification metrics ---
METRIC_REQUIREMENTS: list[MetricRequirement] = [
    MetricRequirement(
        metric_id="accuracy",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"binary_classification"}),
    ),
    MetricRequirement(
        metric_id="auc",
        requires_surface={SurfaceType.PROBABILITY, SurfaceType.SCORE},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"binary_classification"}),
    ),
    MetricRequirement(
        metric_id="pr_auc",
        requires_surface={SurfaceType.PROBABILITY, SurfaceType.SCORE},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"binary_classification"}),
    ),
    MetricRequirement(
        metric_id="f1",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"binary_classification"}),
    ),
    MetricRequirement(
        metric_id="precision",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"binary_classification"}),
    ),
    MetricRequirement(
        metric_id="recall",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"binary_classification"}),
    ),
    MetricRequirement(
        metric_id="logloss",
        requires_surface={SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"binary_classification"}),
        higher_is_better=False,
    ),
    MetricRequirement(
        metric_id="brier_score",
        requires_surface={SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"binary_classification"}),
        higher_is_better=False,
    ),
    MetricRequirement(
        metric_id="ece",
        requires_surface={SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"binary_classification"}),
        higher_is_better=False,
    ),
    MetricRequirement(
        metric_id="matthews_corrcoef",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"binary_classification"}),
    ),
    # --- Multiclass classification metrics ---
    MetricRequirement(
        metric_id="macro_f1",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"multiclass_classification"}),
    ),
    MetricRequirement(
        metric_id="weighted_f1",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"multiclass_classification"}),
    ),
    MetricRequirement(
        metric_id="macro_precision",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"multiclass_classification"}),
    ),
    MetricRequirement(
        metric_id="macro_recall",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"multiclass_classification"}),
    ),
    MetricRequirement(
        metric_id="macro_auc",
        requires_surface={SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"multiclass_classification"}),
    ),
    MetricRequirement(
        metric_id="cohens_kappa",
        requires_surface={SurfaceType.LABEL, SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"multiclass_classification"}),
    ),
    MetricRequirement(
        metric_id="top_k_accuracy",
        requires_surface={SurfaceType.PROBABILITY},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=True,
        task_types=frozenset({"multiclass_classification"}),
    ),
    # --- Regression metrics ---
    MetricRequirement(
        metric_id="mae",
        requires_surface={SurfaceType.CONTINUOUS},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=False,
        task_types=frozenset({"regression"}),
        higher_is_better=False,
    ),
    MetricRequirement(
        metric_id="mse",
        requires_surface={SurfaceType.CONTINUOUS},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=False,
        task_types=frozenset({"regression"}),
        higher_is_better=False,
    ),
    MetricRequirement(
        metric_id="rmse",
        requires_surface={SurfaceType.CONTINUOUS},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=False,
        task_types=frozenset({"regression"}),
        higher_is_better=False,
    ),
    MetricRequirement(
        metric_id="r2",
        requires_surface={SurfaceType.CONTINUOUS},
        requires_labels=True,
        min_samples=2,
        requires_both_classes=False,
        task_types=frozenset({"regression"}),
    ),
    MetricRequirement(
        metric_id="max_error",
        requires_surface={SurfaceType.CONTINUOUS},
        requires_labels=True,
        min_samples=1,
        requires_both_classes=False,
        task_types=frozenset({"regression"}),
        higher_is_better=False,
    ),
]

METRIC_DIRECTION: dict[str, bool] = {
    r.metric_id: r.higher_is_better for r in METRIC_REQUIREMENTS
}
