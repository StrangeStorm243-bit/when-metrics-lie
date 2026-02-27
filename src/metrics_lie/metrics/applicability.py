from __future__ import annotations

from dataclasses import dataclass

from metrics_lie.model.surface import SurfaceType
from .registry import METRIC_REQUIREMENTS, MetricRequirement


@dataclass(frozen=True)
class DatasetProperties:
    n_samples: int
    n_positive: int
    n_negative: int
    has_subgroups: bool
    positive_rate: float
    n_classes: int | None = None


@dataclass(frozen=True)
class ApplicableMetricSet:
    task_type: str
    surface_type: SurfaceType
    metrics: list[str]
    excluded: list[tuple[str, str]]
    reasoning_trace: list[str]
    warnings: list[str]


class MetricResolver:
    def __init__(self, requirements: list[MetricRequirement] | None = None) -> None:
        self._requirements = requirements or METRIC_REQUIREMENTS

    def resolve(
        self,
        *,
        task_type: str,
        surface_type: SurfaceType,
        dataset_props: DatasetProperties,
    ) -> ApplicableMetricSet:
        metrics: list[str] = []
        excluded: list[tuple[str, str]] = []
        trace: list[str] = []
        warnings: list[str] = []

        trace.append(f"task_type={task_type}")
        trace.append(f"surface_type={surface_type.value}")
        trace.append(
            f"dataset: n={dataset_props.n_samples}, pos={dataset_props.n_positive}, "
            f"neg={dataset_props.n_negative}, pos_rate={dataset_props.positive_rate:.4f}"
        )

        for req in self._requirements:
            if req.task_types is not None and task_type not in req.task_types:
                excluded.append((req.metric_id, f"not applicable to task_type={task_type}"))
                continue
            if surface_type not in req.requires_surface:
                excluded.append(
                    (
                        req.metric_id,
                        f"requires surface {sorted([s.value for s in req.requires_surface])}",
                    )
                )
                continue
            if (
                task_type == "binary_classification"
                and req.requires_both_classes
                and (dataset_props.n_positive == 0 or dataset_props.n_negative == 0)
            ):
                excluded.append((req.metric_id, "requires both classes"))
                continue
            if dataset_props.n_samples < req.min_samples:
                excluded.append(
                    (req.metric_id, f"requires at least {req.min_samples} samples")
                )
                continue
            metrics.append(req.metric_id)

        if dataset_props.n_samples < 30:
            warnings.append("low_sample_warning")
        is_classification = task_type in (
            "binary_classification",
            "multiclass_classification",
            "multilabel_classification",
        )
        if is_classification and (
            dataset_props.positive_rate < 0.05 or dataset_props.positive_rate > 0.95
        ):
            warnings.append("severe_imbalance_warning")
            if (
                "pr_auc" not in metrics
                and surface_type in {SurfaceType.PROBABILITY, SurfaceType.SCORE}
                and dataset_props.n_positive > 0
                and dataset_props.n_negative > 0
                and any(r.metric_id == "pr_auc" for r in self._requirements)
            ):
                metrics.append("pr_auc")

        return ApplicableMetricSet(
            task_type=task_type,
            surface_type=surface_type,
            metrics=metrics,
            excluded=excluded,
            reasoning_trace=trace,
            warnings=warnings,
        )
