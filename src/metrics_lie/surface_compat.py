"""Canonical surface-type routing tables and defaults.

These constants are the single source of truth for:
- Mapping spec surface_type strings to SurfaceType enums
- Which scenarios are compatible with each surface type
- The default decision threshold for binary classification
"""
from __future__ import annotations

from typing import Any, Sequence

from metrics_lie.model.surface import SurfaceType

SURFACE_TYPE_MAP: dict[str, SurfaceType] = {
    "probability": SurfaceType.PROBABILITY,
    "score": SurfaceType.SCORE,
    "label": SurfaceType.LABEL,
    "continuous": SurfaceType.CONTINUOUS,
}

# Phase 9B: scenario compatibility by surface type.
# Only applied when surface_source is set.
SCENARIO_SURFACE_COMPAT: dict[SurfaceType, set[str]] = {
    SurfaceType.PROBABILITY: {
        "label_noise",
        "score_noise",
        "class_imbalance",
        "threshold_gaming",
    },
    SurfaceType.SCORE: {"label_noise", "score_noise", "class_imbalance"},
    SurfaceType.LABEL: {"label_noise", "class_imbalance"},
    SurfaceType.CONTINUOUS: {"label_noise", "score_noise"},
}

DEFAULT_THRESHOLD: float = 0.5


def filter_compatible_scenarios(
    scenarios: Sequence[Any],
    surface_type: SurfaceType,
) -> tuple[list[Any], list[str]]:
    """Filter scenarios by surface-type compatibility.

    Only applied when surface_source is set (Phase 9B).

    Returns:
        (compatible_scenarios, skipped_scenario_ids)
    """
    allowed = SCENARIO_SURFACE_COMPAT[surface_type]
    compatible = [s for s in scenarios if s.id in allowed]
    skipped = [s.id for s in scenarios if s.id not in allowed]
    return compatible, skipped


# Task-type compatibility: which scenarios apply to each task type.
SCENARIO_TASK_COMPAT: dict[str, set[str]] = {
    "binary_classification": {"label_noise", "score_noise", "class_imbalance", "threshold_gaming"},
    "multiclass_classification": {"label_noise", "score_noise", "class_imbalance"},
    "multilabel_classification": {"label_noise", "class_imbalance"},
    "regression": {"label_noise", "score_noise"},
    "ranking": {"label_noise", "score_noise"},
}


def filter_compatible_scenarios_by_task(
    scenarios: Sequence[Any],
    task_type: str,
) -> tuple[list[Any], list[str]]:
    """Filter scenarios by task-type compatibility.

    Returns:
        (compatible_scenarios, skipped_scenario_ids)
    """
    allowed = SCENARIO_TASK_COMPAT.get(task_type, set())
    compatible = [s for s in scenarios if s.id in allowed]
    skipped = [s.id for s in scenarios if s.id not in allowed]
    return compatible, skipped
