"""Catalog helpers for listing available metrics, scenarios, and model formats."""
from __future__ import annotations

from metrics_lie.metrics.registry import METRIC_REQUIREMENTS
from metrics_lie.surface_compat import SCENARIO_TASK_COMPAT


def list_metrics(task: str | None = None) -> list[str]:
    """List available metric IDs, optionally filtered by task type."""
    result = []
    for req in METRIC_REQUIREMENTS:
        if task is None:
            result.append(req.metric_id)
        elif req.task_types is None or task in req.task_types:
            result.append(req.metric_id)
    return sorted(result)


def list_scenarios(task: str | None = None) -> list[str]:
    """List available scenario IDs, optionally filtered by task type.

    SCENARIO_TASK_COMPAT maps task_type -> set of scenario IDs.
    When task is None we return all unique scenario IDs across every task.
    """
    if task is None:
        all_scenarios: set[str] = set()
        for scenarios in SCENARIO_TASK_COMPAT.values():
            all_scenarios.update(scenarios)
        return sorted(all_scenarios)
    return sorted(SCENARIO_TASK_COMPAT.get(task, set()))


def list_model_formats() -> list[str]:
    """List supported model format identifiers."""
    return [
        "pickle",
        "onnx",
        "xgboost",
        "lightgbm",
        "catboost",
        "http",
    ]
