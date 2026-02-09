from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metrics_lie.model.surface import PredictionSurface, SurfaceType


@dataclass(frozen=True)
class FailureModeReport:
    total_samples: int
    failure_samples: list[int]
    failure_reasons: list[dict]
    worst_subgroup: str | None
    summary: dict

    def to_jsonable(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "failure_samples": self.failure_samples,
            "failure_reasons": self.failure_reasons,
            "worst_subgroup": self.worst_subgroup,
            "summary": self.summary,
        }


def locate_failure_modes(
    *,
    y_true: np.ndarray,
    surface: PredictionSurface,
    metrics: list[str],
    subgroup: np.ndarray | None = None,
    top_k: int = 20,
) -> FailureModeReport:
    n = int(y_true.shape[0])
    if n == 0:
        return FailureModeReport(
            total_samples=0,
            failure_samples=[],
            failure_reasons=[],
            worst_subgroup=None,
            summary={"reason": "empty_dataset"},
        )

    # Build a simple per-sample contribution score.
    contributions = np.zeros(n, dtype=float)
    reasons: list[dict] = []

    if surface.surface_type == SurfaceType.PROBABILITY:
        prob = surface.values.astype(float)
        contributions += np.abs(prob - y_true)
        pred = (prob >= (surface.threshold or 0.5)).astype(int)
        misclassified = pred != y_true
        contributions += misclassified.astype(float)
    elif surface.surface_type == SurfaceType.LABEL:
        pred = surface.values.astype(int)
        misclassified = pred != y_true
        contributions += misclassified.astype(float)
    else:
        scores = surface.values.astype(float)
        contributions += np.abs(scores - np.mean(scores))

    top_k = min(top_k, n)
    top_indices = np.argsort(contributions)[-top_k:][::-1]

    for idx in top_indices:
        reasons.append(
            {
                "index": int(idx),
                "contribution": float(contributions[idx]),
            }
        )

    worst_subgroup = None
    summary = {
        "mean_contribution": float(np.mean(contributions)),
        "max_contribution": float(np.max(contributions)),
        "top_k": int(top_k),
    }

    if subgroup is not None and len(subgroup) == n:
        group_scores: dict[str, list[float]] = {}
        for idx, g in enumerate(subgroup):
            key = str(g)
            group_scores.setdefault(key, []).append(float(contributions[idx]))
        if group_scores:
            group_means = {k: float(np.mean(v)) for k, v in group_scores.items()}
            worst_subgroup = max(group_means, key=group_means.get)
            summary["subgroup_means"] = group_means

    return FailureModeReport(
        total_samples=n,
        failure_samples=[int(i) for i in top_indices.tolist()],
        failure_reasons=reasons,
        worst_subgroup=worst_subgroup,
        summary=summary,
    )
