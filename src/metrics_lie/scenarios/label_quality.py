from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from metrics_lie.scenarios.base import ScenarioContext
from metrics_lie.scenarios.registry import register_scenario


@dataclass(frozen=True)
class LabelQualityScenario:
    """Injects realistic label errors by flipping labels for low-confidence predictions."""

    id: str = "label_quality"
    error_rate: float = 0.1

    def apply(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        rng: np.random.Generator,
        ctx: ScenarioContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        y = y_true.copy()
        n = len(y)
        n_errors = max(1, int(n * self.error_rate))

        # Compute confidence to find low-confidence samples
        if y_score.ndim == 1:
            confidence = np.abs(y_score - 0.5)
        else:
            confidence = np.max(y_score, axis=1) - (1.0 / y_score.shape[1])
        low_conf_idx = np.argsort(confidence)[:n_errors]

        if ctx.task == "regression":
            std = float(np.std(y)) if np.std(y) > 0 else 1.0
            y = y.astype(float).copy()
            y[low_conf_idx] += rng.normal(scale=std * 0.5, size=len(low_conf_idx))
        elif ctx.task == "multiclass_classification":
            classes = np.unique(y_true)
            for idx in low_conf_idx:
                other = classes[classes != y[idx]]
                if len(other) > 0:
                    y[idx] = rng.choice(other)
        else:
            # Binary classification
            y[low_conf_idx] = 1 - y[low_conf_idx]

        return y, y_score

    def describe(self) -> Dict[str, Any]:
        return {"id": self.id, "error_rate": self.error_rate}


def _factory(params: dict[str, Any]) -> LabelQualityScenario:
    return LabelQualityScenario(error_rate=float(params.get("error_rate", 0.1)))


register_scenario("label_quality", _factory)
