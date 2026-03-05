from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from metrics_lie.scenarios.base import ScenarioContext
from metrics_lie.scenarios.registry import register_scenario


@dataclass(frozen=True)
class MissingFeaturesScenario:
    """Randomly drops feature values by replacing them with NaN."""

    id: str = "missing_features"
    drop_rate: float = 0.2

    def apply(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        rng: np.random.Generator,
        ctx: ScenarioContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        if y_score.ndim < 2:
            return y_true, y_score
        result = y_score.astype(float).copy()
        mask = rng.random(size=result.shape) < self.drop_rate
        result[mask] = np.nan
        return y_true, result

    def describe(self) -> Dict[str, Any]:
        return {"id": self.id, "drop_rate": self.drop_rate}


def _factory(params: dict[str, Any]) -> MissingFeaturesScenario:
    return MissingFeaturesScenario(drop_rate=float(params.get("drop_rate", 0.2)))


register_scenario("missing_features", _factory)
