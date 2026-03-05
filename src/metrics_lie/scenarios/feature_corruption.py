from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from metrics_lie.scenarios.base import ScenarioContext
from metrics_lie.scenarios.registry import register_scenario


@dataclass(frozen=True)
class FeatureCorruptionScenario:
    """Adds Gaussian noise to randomly selected feature values."""

    id: str = "feature_corruption"
    corruption_rate: float = 0.1
    noise_scale: float = 3.0

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
        mask = rng.random(size=result.shape) < self.corruption_rate
        noise = rng.normal(scale=self.noise_scale, size=result.shape)
        result[mask] += noise[mask]
        return y_true, result

    def describe(self) -> Dict[str, Any]:
        return {"id": self.id, "corruption_rate": self.corruption_rate, "noise_scale": self.noise_scale}


def _factory(params: dict[str, Any]) -> FeatureCorruptionScenario:
    return FeatureCorruptionScenario(
        corruption_rate=float(params.get("corruption_rate", 0.1)),
        noise_scale=float(params.get("noise_scale", 3.0)),
    )


register_scenario("feature_corruption", _factory)
