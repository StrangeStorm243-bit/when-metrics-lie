from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from metrics_lie.scenarios.base import ScenarioContext
from metrics_lie.scenarios.registry import register_scenario


@dataclass(frozen=True)
class ThresholdGamingScenario:
    id: str = "threshold_gaming"
    grid_size: int = 101

    def apply(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        rng: np.random.Generator,
        ctx: ScenarioContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.grid_size < 5:
            raise ValueError("threshold_gaming.grid_size must be >= 5")
        # Identity transform for now (marker scenario)
        return y_true, y_score

    def describe(self) -> Dict[str, Any]:
        return {"id": self.id, "grid_size": self.grid_size}


def _factory(params: dict[str, Any]) -> ThresholdGamingScenario:
    grid_size = int(params.get("grid_size", 101))
    return ThresholdGamingScenario(grid_size=grid_size)


register_scenario("threshold_gaming", _factory)
