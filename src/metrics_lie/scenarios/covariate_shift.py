from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from metrics_lie.scenarios.base import ScenarioContext
from metrics_lie.scenarios.registry import register_scenario


@dataclass(frozen=True)
class CovariateShiftScenario:
    """Shifts feature distributions by adding per-feature offsets."""

    id: str = "covariate_shift"
    shift_scale: float = 1.0

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
        shifts = rng.normal(scale=self.shift_scale, size=result.shape[1])
        result += shifts[np.newaxis, :]
        return y_true, result

    def describe(self) -> Dict[str, Any]:
        return {"id": self.id, "shift_scale": self.shift_scale}


def _factory(params: dict[str, Any]) -> CovariateShiftScenario:
    return CovariateShiftScenario(shift_scale=float(params.get("shift_scale", 1.0)))


register_scenario("covariate_shift", _factory)
