from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from metrics_lie.scenarios.base import ScenarioContext
from metrics_lie.scenarios.registry import register_scenario


@dataclass(frozen=True)
class TemporalShiftScenario:
    """Shifts numerical features by a time-based offset (simulates temporal distribution shift)."""

    id: str = "temporal_shift"
    shift_fraction: float = 0.1

    def apply(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        rng: np.random.Generator,
        ctx: ScenarioContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        if y_score.ndim < 2:
            # 1D: shift the values
            result = y_score.astype(float).copy()
            std = float(np.std(result)) if np.std(result) > 0 else 1.0
            result += self.shift_fraction * std
            return y_true, result
        result = y_score.astype(float).copy()
        n = result.shape[0]
        shift_n = max(1, int(n * self.shift_fraction))
        result = np.roll(result, shift_n, axis=0)
        return y_true, result

    def describe(self) -> Dict[str, Any]:
        return {"id": self.id, "shift_fraction": self.shift_fraction}


def _factory(params: dict[str, Any]) -> TemporalShiftScenario:
    return TemporalShiftScenario(shift_fraction=float(params.get("shift_fraction", 0.1)))


register_scenario("temporal_shift", _factory)
