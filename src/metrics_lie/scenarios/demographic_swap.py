from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from metrics_lie.scenarios.base import ScenarioContext
from metrics_lie.scenarios.registry import register_scenario

_SWAPS: dict[str, str] = {
    "male": "female",
    "female": "male",
    "man": "woman",
    "woman": "man",
    "he": "she",
    "she": "he",
    "his": "her",
    "her": "his",
    "boy": "girl",
    "girl": "boy",
}


@dataclass(frozen=True)
class DemographicSwapScenario:
    """Swaps gendered/demographic terms in text or categorical features."""

    id: str = "demographic_swap"

    def apply(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        rng: np.random.Generator,
        ctx: ScenarioContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        if y_score.dtype.kind not in ("U", "O"):
            return y_true, y_score
        result = y_score.copy()
        for i in range(len(result)):
            if isinstance(result[i], str):
                words = result[i].split()
                swapped = [_SWAPS.get(w.lower(), w) for w in words]
                result[i] = " ".join(swapped)
        return y_true, result

    def describe(self) -> Dict[str, Any]:
        return {"id": self.id}


def _factory(params: dict[str, Any]) -> DemographicSwapScenario:
    return DemographicSwapScenario()


register_scenario("demographic_swap", _factory)
