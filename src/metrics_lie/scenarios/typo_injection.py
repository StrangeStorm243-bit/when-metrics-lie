from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from metrics_lie.scenarios.base import ScenarioContext
from metrics_lie.scenarios.registry import register_scenario


@dataclass(frozen=True)
class TypoInjectionScenario:
    """Character-level perturbation for text data."""

    id: str = "typo_injection"
    typo_rate: float = 0.1

    def apply(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        rng: np.random.Generator,
        ctx: ScenarioContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        if y_score.dtype.kind not in ("U", "O"):  # Not string type
            return y_true, y_score
        result = y_score.copy()
        for i in range(len(result)):
            if isinstance(result[i], str) and len(result[i]) > 0:
                chars = list(result[i])
                for j in range(len(chars)):
                    if rng.random() < self.typo_rate:
                        chars[j] = rng.choice(list("abcdefghijklmnopqrstuvwxyz"))
                result[i] = "".join(chars)
        return y_true, result

    def describe(self) -> Dict[str, Any]:
        return {"id": self.id, "typo_rate": self.typo_rate}


def _factory(params: dict[str, Any]) -> TypoInjectionScenario:
    return TypoInjectionScenario(typo_rate=float(params.get("typo_rate", 0.1)))


register_scenario("typo_injection", _factory)
