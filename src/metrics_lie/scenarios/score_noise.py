from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from metrics_lie.scenarios.base import ScenarioContext
from metrics_lie.scenarios.registry import register_scenario


@dataclass(frozen=True)
class ScoreNoiseScenario:
    id: str = "score_noise"
    sigma: float = 0.05  # std of gaussian noise added to scores

    def apply(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        rng: np.random.Generator,
        ctx: ScenarioContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.sigma < 0:
            raise ValueError("score_noise.sigma must be >= 0")
        s = y_score.astype(float).copy()
        s = s + rng.normal(loc=0.0, scale=self.sigma, size=s.shape[0])
        if ctx.surface_type == "probability":
            s = np.clip(s, 0.0, 1.0)
        return y_true, s

    def describe(self) -> Dict[str, Any]:
        return {"id": self.id, "sigma": self.sigma}


def _factory(params: dict[str, Any]) -> ScoreNoiseScenario:
    sigma = float(params.get("sigma", 0.05))
    return ScoreNoiseScenario(sigma=sigma)


register_scenario("score_noise", _factory)
