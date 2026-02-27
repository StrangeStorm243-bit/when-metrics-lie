from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from metrics_lie.scenarios.base import ScenarioContext
from metrics_lie.scenarios.registry import register_scenario


@dataclass(frozen=True)
class LabelNoiseScenario:
    id: str = "label_noise"
    p: float = 0.1  # flip probability

    def apply(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        rng: np.random.Generator,
        ctx: ScenarioContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not (0.0 <= self.p <= 0.5):
            raise ValueError("label_noise.p must be in [0, 0.5]")

        y = y_true.copy()

        if ctx.task == "regression":
            std = float(np.std(y)) if np.std(y) > 0 else 1.0
            noise = rng.normal(loc=0.0, scale=self.p * std, size=y.shape[0])
            y = y.astype(float) + noise
        elif ctx.task == "multiclass_classification":
            classes = np.unique(y_true)
            flips = rng.random(size=y.shape[0]) < self.p
            for i in np.where(flips)[0]:
                other_classes = classes[classes != y[i]]
                if len(other_classes) > 0:
                    y[i] = rng.choice(other_classes)
        else:
            # Binary classification (original behavior)
            flips = rng.random(size=y.shape[0]) < self.p
            y[flips] = 1 - y[flips]

        return y, y_score

    def describe(self) -> Dict[str, Any]:
        return {"id": self.id, "p": self.p}


def _factory(params: dict[str, Any]) -> LabelNoiseScenario:
    p = float(params.get("p", 0.1))
    return LabelNoiseScenario(p=p)


register_scenario("label_noise", _factory)
