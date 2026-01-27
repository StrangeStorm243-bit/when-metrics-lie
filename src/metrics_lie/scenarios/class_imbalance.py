from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from metrics_lie.scenarios.base import ScenarioContext
from metrics_lie.scenarios.registry import register_scenario


@dataclass(frozen=True)
class ClassImbalanceScenario:
    id: str = "class_imbalance"
    target_pos_rate: float = 0.2
    max_remove_frac: float = 0.8

    def apply(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        rng: np.random.Generator,
        ctx: ScenarioContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not (0.05 <= self.target_pos_rate <= 0.95):
            raise ValueError("class_imbalance.target_pos_rate must be in [0.05, 0.95]")
        if not (0.0 <= self.max_remove_frac <= 0.95):
            raise ValueError("class_imbalance.max_remove_frac must be in [0.0, 0.95]")

        n = len(y_true)
        if n == 0:
            return y_true, y_score

        pos_mask = y_true == 1
        n_pos = int(pos_mask.sum())
        n_neg = n - n_pos

        if n_pos == 0 or n_neg == 0:
            # Cannot shift if one class is missing
            return y_true, y_score

        current_pos_rate = n_pos / n
        target_n_pos = int(round(self.target_pos_rate * n))

        # Determine which class to subsample
        if target_n_pos < n_pos:
            # Need to reduce positives
            remove_count = n_pos - target_n_pos
            max_remove = int(n_pos * self.max_remove_frac)
            remove_count = min(remove_count, max_remove)
            if remove_count <= 0:
                return y_true, y_score
            pos_indices = np.where(pos_mask)[0]
            remove_indices = rng.choice(pos_indices, size=remove_count, replace=False)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[remove_indices] = False
        elif target_n_pos > n_pos:
            # Need to reduce negatives
            target_n_neg = n - target_n_pos
            remove_count = n_neg - target_n_neg
            max_remove = int(n_neg * self.max_remove_frac)
            remove_count = min(remove_count, max_remove)
            if remove_count <= 0:
                return y_true, y_score
            neg_indices = np.where(~pos_mask)[0]
            remove_indices = rng.choice(neg_indices, size=remove_count, replace=False)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[remove_indices] = False
        else:
            # Already at target
            return y_true, y_score

        return y_true[keep_mask], y_score[keep_mask]

    def describe(self) -> Dict[str, Any]:
        return {"id": self.id, "target_pos_rate": self.target_pos_rate, "max_remove_frac": self.max_remove_frac}


def _factory(params: dict[str, Any]) -> ClassImbalanceScenario:
    target_pos_rate = float(params.get("target_pos_rate", 0.2))
    max_remove_frac = float(params.get("max_remove_frac", 0.8))
    return ClassImbalanceScenario(target_pos_rate=target_pos_rate, max_remove_frac=max_remove_frac)


register_scenario("class_imbalance", _factory)

