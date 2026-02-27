from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

import numpy as np


@dataclass(frozen=True)
class ScenarioContext:
    """
    Shared runtime knobs for scenarios.
    Extend later (e.g., task type, subgroup col, etc.).
    """

    task: str = "binary_classification"
    surface_type: str = "probability"
    n_classes: int | None = None


class Scenario(Protocol):
    """
    A scenario perturbs the dataset (y_true and/or y_score) for stress testing.
    """

    id: str

    def apply(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        rng: np.random.Generator,
        ctx: ScenarioContext,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def describe(self) -> Dict[str, Any]: ...
