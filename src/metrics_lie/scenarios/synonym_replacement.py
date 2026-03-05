from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from metrics_lie.scenarios.base import ScenarioContext
from metrics_lie.scenarios.registry import register_scenario

_SYNONYMS: dict[str, list[str]] = {
    "good": ["great", "fine", "excellent"],
    "bad": ["poor", "terrible", "awful"],
    "big": ["large", "huge", "enormous"],
    "small": ["tiny", "little", "miniature"],
    "fast": ["quick", "rapid", "swift"],
    "slow": ["sluggish", "gradual", "leisurely"],
}


@dataclass(frozen=True)
class SynonymReplacementScenario:
    """Simple word-level replacement using a built-in synonym map."""

    id: str = "synonym_replacement"
    replace_rate: float = 0.2

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
                for j, w in enumerate(words):
                    if rng.random() < self.replace_rate and w.lower() in _SYNONYMS:
                        words[j] = rng.choice(_SYNONYMS[w.lower()])
                result[i] = " ".join(words)
        return y_true, result

    def describe(self) -> Dict[str, Any]:
        return {"id": self.id, "replace_rate": self.replace_rate}


def _factory(params: dict[str, Any]) -> SynonymReplacementScenario:
    return SynonymReplacementScenario(replace_rate=float(params.get("replace_rate", 0.2)))


register_scenario("synonym_replacement", _factory)
