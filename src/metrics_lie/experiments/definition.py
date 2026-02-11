from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from metrics_lie.spec import ExperimentSpec

from .identity import canonical_json, sha256_hex, short_id


class ExperimentDefinition(BaseModel):
    experiment_id: str
    name: str
    task: str
    metric: str
    n_trials: int
    seed: int

    dataset: Dict[str, Any]
    scenarios: List[Dict[str, Any]]

    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    spec_schema_version: Optional[str] = None

    @classmethod
    def from_spec(
        cls, spec: ExperimentSpec, dataset_fingerprint: str
    ) -> "ExperimentDefinition":
        dataset_payload = {
            "fingerprint": dataset_fingerprint,
            "y_true_col": spec.dataset.y_true_col,
            "y_score_col": spec.dataset.y_score_col,
            "subgroup_col": spec.dataset.subgroup_col,
        }

        scenarios_payload: List[Dict[str, Any]] = []
        for s in spec.scenarios:
            scenarios_payload.append(
                {
                    "id": s.id,
                    "params": s.params,
                }
            )

        semantics_payload = {
            "task": spec.task,
            "metric": spec.metric,
            "n_trials": spec.n_trials,
            "seed": spec.seed,
            "dataset": dataset_payload,
            "scenarios": scenarios_payload,
        }

        payload_json = canonical_json(semantics_payload)
        digest = sha256_hex(payload_json)
        experiment_id = short_id("exp", digest, n=10)

        return cls(
            experiment_id=experiment_id,
            name=spec.name,
            task=spec.task,
            metric=spec.metric,
            n_trials=spec.n_trials,
            seed=spec.seed,
            dataset=dataset_payload,
            scenarios=scenarios_payload,
        )
