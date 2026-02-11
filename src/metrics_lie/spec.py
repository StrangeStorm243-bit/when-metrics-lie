from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


TaskType = Literal["binary_classification"]


class DatasetSpec(BaseModel):
    """
    Where the data comes from (Phase 1: local CSV).
    Later we can support: built-in datasets, URLs, DB refs, etc.
    """

    source: Literal["csv"] = "csv"
    path: str = Field(..., description="Path to a CSV file on disk.")
    y_true_col: str = Field(
        ..., description="Column name for ground-truth labels (0/1)."
    )
    y_score_col: str = Field(
        ..., description="Column name for predicted probabilities (0..1)."
    )
    subgroup_col: Optional[str] = Field(
        default=None,
        description="Optional subgroup column for fairness/robustness diagnostics.",
    )
    feature_cols: Optional[List[str]] = Field(
        default=None,
        description="Optional list of feature columns for model inference.",
    )


class ModelSourceSpec(BaseModel):
    kind: Literal["pickle", "import"] = Field(
        ...,
        description="Model source type. Callable sources are only supported in Python API.",
    )
    path: Optional[str] = Field(
        default=None,
        description="Path to pickled model (if kind=pickle).",
    )
    import_path: Optional[str] = Field(
        default=None,
        description="Import path to model object (if kind=import), e.g. module:attr",
    )
    threshold: Optional[float] = Field(
        default=0.5, description="Threshold for probability surfaces."
    )
    positive_label: Optional[int] = Field(
        default=1, description="Positive label index."
    )


class ScenarioSpec(BaseModel):
    """
    A single stress test configuration.
    Example: label_noise with p=0.1
    """

    id: str = Field(..., description="Scenario identifier, e.g. 'label_noise'.")
    params: Dict[str, Any] = Field(default_factory=dict)


class ExperimentSpec(BaseModel):
    """
    A full experiment = dataset + metric + scenarios + run controls.
    """

    name: str = Field(..., description="Human-readable name for the run.")
    task: TaskType = "binary_classification"

    dataset: DatasetSpec
    metric: str = Field(
        ..., description="Metric identifier, e.g. 'auc', 'logloss', 'accuracy'."
    )
    model_source: Optional[ModelSourceSpec] = Field(
        default=None,
        description="Optional model source to generate predictions via inference.",
    )

    scenarios: List[ScenarioSpec] = Field(default_factory=list)

    n_trials: int = Field(
        default=200,
        ge=1,
        le=50_000,
        description="How many Monte Carlo trials per scenario.",
    )
    seed: int = Field(default=42, description="Random seed for reproducibility.")

    tags: Dict[str, str] = Field(
        default_factory=dict, description="Optional metadata tags."
    )


def load_experiment_spec(data: Dict[str, Any]) -> ExperimentSpec:
    """
    Convenience function so the rest of the code doesn't care about pydantic specifics.
    """
    return ExperimentSpec.model_validate(data)
