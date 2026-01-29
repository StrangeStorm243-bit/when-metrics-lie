from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class DecisionProfile(BaseModel):
    """
    Scenario-first decision profile defining how scenario results are aggregated
    and interpreted for decision-making.
    """

    name: str = Field(..., description="Profile name")
    description: str | None = Field(None, description="Human-readable description")

    aggregation: dict = Field(
        ...,
        description="Scenario aggregation strategy",
    )

    objectives: dict = Field(
        ...,
        description="Primary and secondary objectives for evaluation",
    )

    thresholds: dict = Field(
        default_factory=dict,
        description="Regression thresholds for various diagnostics",
    )

    weights: dict[str, float] = Field(
        default_factory=dict,
        description="Weights for different components (for scoring engine)",
    )

    @field_validator("aggregation")
    @classmethod
    def validate_aggregation(cls, v: dict) -> dict:
        """Validate aggregation configuration."""
        mode = v.get("mode")
        if mode not in ["worst_case", "mean", "percentile"]:
            raise ValueError(f"aggregation.mode must be one of: worst_case, mean, percentile. Got: {mode}")

        if mode == "percentile":
            percentile = v.get("percentile")
            if percentile is None:
                raise ValueError("aggregation.percentile is required when mode='percentile'")
            if not (0 < percentile <= 0.5):
                raise ValueError(f"aggregation.percentile must be in (0, 0.5]. Got: {percentile}")

        scenario_scope = v.get("scenario_scope")
        if scenario_scope not in ["all", "subset"]:
            raise ValueError(f"aggregation.scenario_scope must be 'all' or 'subset'. Got: {scenario_scope}")

        if scenario_scope == "subset":
            scenario_subset = v.get("scenario_subset")
            if not scenario_subset or len(scenario_subset) == 0:
                raise ValueError("aggregation.scenario_subset must be non-empty when scenario_scope='subset'")

        return v

    @field_validator("objectives")
    @classmethod
    def validate_objectives(cls, v: dict) -> dict:
        """Validate objectives configuration."""
        primary = v.get("primary")
        if primary != "metric_mean_delta":
            raise ValueError(f"objectives.primary must be 'metric_mean_delta' (scenario-first). Got: {primary}")

        secondary = v.get("secondary", [])
        allowed_secondary = [
            "ece_mean_delta",
            "brier_mean_delta",
            "subgroup_gap_delta",
            "sensitivity_abs_delta",
            "metric_inflation_delta",
        ]
        for obj in secondary:
            if obj not in allowed_secondary:
                raise ValueError(
                    f"objectives.secondary contains invalid value: {obj}. "
                    f"Allowed: {allowed_secondary}"
                )

        return v

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate weights keys match allowed component names."""
        allowed_keys = [
            "metric_mean_delta",
            "ece_mean_delta",
            "brier_mean_delta",
            "subgroup_gap_delta",
            "sensitivity_abs_delta",
            "metric_inflation_delta",
        ]
        for key in v.keys():
            if key not in allowed_keys:
                raise ValueError(
                    f"weights contains invalid key: {key}. "
                    f"Allowed: {allowed_keys}"
                )
        return v

    @model_validator(mode="after")
    def set_default_thresholds(self) -> "DecisionProfile":
        """Set default thresholds if not provided."""
        defaults = {
            "calibration_regression_ece": 0.02,
            "calibration_regression_brier": 0.02,
            "subgroup_gap_regression": 0.03,
            "metric_regression": -0.01,
        }
        for key, default_value in defaults.items():
            if key not in self.thresholds:
                self.thresholds[key] = default_value
        return self

