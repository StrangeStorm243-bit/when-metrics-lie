from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DecisionComponents(BaseModel):
    """
    Aggregated decision components extracted from a compare report
    according to a DecisionProfile's aggregation rules.
    """

    model_config = ConfigDict()

    profile_name: str = Field(..., description="Name of the profile used")
    aggregation: dict[str, Any] = Field(..., description="Echo of profile.aggregation used")
    scenario_scope: dict[str, Any] = Field(..., description="Echo of scenario list actually used")
    components: dict[str, float | None] = Field(
        ...,
        description="Aggregated component values (scenario-first)",
    )
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about extraction (must be JSON-serializable)",
    )

