from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .components import DecisionComponents
from .score import score_components
from metrics_lie.profiles.schema import DecisionProfile


class DecisionScorecard(BaseModel):
    """
    Transparent scorecard showing weighted scoring of decision components.
    """

    profile_name: str = Field(..., description="Name of the profile used")
    total_score: float = Field(..., description="Total weighted score")
    contributions: dict[str, float] = Field(
        ..., description="Per-component contributions (weight * value)"
    )
    used_components: list[str] = Field(
        ..., description="List of components that were scored"
    )
    ignored_components: list[dict[str, str]] = Field(
        ..., description="List of components that were ignored with reasons"
    )
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata including weights, component values, aggregation info, top contributors",
    )


def build_scorecard(
    decision_components: DecisionComponents, profile: DecisionProfile
) -> DecisionScorecard:
    """
    Build a scorecard from decision components and profile weights.

    Args:
        decision_components: DecisionComponents from extract_components
        profile: DecisionProfile with weights

    Returns:
        DecisionScorecard with scoring results and metadata
    """
    # Score the components
    score_result = score_components(decision_components.components, profile.weights)

    # Compute top contributors (top 3 by absolute contribution)
    contributions = score_result["contributions"]
    top_contributors = sorted(
        contributions.items(), key=lambda x: abs(x[1]), reverse=True
    )[:3]
    top_contributors_dict = {comp: contrib for comp, contrib in top_contributors}

    # Build meta
    meta: dict[str, Any] = {
        "weights": profile.weights.copy(),
        "component_values": {
            k: v for k, v in decision_components.components.items() if v is not None
        },
        "aggregation": decision_components.aggregation.copy(),
        "scenario_scope": decision_components.scenario_scope.copy(),
        "used_scenarios": decision_components.meta.get("used_scenarios", []),
        "top_contributors": top_contributors_dict,
    }

    return DecisionScorecard(
        profile_name=profile.name,
        total_score=score_result["total_score"],
        contributions=score_result["contributions"],
        used_components=score_result["used_components"],
        ignored_components=score_result["ignored_components"],
        meta=meta,
    )
