from __future__ import annotations

from typing import Any


def score_components(components: dict[str, float | None], weights: dict[str, float]) -> dict[str, Any]:
    """
    Compute a transparent scorecard from component values and weights.
    
    Rules:
    - Only score keys that exist in weights AND have non-None values.
    - contribution = weight * value
    - total_score = sum(contributions)
    - Missing keys or None values are added to ignored_components with reason.
    
    Args:
        components: Dictionary of component name -> value (float or None)
        weights: Dictionary of component name -> weight (float)
    
    Returns:
        Dictionary with:
        - total_score: float
        - contributions: dict[str, float] (component -> contribution)
        - used_components: list[str]
        - ignored_components: list[dict[str, str]] with "component" and "reason" keys
    """
    contributions: dict[str, float] = {}
    used_components: list[str] = []
    ignored_components: list[dict[str, str]] = []
    
    # Process all keys in weights
    for comp_name, weight in weights.items():
        value = components.get(comp_name)
        
        if value is None:
            ignored_components.append({
                "component": comp_name,
                "reason": "value_is_none"
            })
        else:
            # Valid component: compute contribution
            contribution = weight * float(value)
            contributions[comp_name] = contribution
            used_components.append(comp_name)
    
    # Also check for components that exist but aren't in weights
    for comp_name in components.keys():
        if comp_name not in weights:
            # Only add if not already in ignored_components
            if not any(ic["component"] == comp_name for ic in ignored_components):
                ignored_components.append({
                    "component": comp_name,
                    "reason": "not_in_weights"
                })
    
    total_score = sum(contributions.values())
    
    return {
        "total_score": total_score,
        "contributions": contributions,
        "used_components": used_components,
        "ignored_components": ignored_components,
    }

