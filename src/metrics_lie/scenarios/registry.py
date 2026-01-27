from __future__ import annotations

from typing import Any, Callable, Dict

from metrics_lie.scenarios.base import Scenario

# factory signature: params -> scenario
ScenarioFactory = Callable[[dict[str, Any]], Scenario]

_REGISTRY: Dict[str, ScenarioFactory] = {}


def register_scenario(scenario_id: str, factory: ScenarioFactory) -> None:
    if scenario_id in _REGISTRY:
        raise ValueError(f"Scenario already registered: {scenario_id}")
    _REGISTRY[scenario_id] = factory


def create_scenario(scenario_id: str, params: dict[str, Any]) -> Scenario:
    if scenario_id not in _REGISTRY:
        raise ValueError(f"Unknown scenario '{scenario_id}'. Registered: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[scenario_id](params)


def list_scenarios() -> list[str]:
    return sorted(_REGISTRY.keys())
