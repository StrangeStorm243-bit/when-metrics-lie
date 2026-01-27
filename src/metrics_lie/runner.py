from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from metrics_lie.diagnostics.calibration import brier_score, expected_calibration_error
from metrics_lie.schema import MetricSummary, ScenarioResult
from metrics_lie.scenarios.base import ScenarioContext
from metrics_lie.scenarios.registry import create_scenario


def summarize(values: list[float]) -> MetricSummary:
    arr = np.array(values, dtype=float)
    q05, q50, q95 = np.quantile(arr, [0.05, 0.50, 0.95])
    return MetricSummary(
        mean=float(arr.mean()),
        std=float(arr.std(ddof=0)),
        q05=float(q05),
        q50=float(q50),
        q95=float(q95),
        n=int(arr.size),
    )


@dataclass(frozen=True)
class RunConfig:
    n_trials: int
    seed: int


def run_scenarios(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_name: str,
    metric_fn: Callable[..., float],
    scenario_specs: list[dict],
    cfg: RunConfig,
    ctx: ScenarioContext,
) -> list[ScenarioResult]:
    rng = np.random.default_rng(cfg.seed)
    results: list[ScenarioResult] = []

    for s in scenario_specs:
        scenario_id = s["id"]
        params = s.get("params", {}) or {}
        scenario = create_scenario(scenario_id, params)

        vals: list[float] = []
        briers: list[float] = []
        eces: list[float] = []
        for _ in range(cfg.n_trials):
            y_p, s_p = scenario.apply(y_true, y_score, rng, ctx)
            if metric_name == "accuracy":
                v = metric_fn(y_p, s_p, threshold=0.5)
            else:
                v = metric_fn(y_p, s_p)
            vals.append(float(v))
            briers.append(brier_score(y_p, s_p))
            eces.append(expected_calibration_error(y_p, s_p, n_bins=10))

        results.append(
            ScenarioResult(
                scenario_id=scenario_id,
                params=params,
                metric=summarize(vals),
                diagnostics={
                    "brier": summarize(briers).model_dump(),
                    "ece": summarize(eces).model_dump(),
                },
                artifacts=[],
            )
        )

    return results
