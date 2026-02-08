from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from metrics_lie.diagnostics.calibration import brier_score, expected_calibration_error
from metrics_lie.diagnostics.metric_gaming import (
    accuracy_at_threshold,
    find_optimal_threshold,
)
from metrics_lie.diagnostics.subgroups import (
    compute_group_sizes,
    group_indices,
    safe_metric_for_group,
)
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
    subgroup: np.ndarray | None = None,
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

        # Metric gaming: threshold optimization (only for accuracy)
        baseline_accs: list[float] = []
        optimized_accs: list[float] = []
        optimal_thresholds: list[float] = []
        gaming_trial_data: list[tuple[np.ndarray, np.ndarray]] = []

        # Subgroup diagnostics: per-group lists
        subgroup_metric_vals: Dict[str, list[float]] = {}
        subgroup_brier_vals: Dict[str, list[float]] = {}
        subgroup_ece_vals: Dict[str, list[float]] = {}
        subgroup_p_aligned: np.ndarray | None = None

        for _ in range(cfg.n_trials):
            y_p, s_p = scenario.apply(y_true, y_score, rng, ctx)
            if metric_name == "accuracy":
                v = metric_fn(y_p, s_p, threshold=0.5)
            else:
                v = metric_fn(y_p, s_p)
            vals.append(float(v))
            if ctx.surface_type == "probability":
                briers.append(brier_score(y_p, s_p))
                eces.append(expected_calibration_error(y_p, s_p, n_bins=10))

            # Metric gaming: threshold optimization (only for accuracy)
            if metric_name == "accuracy":
                thresholds = np.linspace(0.05, 0.95, 19)
                baseline_acc = accuracy_at_threshold(y_p, s_p, 0.5)
                opt_thresh, opt_acc = find_optimal_threshold(y_p, s_p, thresholds)
                baseline_accs.append(baseline_acc)
                optimized_accs.append(opt_acc)
                optimal_thresholds.append(opt_thresh)
                gaming_trial_data.append((y_p.copy(), s_p.copy()))

            # Subgroup diagnostics (only if subgroup provided and lengths align)
            if subgroup is not None and len(y_p) == len(y_true) and len(subgroup) == len(y_p):
                subgroup_p = subgroup
                if subgroup_p_aligned is None:
                    subgroup_p_aligned = subgroup_p
                groups = group_indices(subgroup_p)
                for group_key, group_mask in groups.items():
                    y_g = y_p[group_mask]
                    s_g = s_p[group_mask]
                    if len(y_g) > 0:
                        # Metric per group
                        metric_val = safe_metric_for_group(metric_name, metric_fn, y_g, s_g)
                        if metric_val is not None:
                            if group_key not in subgroup_metric_vals:
                                subgroup_metric_vals[group_key] = []
                            subgroup_metric_vals[group_key].append(metric_val)

                        # Calibration per group (only for probability surface)
                        if group_key not in subgroup_brier_vals:
                            subgroup_brier_vals[group_key] = []
                            subgroup_ece_vals[group_key] = []
                        if ctx.surface_type == "probability":
                            subgroup_brier_vals[group_key].append(brier_score(y_g, s_g))
                            subgroup_ece_vals[group_key].append(
                                expected_calibration_error(y_g, s_g, n_bins=10)
                            )

        diag: Dict = {}
        if ctx.surface_type == "probability" and briers and eces:
            diag["brier"] = summarize(briers).model_dump()
            diag["ece"] = summarize(eces).model_dump()

        # Add subgroup diagnostics if computed
        if subgroup is not None and (
            len(subgroup_metric_vals) > 0 or len(subgroup_brier_vals) > 0
        ):
            # Per-group metric summaries
            subgroup_metric_summaries: Dict[str, Dict] = {}
            for group_key, metric_list in subgroup_metric_vals.items():
                if len(metric_list) > 0:
                    subgroup_metric_summaries[group_key] = summarize(metric_list).model_dump()
            if subgroup_metric_summaries:
                diag["subgroup_metric"] = subgroup_metric_summaries

            # Per-group calibration summaries
            subgroup_brier_summaries: Dict[str, Dict] = {}
            subgroup_ece_summaries: Dict[str, Dict] = {}
            for group_key in subgroup_brier_vals:
                if len(subgroup_brier_vals[group_key]) > 0:
                    subgroup_brier_summaries[group_key] = summarize(
                        subgroup_brier_vals[group_key]
                    ).model_dump()
                    subgroup_ece_summaries[group_key] = summarize(
                        subgroup_ece_vals[group_key]
                    ).model_dump()
            if subgroup_brier_summaries:
                diag["subgroup_brier"] = subgroup_brier_summaries
                diag["subgroup_ece"] = subgroup_ece_summaries

            # Subgroup gap analysis
            group_sizes = compute_group_sizes(subgroup_p_aligned if subgroup_p_aligned is not None else subgroup)
            group_means: Dict[str, float] = {}
            for group_key, metric_list in subgroup_metric_vals.items():
                if len(metric_list) > 0:
                    group_means[group_key] = float(np.mean(metric_list))

            if group_means:
                worst_group = min(group_means, key=group_means.get)  # type: ignore
                best_group = max(group_means, key=group_means.get)  # type: ignore
                gap = group_means[best_group] - group_means[worst_group]
                diag["subgroup_gap"] = {
                    "worst_group": worst_group,
                    "best_group": best_group,
                    "gap": float(gap),
                    "means": {k: float(v) for k, v in group_means.items()},
                    "group_sizes": group_sizes,
                }

        # Metric gaming diagnostics (only for accuracy)
        if metric_name == "accuracy" and len(baseline_accs) > 0:
            baseline_mean = float(np.mean(baseline_accs))
            optimized_mean = float(np.mean(optimized_accs))
            delta = optimized_mean - baseline_mean
            mean_opt_thresh = float(np.mean(optimal_thresholds))

            # Downstream impacts: compute on representative trial with mean optimal threshold
            downstream: Dict = {}
            if gaming_trial_data and ctx.surface_type == "probability":
                # Use first trial as representative
                y_rep, s_rep = gaming_trial_data[0]
                y_pred_opt = (s_rep >= mean_opt_thresh).astype(int)
                downstream["brier"] = float(brier_score(y_rep, s_rep))
                downstream["ece"] = float(expected_calibration_error(y_rep, s_rep, n_bins=10))

                # Subgroup gap at optimized threshold (if subgroup exists)
                if subgroup is not None and len(y_rep) == len(y_true) and len(subgroup) == len(y_rep):
                    groups = group_indices(subgroup)
                    group_accs_opt: Dict[str, float] = {}
                    for group_key, group_mask in groups.items():
                        y_g = y_rep[group_mask]
                        s_g = s_rep[group_mask]
                        if len(y_g) > 0:
                            acc_opt = accuracy_at_threshold(y_g, s_g, mean_opt_thresh)
                            group_accs_opt[group_key] = acc_opt
                    if group_accs_opt:
                        worst = min(group_accs_opt.values())
                        best = max(group_accs_opt.values())
                        downstream["subgroup_gap"] = float(best - worst)
                    else:
                        downstream["subgroup_gap"] = None
                else:
                    downstream["subgroup_gap"] = None

            diag["metric_inflation"] = {
                "metric": "accuracy",
                "baseline": baseline_mean,
                "optimized": optimized_mean,
                "delta": delta,
                "downstream": downstream,
            }

        results.append(
            ScenarioResult(
                scenario_id=scenario_id,
                params=params,
                metric=summarize(vals),
                diagnostics=diag,
                artifacts=[],
            )
        )

    return results
