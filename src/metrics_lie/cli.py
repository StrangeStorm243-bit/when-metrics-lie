from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from metrics_lie.datasets.loaders import load_binary_csv
from metrics_lie.metrics.core import METRICS
from metrics_lie.schema import MetricSummary, ResultBundle
from metrics_lie.spec import load_experiment_spec
from metrics_lie.utils.paths import get_run_dir

# Ensure scenario registration occurs (import-time registration)
from metrics_lie.scenarios import label_noise, score_noise  # noqa: F401
from metrics_lie.runner import RunConfig, run_scenarios
from metrics_lie.scenarios.base import ScenarioContext


def _summary_from_single_value(v: float) -> MetricSummary:
    # Baseline is a single deterministic value (Phase 1.3/1.4)
    return MetricSummary(mean=v, std=0.0, q05=v, q50=v, q95=v, n=1)


def run(spec_path: str) -> str:
    spec_json = json.loads(Path(spec_path).read_text())
    spec = load_experiment_spec(spec_json)

    if spec.metric not in METRICS:
        raise ValueError(f"Unknown metric '{spec.metric}'. Supported: {sorted(METRICS.keys())}")

    ds = load_binary_csv(
        path=spec.dataset.path,
        y_true_col=spec.dataset.y_true_col,
        y_score_col=spec.dataset.y_score_col,
        subgroup_col=spec.dataset.subgroup_col,
    )

    y_true = ds.y_true.to_numpy(dtype=int)
    y_score = ds.y_score.to_numpy(dtype=float)

    metric_fn = METRICS[spec.metric]
    if spec.metric == "accuracy":
        baseline_value = metric_fn(y_true, y_score, threshold=0.5)
    else:
        baseline_value = metric_fn(y_true, y_score)

    # --- Phase 1.4: run scenario stress tests (Monte Carlo) ---
    scenario_results = run_scenarios(
        y_true=y_true,
        y_score=y_score,
        metric_name=spec.metric,
        metric_fn=metric_fn,
        scenario_specs=[s.model_dump() for s in spec.scenarios],
        cfg=RunConfig(n_trials=spec.n_trials, seed=spec.seed),
        ctx=ScenarioContext(task=spec.task),
    )

    run_id = uuid.uuid4().hex[:10].upper()
    paths = get_run_dir(run_id)
    paths.ensure()

    bundle = ResultBundle(
        run_id=run_id,
        experiment_name=spec.name,
        metric_name=spec.metric,
        baseline=_summary_from_single_value(baseline_value),
        scenarios=scenario_results,
        notes={"phase": "1.4", "spec_path": spec_path},
    )

    paths.results_json.write_text(bundle.to_pretty_json(), encoding="utf-8")
    print(f"✅ Wrote results: {paths.results_json}")
    print(f"Baseline {spec.metric} = {baseline_value:.6f}")
    if spec.scenarios:
        print(f"✅ Ran {len(spec.scenarios)} scenario(s) with n_trials={spec.n_trials}")

    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(prog="metrics-lie")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run an experiment from a spec JSON (baseline + scenarios)")
    p_run.add_argument("spec", type=str, help="Path to experiment spec JSON")

    args = parser.parse_args()

    if args.cmd == "run":
        run(args.spec)


if __name__ == "__main__":
    main()

