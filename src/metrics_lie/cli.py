from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import numpy as np

from metrics_lie.artifacts.plots import (
    plot_calibration_curve,
    plot_metric_distribution,
    plot_subgroup_bars,
    plot_threshold_curve,
)
from metrics_lie.datasets.loaders import load_binary_csv
from metrics_lie.diagnostics.calibration import brier_score, expected_calibration_error
from metrics_lie.metrics.core import METRICS
from metrics_lie.schema import Artifact, MetricSummary, ResultBundle, ScenarioResult
from metrics_lie.spec import load_experiment_spec
from metrics_lie.utils.paths import get_run_dir
from metrics_lie.experiments.datasets import dataset_fingerprint_csv
from metrics_lie.experiments.definition import ExperimentDefinition
from metrics_lie.experiments.registry import upsert_experiment as upsert_experiment_jsonl, log_run as log_run_jsonl
from metrics_lie.experiments.runs import RunRecord
from metrics_lie.db.session import get_session
from metrics_lie.db.crud import (
    upsert_experiment,
    insert_run,
    update_run,
    insert_artifacts,
    get_experiment_spec_json,
    get_experiment_id_for_run,
)
from metrics_lie.compare.compare import compare_runs
from metrics_lie.experiments.identity import canonical_json

# Ensure scenario registration occurs (import-time registration)
from metrics_lie.scenarios import class_imbalance, label_noise, score_noise, threshold_gaming  # noqa: F401
from metrics_lie.runner import RunConfig, run_scenarios
from metrics_lie.scenarios.base import ScenarioContext
from metrics_lie.scenarios.registry import create_scenario


def _summary_from_single_value(v: float) -> MetricSummary:
    # Baseline is a single deterministic value (Phase 1.3/1.4)
    return MetricSummary(mean=v, std=0.0, q05=v, q50=v, q95=v, n=1)


def run_from_spec_dict(spec_dict: dict, *, spec_path_for_notes: str | None = None, rerun_of: str | None = None) -> str:
    """
    Execute an experiment run given a parsed spec dictionary.

    This is the canonical execution path used both for CLI `run` (from file)
    and for `rerun` (from a stored spec_json snapshot in the DB).
    """
    spec = load_experiment_spec(spec_dict)

    dataset_fp = dataset_fingerprint_csv(spec.dataset.path)
    exp_def = ExperimentDefinition.from_spec(spec, dataset_fingerprint=dataset_fp)

    # Canonical JSON snapshot of the original spec for deterministic reruns.
    spec_json_str = canonical_json(spec_dict)

    # Phase 2.2: Write to DB
    with get_session() as session:
        upsert_experiment(session, exp_def, spec_json_str)

    # Phase 2.1: Keep JSONL logging optional
    upsert_experiment_jsonl(exp_def)

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
    subgroup = None
    if ds.subgroup is not None:
        subgroup = ds.subgroup.to_numpy()

    metric_fn = METRICS[spec.metric]
    if spec.metric == "accuracy":
        baseline_value = metric_fn(y_true, y_score, threshold=0.5)
    else:
        baseline_value = metric_fn(y_true, y_score)

    baseline_cal = {
        "brier": brier_score(y_true, y_score),
        "ece": expected_calibration_error(y_true, y_score, n_bins=10),
    }

    # --- Phase 1.4: run scenario stress tests (Monte Carlo) ---
    scenario_results = run_scenarios(
        y_true=y_true,
        y_score=y_score,
        metric_name=spec.metric,
        metric_fn=metric_fn,
        scenario_specs=[s.model_dump() for s in spec.scenarios],
        cfg=RunConfig(n_trials=spec.n_trials, seed=spec.seed),
        ctx=ScenarioContext(task=spec.task),
        subgroup=subgroup,
    )

    # --- Phase 1.5: add sensitivity_abs diagnostic ---
    baseline_mean = baseline_value
    scenario_results_with_diag = []
    for sr in scenario_results:
        sensitivity_abs = abs(sr.metric.mean - baseline_mean)
        diag = sr.diagnostics.copy()
        diag["sensitivity_abs"] = sensitivity_abs
        scenario_results_with_diag.append(
            ScenarioResult(
                scenario_id=sr.scenario_id,
                params=sr.params,
                metric=sr.metric,
                diagnostics=diag,
                artifacts=sr.artifacts,
            )
        )

    run_id = uuid.uuid4().hex[:10].upper()
    paths = get_run_dir(run_id)
    paths.ensure()

    run_record = RunRecord(
        run_id=run_id,
        experiment_id=exp_def.experiment_id,
        results_path=str(paths.results_json),
        artifacts_dir=str(paths.artifacts_dir),
        seed_used=spec.seed,
        rerun_of=rerun_of,
    )
    
    # Phase 2.2: Write to DB (queued)
    with get_session() as session:
        insert_run(session, run_record)
    
    # Phase 2.1: Keep JSONL logging optional
    log_run_jsonl(run_record)

    try:
        run_record.mark_running()
        
        # Phase 2.2: Update DB (running)
        with get_session() as session:
            update_run(session, run_record)
        
        # Phase 2.1: Keep JSONL logging optional
        log_run_jsonl(run_record)

        # --- Phase 1.7B: generate artifacts (plots) ---
        rng_artifacts = np.random.default_rng(spec.seed)
        scenario_results_with_artifacts: list[ScenarioResult] = []
        for sr in scenario_results_with_diag:
            artifacts_list: list[Artifact] = []
            scenario_id = sr.scenario_id

            # 1. Metric distribution plot
            try:
                metric_dist_path = paths.artifacts_dir / f"metric_dist_{scenario_id}.png"
                plot_metric_distribution(
                    metric_summary=sr.metric.model_dump(),
                    metric_name=spec.metric,
                    scenario_id=scenario_id,
                    out_path=metric_dist_path,
                )
                artifacts_list.append(
                    Artifact(
                        kind="plot",
                        path=f"artifacts/metric_dist_{scenario_id}.png",
                        meta={"type": "metric_distribution"},
                    )
                )
            except Exception:
                pass  # Skip if plot generation fails

            # 2. Calibration curve (run one representative trial)
            try:
                scenario = create_scenario(scenario_id, sr.params)
                y_p_rep, s_p_rep = scenario.apply(y_true, y_score, rng_artifacts, ScenarioContext(task=spec.task))
                if len(y_p_rep) > 0 and len(s_p_rep) > 0:
                    cal_path = paths.artifacts_dir / f"calibration_{scenario_id}.png"
                    plot_calibration_curve(
                        y_true=y_p_rep,
                        y_score=s_p_rep,
                        scenario_id=scenario_id,
                        out_path=cal_path,
                    )
                    artifacts_list.append(
                        Artifact(
                            kind="plot",
                            path=f"artifacts/calibration_{scenario_id}.png",
                            meta={"type": "calibration_curve"},
                        )
                    )
            except Exception:
                pass  # Skip if plot generation fails

            # 3. Subgroup metric bars (if subgroup diagnostics exist)
            try:
                subgroup_metric = sr.diagnostics.get("subgroup_metric")
                if subgroup_metric:
                    group_means = {k: v["mean"] for k, v in subgroup_metric.items()}
                    if group_means:
                        subgroup_path = paths.artifacts_dir / f"subgroup_metric_{scenario_id}.png"
                        plot_subgroup_bars(
                            group_means=group_means,
                            scenario_id=scenario_id,
                            out_path=subgroup_path,
                        )
                        artifacts_list.append(
                            Artifact(
                                kind="plot",
                                path=f"artifacts/subgroup_metric_{scenario_id}.png",
                                meta={"type": "subgroup_comparison"},
                            )
                        )
            except Exception:
                pass  # Skip if plot generation fails

            # 4. Threshold curve (only for accuracy with metric_inflation)
            if spec.metric == "accuracy":
                try:
                    metric_inflation = sr.diagnostics.get("metric_inflation")
                    if metric_inflation:
                        scenario = create_scenario(scenario_id, sr.params)
                        y_p_rep, s_p_rep = scenario.apply(y_true, y_score, rng_artifacts, ScenarioContext(task=spec.task))
                        if len(y_p_rep) > 0 and len(s_p_rep) > 0:
                            # Get mean optimal threshold from diagnostics (approximate)
                            # Use a representative threshold from the inflation data
                            baseline_thresh = 0.5
                            # Estimate optimized threshold from delta (use 0.5 + small adjustment as proxy)
                            # Actually, we need to recompute or store it - let's use a simple heuristic
                            # For now, use 0.5 as baseline and compute optimal from representative trial
                            from metrics_lie.diagnostics.metric_gaming import find_optimal_threshold
                            thresholds = np.linspace(0.05, 0.95, 19)
                            opt_thresh, _ = find_optimal_threshold(y_p_rep, s_p_rep, thresholds)
                            
                            threshold_path = paths.artifacts_dir / f"threshold_curve_{scenario_id}.png"
                            plot_threshold_curve(
                                y_true=y_p_rep,
                                y_score=s_p_rep,
                                baseline_threshold=baseline_thresh,
                                optimized_threshold=opt_thresh,
                                scenario_id=scenario_id,
                                out_path=threshold_path,
                            )
                            artifacts_list.append(
                                Artifact(
                                    kind="plot",
                                    path=f"artifacts/threshold_curve_{scenario_id}.png",
                                    meta={"type": "threshold_optimization"},
                                )
                            )
                except Exception:
                    pass  # Skip if plot generation fails

            scenario_results_with_artifacts.append(
                ScenarioResult(
                    scenario_id=sr.scenario_id,
                    params=sr.params,
                    metric=sr.metric,
                    diagnostics=sr.diagnostics,
                    artifacts=artifacts_list,
                )
            )

            if artifacts_list:
                print(f"[PLOT] Saved {len(artifacts_list)} artifacts for scenario {scenario_id}")

        notes = {
            "phase": "1.7B",
            "spec_path": spec_path_for_notes,
            "baseline_diagnostics": baseline_cal,
        }

        bundle = ResultBundle(
            run_id=run_id,
            experiment_name=spec.name,
            metric_name=spec.metric,
            baseline=_summary_from_single_value(baseline_value),
            scenarios=scenario_results_with_artifacts,
            notes=notes,
        )

        paths.results_json.write_text(bundle.to_pretty_json(), encoding="utf-8")
        print(f"[OK] Wrote results: {paths.results_json}")
        print(f"Baseline {spec.metric} = {baseline_value:.6f}")
        if spec.scenarios:
            print(f"[OK] Ran {len(spec.scenarios)} scenario(s) with n_trials={spec.n_trials}")

        # Phase 2.2: Insert artifacts into DB
        all_artifacts: list[Artifact] = []
        for sr in scenario_results_with_artifacts:
            all_artifacts.extend(sr.artifacts)
        if all_artifacts:
            with get_session() as session:
                insert_artifacts(session, run_id, all_artifacts)

        run_record.mark_completed()
        
        # Phase 2.2: Update DB (completed)
        with get_session() as session:
            update_run(session, run_record)
        
        # Phase 2.1: Keep JSONL logging optional
        log_run_jsonl(run_record)
    except Exception as exc:  # pragma: no cover - simple logging wrapper
        run_record.mark_failed(str(exc))
        
        # Phase 2.2: Update DB (failed)
        with get_session() as session:
            update_run(session, run_record)
        
        # Phase 2.1: Keep JSONL logging optional
        log_run_jsonl(run_record)
        raise

    return run_id


def run(spec_path: str) -> str:
    """Run an experiment from a spec JSON file path."""
    spec_dict = json.loads(Path(spec_path).read_text())
    return run_from_spec_dict(spec_dict, spec_path_for_notes=spec_path)


def rerun(run_id: str) -> str:
    """
    Deterministically rerun a completed experiment using the stored spec_json snapshot.

    This creates a new run (with a new run_id) linked to the same experiment_id,
    and optionally records the original run_id in the DB.
    """
    with get_session() as session:
        experiment_id = get_experiment_id_for_run(session, run_id)
        spec_json_str = get_experiment_spec_json(session, experiment_id)

    if not spec_json_str:
        raise ValueError(
            "No stored spec_json found for experiment. "
            "Please run database migrations with 'alembic upgrade head' before using 'rerun'."
        )

    spec_dict = json.loads(spec_json_str)
    # Use a descriptive marker in notes so results.json keeps the same schema.
    spec_path_for_notes = f"<rerun_of:{run_id}>"
    new_run_id = run_from_spec_dict(spec_dict, spec_path_for_notes=spec_path_for_notes, rerun_of=run_id)
    print(new_run_id)
    return new_run_id


def main() -> None:
    parser = argparse.ArgumentParser(prog="metrics-lie")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run an experiment from a spec JSON (baseline + scenarios)")
    p_run.add_argument("spec", type=str, help="Path to experiment spec JSON")

    p_compare = sub.add_parser("compare", help="Compare two runs by run_id and print a JSON report")
    p_compare.add_argument("run_a", type=str, help="Run ID A")
    p_compare.add_argument("run_b", type=str, help="Run ID B")

    p_rerun = sub.add_parser("rerun", help="Deterministically rerun an experiment by run_id using stored spec")
    p_rerun.add_argument("run_id", type=str, help="Existing run ID to rerun")

    args = parser.parse_args()

    if args.cmd == "run":
        run(args.spec)
    elif args.cmd == "compare":
        report = compare_runs(args.run_a, args.run_b)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "rerun":
        rerun(args.run_id)


if __name__ == "__main__":
    main()

