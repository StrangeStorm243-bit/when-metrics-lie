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
    enqueue_job_run_experiment,
    enqueue_job_rerun,
    list_experiments,
    get_experiment,
    list_runs,
    get_run,
    list_jobs,
    get_job,
    list_artifacts_for_run,
)
from metrics_lie.worker import process_one_job
from metrics_lie.compare.compare import compare_runs
from metrics_lie.execution import run_from_spec_dict, rerun
from metrics_lie.cli_format import format_table, short
from metrics_lie.decision import extract_components, build_scorecard
from metrics_lie.profiles import get_profile_or_load


def run(spec_path: str) -> str:
    """Run an experiment from a spec JSON file path."""
    spec_dict = json.loads(Path(spec_path).read_text())
    return run_from_spec_dict(spec_dict, spec_path_for_notes=spec_path)


def main() -> None:
    parser = argparse.ArgumentParser(prog="metrics-lie")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run an experiment from a spec JSON (baseline + scenarios)")
    p_run.add_argument("spec", type=str, help="Path to experiment spec JSON")

    p_compare = sub.add_parser("compare", help="Compare two runs by run_id and print a JSON report")
    p_compare.add_argument("run_a", type=str, help="Run ID A")
    p_compare.add_argument("run_b", type=str, help="Run ID B")

    p_score = sub.add_parser("score", help="Score two runs using a decision profile and print a JSON report")
    p_score.add_argument("run_a", type=str, help="Run ID A")
    p_score.add_argument("run_b", type=str, help="Run ID B")
    p_score.add_argument("--profile", type=str, default="balanced", help="Profile preset name or JSON file path (default: balanced)")

    p_rerun = sub.add_parser("rerun", help="Deterministically rerun an experiment by run_id using stored spec")
    p_rerun.add_argument("run_id", type=str, help="Existing run ID to rerun")

    p_enqueue_run = sub.add_parser("enqueue-run", help="Enqueue a job to run an experiment")
    p_enqueue_run.add_argument("experiment_id", type=str, help="Experiment ID to run")

    p_enqueue_rerun = sub.add_parser("enqueue-rerun", help="Enqueue a job to rerun a run")
    p_enqueue_rerun.add_argument("run_id", type=str, help="Run ID to rerun")

    p_worker_once = sub.add_parser("worker-once", help="Process one job from the queue and exit")

    # Phase 2.6: Query commands
    p_experiments = sub.add_parser("experiments", help="Query experiments")
    exp_sub = p_experiments.add_subparsers(dest="exp_cmd", required=True)
    exp_list = exp_sub.add_parser("list", help="List experiments")
    exp_list.add_argument("--limit", type=int, default=20, help="Maximum number of results")
    exp_show = exp_sub.add_parser("show", help="Show experiment details")
    exp_show.add_argument("experiment_id", type=str, help="Experiment ID")

    p_runs = sub.add_parser("runs", help="Query runs")
    runs_sub = p_runs.add_subparsers(dest="runs_cmd", required=True)
    runs_list = runs_sub.add_parser("list", help="List runs")
    runs_list.add_argument("--limit", type=int, default=50, help="Maximum number of results")
    runs_list.add_argument("--status", type=str, help="Filter by status")
    runs_list.add_argument("--experiment", type=str, help="Filter by experiment_id")
    runs_show = runs_sub.add_parser("show", help="Show run details")
    runs_show.add_argument("run_id", type=str, help="Run ID")

    p_jobs = sub.add_parser("jobs", help="Query jobs")
    jobs_sub = p_jobs.add_subparsers(dest="jobs_cmd", required=True)
    jobs_list = jobs_sub.add_parser("list", help="List jobs")
    jobs_list.add_argument("--limit", type=int, default=50, help="Maximum number of results")
    jobs_list.add_argument("--status", type=str, help="Filter by status")
    jobs_show = jobs_sub.add_parser("show", help="Show job details")
    jobs_show.add_argument("job_id", type=str, help="Job ID")

    args = parser.parse_args()

    if args.cmd == "run":
        run(args.spec)
    elif args.cmd == "compare":
        report = compare_runs(args.run_a, args.run_b)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "score":
        report = compare_runs(args.run_a, args.run_b)
        profile = get_profile_or_load(args.profile)
        comps = extract_components(report, profile)
        scorecard = build_scorecard(comps, profile)
        
        output = {
            "run_a": args.run_a,
            "run_b": args.run_b,
            "profile": profile.name,
            "scorecard": scorecard.model_dump(),
            "risk_flags": report.get("risk_flags", []),
            "regressions": report.get("regressions", {}),
            "decision_components": comps.model_dump(),
        }
        print(json.dumps(output, indent=2, sort_keys=True))
    elif args.cmd == "rerun":
        rerun(args.run_id)
    elif args.cmd == "enqueue-run":
        with get_session() as session:
            job_id = enqueue_job_run_experiment(session, args.experiment_id)
        print(job_id)
    elif args.cmd == "enqueue-rerun":
        with get_session() as session:
            job_id = enqueue_job_rerun(session, args.run_id)
        print(job_id)
    elif args.cmd == "worker-once":
        processed = process_one_job()
        if processed:
            print("[OK] Processed 1 job")
        else:
            print("[INFO] No jobs available")
    elif args.cmd == "experiments":
        if args.exp_cmd == "list":
            with get_session() as session:
                experiments = list_experiments(session, limit=args.limit)
                if not experiments:
                    print("No experiments found.")
                else:
                    rows = []
                    for exp in experiments:
                        rows.append([
                            exp.experiment_id,
                            exp.metric,
                            str(exp.n_trials),
                            str(exp.seed),
                            short(exp.dataset_fingerprint, 10),
                            exp.created_at[:19] if len(exp.created_at) > 19 else exp.created_at,
                        ])
                    print(format_table(rows, [
                        "experiment_id",
                        "metric",
                        "n_trials",
                        "seed",
                        "dataset_fp",
                        "created_at",
                    ]))
        elif args.exp_cmd == "show":
            with get_session() as session:
                try:
                    exp = get_experiment(session, args.experiment_id)
                    print(f"experiment_id: {exp.experiment_id}")
                    print(f"name: {exp.name}")
                    print(f"task: {exp.task}")
                    print(f"metric: {exp.metric}")
                    print(f"n_trials: {exp.n_trials}")
                    print(f"seed: {exp.seed}")
                    print(f"dataset_fingerprint: {exp.dataset_fingerprint}")
                    print(f"dataset_schema_json: {exp.dataset_schema_json}")
                    print(f"scenarios_json: {exp.scenarios_json}")
                    print(f"created_at: {exp.created_at}")
                    spec_len = len(exp.spec_json) if exp.spec_json else 0
                    print(f"spec_json length: {spec_len}")
                except ValueError as e:
                    print(f"Error: {e}")
    elif args.cmd == "runs":
        if args.runs_cmd == "list":
            with get_session() as session:
                runs = list_runs(
                    session,
                    limit=args.limit,
                    status=args.status,
                    experiment_id=args.experiment,
                )
                if not runs:
                    print("No runs found.")
                else:
                    rows = []
                    for r in runs:
                        rows.append([
                            r.run_id,
                            r.experiment_id,
                            r.status,
                            r.created_at[:19] if len(r.created_at) > 19 else r.created_at,
                            r.rerun_of if r.rerun_of else "-",
                            r.results_path,
                        ])
                    print(format_table(rows, [
                        "run_id",
                        "experiment_id",
                        "status",
                        "created_at",
                        "rerun_of",
                        "results_path",
                    ]))
        elif args.runs_cmd == "show":
            with get_session() as session:
                try:
                    run = get_run(session, args.run_id)
                    print(f"run_id: {run.run_id}")
                    print(f"experiment_id: {run.experiment_id}")
                    print(f"status: {run.status}")
                    print(f"created_at: {run.created_at}")
                    print(f"started_at: {run.started_at if run.started_at else '-'}")
                    print(f"finished_at: {run.finished_at if run.finished_at else '-'}")
                    print(f"rerun_of: {run.rerun_of if run.rerun_of else '-'}")
                    print(f"results_path: {run.results_path}")
                    print(f"artifacts_dir: {run.artifacts_dir}")
                    if run.error:
                        print(f"error: {run.error}")
                    
                    artifacts = list_artifacts_for_run(session, args.run_id)
                    print("\nArtifacts:")
                    if artifacts:
                        for art in artifacts:
                            print(f"  - {art.path} ({art.kind})")
                    else:
                        print("  (none)")
                except ValueError as e:
                    print(f"Error: {e}")
    elif args.cmd == "jobs":
        if args.jobs_cmd == "list":
            with get_session() as session:
                jobs = list_jobs(session, limit=args.limit, status=args.status)
                if not jobs:
                    print("No jobs found.")
                else:
                    rows = []
                    for j in jobs:
                        rows.append([
                            j.job_id,
                            j.kind,
                            j.status,
                            j.created_at[:19] if len(j.created_at) > 19 else j.created_at,
                            j.started_at[:19] if j.started_at and len(j.started_at) > 19 else (j.started_at if j.started_at else "-"),
                            j.finished_at[:19] if j.finished_at and len(j.finished_at) > 19 else (j.finished_at if j.finished_at else "-"),
                            j.result_run_id if j.result_run_id else "-",
                        ])
                    print(format_table(rows, [
                        "job_id",
                        "kind",
                        "status",
                        "created_at",
                        "started_at",
                        "finished_at",
                        "result_run_id",
                    ]))
        elif args.jobs_cmd == "show":
            with get_session() as session:
                try:
                    job = get_job(session, args.job_id)
                    print(f"job_id: {job.job_id}")
                    print(f"kind: {job.kind}")
                    print(f"status: {job.status}")
                    if job.experiment_id:
                        print(f"experiment_id: {job.experiment_id}")
                    if job.run_id:
                        print(f"run_id: {job.run_id}")
                    print(f"created_at: {job.created_at}")
                    print(f"started_at: {job.started_at if job.started_at else '-'}")
                    print(f"finished_at: {job.finished_at if job.finished_at else '-'}")
                    print(f"result_run_id: {job.result_run_id if job.result_run_id else '-'}")
                    if job.error:
                        print(f"error: {job.error}")
                except ValueError as e:
                    print(f"Error: {e}")


if __name__ == "__main__":
    main()

