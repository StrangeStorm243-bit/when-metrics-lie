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
)
from metrics_lie.worker import process_one_job
from metrics_lie.compare.compare import compare_runs
from metrics_lie.execution import run_from_spec_dict, rerun


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

    p_rerun = sub.add_parser("rerun", help="Deterministically rerun an experiment by run_id using stored spec")
    p_rerun.add_argument("run_id", type=str, help="Existing run ID to rerun")

    p_enqueue_run = sub.add_parser("enqueue-run", help="Enqueue a job to run an experiment")
    p_enqueue_run.add_argument("experiment_id", type=str, help="Experiment ID to run")

    p_enqueue_rerun = sub.add_parser("enqueue-rerun", help="Enqueue a job to rerun a run")
    p_enqueue_rerun.add_argument("run_id", type=str, help="Run ID to rerun")

    p_worker_once = sub.add_parser("worker-once", help="Process one job from the queue and exit")

    args = parser.parse_args()

    if args.cmd == "run":
        run(args.spec)
    elif args.cmd == "compare":
        report = compare_runs(args.run_a, args.run_b)
        print(json.dumps(report, indent=2, sort_keys=True))
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


if __name__ == "__main__":
    main()

