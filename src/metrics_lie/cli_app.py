"""Typer-based CLI for Spectra."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="spectra", help="Spectra — stress-test your ML models.")


def _version_callback(value: bool) -> None:
    if value:
        from metrics_lie import __version__

        typer.echo(f"Spectra {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Spectra — stress-test your ML models."""


@app.command()
def run(
    spec: str = typer.Argument(..., help="Path to experiment spec JSON file."),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override model path."
    ),
    task: Optional[str] = typer.Option(
        None, "--task", "-t", help="Override task type."
    ),
) -> None:
    """Run an experiment from a spec file."""
    from metrics_lie.execution import run_from_spec_dict

    spec_dict = json.loads(Path(spec).read_text(encoding="utf-8"))
    if model:
        if "model_source" not in spec_dict:
            spec_dict["model_source"] = {"kind": "pickle"}
        spec_dict["model_source"]["path"] = model
    if task:
        spec_dict["task"] = task

    run_id = run_from_spec_dict(spec_dict, spec_path_for_notes=spec)
    typer.echo(f"Run ID: {run_id}")


@app.command()
def evaluate(
    model: str = typer.Argument(..., help="Path to model file."),
    dataset: str = typer.Option(
        ..., "--dataset", "-d", help="Path to CSV dataset."
    ),
    metric: str = typer.Option("auc", "--metric", "-m", help="Primary metric."),
    task: str = typer.Option(
        "binary_classification", "--task", "-t", help="Task type."
    ),
    n_trials: int = typer.Option(
        200, "--trials", "-n", help="Monte Carlo trials."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
) -> None:
    """Quick evaluation: model + dataset -> results."""
    from metrics_lie.sdk import evaluate as sdk_evaluate

    result = sdk_evaluate(
        name=f"eval_{Path(model).stem}",
        dataset=dataset,
        model=model,
        metric=metric,
        task=task,
        n_trials=n_trials,
        seed=seed,
    )
    typer.echo(f"Run ID: {result.run_id}")
    typer.echo(f"Baseline {metric} = {result.baseline.mean:.6f}")


@app.command("compare")
def compare_cmd(
    run_a: str = typer.Argument(..., help="First run ID."),
    run_b: str = typer.Argument(..., help="Second run ID."),
    output_format: str = typer.Option(
        "json", "--format", "-f", help="Output format: json or table."
    ),
) -> None:
    """Compare two runs."""
    from metrics_lie.compare.compare import compare_runs

    report = compare_runs(run_a, run_b)
    if output_format == "json":
        typer.echo(json.dumps(report, indent=2, default=str))
    else:
        _print_compare_table(report)


@app.command("score")
def score_cmd(
    run_a: str = typer.Argument(..., help="First run ID."),
    run_b: str = typer.Argument(..., help="Second run ID."),
    profile: str = typer.Option(
        "balanced", "--profile", "-p", help="Decision profile name."
    ),
) -> None:
    """Score a comparison with a decision profile."""
    from metrics_lie.compare.compare import compare_runs
    from metrics_lie.decision.extract import extract_components
    from metrics_lie.decision.scorecard import build_scorecard
    from metrics_lie.profiles.load import get_profile_or_load

    report = compare_runs(run_a, run_b)
    prof = get_profile_or_load(profile)
    components = extract_components(report, prof)
    card = build_scorecard(components, prof)
    typer.echo(json.dumps(card.model_dump(), indent=2, default=str))


@app.command()
def rerun(run_id: str = typer.Argument(..., help="Run ID to rerun.")) -> None:
    """Deterministic rerun of a completed run."""
    from metrics_lie.execution import rerun as _rerun

    new_id = _rerun(run_id)
    typer.echo(f"Rerun complete. New run ID: {new_id}")


@app.command("enqueue-run")
def enqueue_run(
    experiment_id: str = typer.Argument(..., help="Experiment ID."),
) -> None:
    """Queue a run job for an experiment."""
    from metrics_lie.db.crud import enqueue_job_run_experiment
    from metrics_lie.db.session import get_session

    with get_session() as session:
        job_id = enqueue_job_run_experiment(session, experiment_id)
    typer.echo(job_id)


@app.command("enqueue-rerun")
def enqueue_rerun(
    run_id: str = typer.Argument(..., help="Run ID to rerun."),
) -> None:
    """Queue a rerun job."""
    from metrics_lie.db.crud import enqueue_job_rerun
    from metrics_lie.db.session import get_session

    with get_session() as session:
        job_id = enqueue_job_rerun(session, run_id)
    typer.echo(job_id)


@app.command("worker-once")
def worker_once() -> None:
    """Process one job from the queue."""
    from metrics_lie.worker import process_one_job

    processed = process_one_job()
    if processed:
        typer.echo("[OK] Processed 1 job")
    else:
        typer.echo("[INFO] No jobs available")


# --- Catalog subcommands ---

metrics_app = typer.Typer(help="List and explore available metrics.")
app.add_typer(metrics_app, name="metrics")

scenarios_app = typer.Typer(help="List and explore available scenarios.")
app.add_typer(scenarios_app, name="scenarios")

models_app = typer.Typer(help="List supported model formats.")
app.add_typer(models_app, name="models")


# --- DB query subcommands ---

experiments_app = typer.Typer(help="Query experiments.")
app.add_typer(experiments_app, name="experiments")

runs_app = typer.Typer(help="Query runs.")
app.add_typer(runs_app, name="runs")

jobs_app = typer.Typer(help="Query jobs.")
app.add_typer(jobs_app, name="jobs")


@metrics_app.command("list")
def metrics_list(
    task: Optional[str] = typer.Option(
        None, "--task", "-t", help="Filter by task type."
    ),
) -> None:
    """List available metrics."""
    from metrics_lie.catalog import list_metrics
    from metrics_lie.metrics.registry import METRIC_DIRECTION

    metrics = list_metrics(task=task)
    for m in metrics:
        direction = (
            "higher-is-better"
            if METRIC_DIRECTION.get(m, True)
            else "lower-is-better"
        )
        typer.echo(f"  {m:<25} {direction}")


@scenarios_app.command("list")
def scenarios_list(
    task: Optional[str] = typer.Option(
        None, "--task", "-t", help="Filter by task type."
    ),
) -> None:
    """List available scenarios."""
    from metrics_lie.catalog import list_scenarios

    for s in list_scenarios(task=task):
        typer.echo(f"  {s}")


@models_app.command("list")
def models_list() -> None:
    """List supported model formats."""
    from metrics_lie.catalog import list_model_formats

    for fmt in list_model_formats():
        typer.echo(f"  {fmt}")


@experiments_app.command("list")
def experiments_list_cmd(
    limit: int = typer.Option(20, "--limit", "-l", help="Max results."),
) -> None:
    """List experiments."""
    from metrics_lie.db.crud import list_experiments
    from metrics_lie.db.session import get_session

    with get_session() as session:
        rows = list_experiments(session, limit=limit)
    for r in rows:
        typer.echo(f"  {r.experiment_id}  {r.name}")


@experiments_app.command("show")
def experiments_show_cmd(
    experiment_id: str = typer.Argument(..., help="Experiment ID."),
) -> None:
    """Show experiment details."""
    from metrics_lie.db.crud import get_experiment
    from metrics_lie.db.session import get_session

    with get_session() as session:
        try:
            exp = get_experiment(session, experiment_id)
        except ValueError:
            typer.echo(f"Experiment {experiment_id} not found.")
            raise typer.Exit(code=1)
    typer.echo(
        json.dumps(
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "task": exp.task,
                "metric": exp.metric,
            },
            indent=2,
        )
    )


@runs_app.command("list")
def runs_list_cmd(
    limit: int = typer.Option(20, "--limit", "-l", help="Max results."),
    status: Optional[str] = typer.Option(
        None, "--status", "-s", help="Filter by status."
    ),
    experiment: Optional[str] = typer.Option(
        None, "--experiment", "-e", help="Filter by experiment ID."
    ),
) -> None:
    """List runs."""
    from metrics_lie.db.crud import list_runs
    from metrics_lie.db.session import get_session

    with get_session() as session:
        rows = list_runs(
            session, limit=limit, status=status, experiment_id=experiment
        )
    for r in rows:
        typer.echo(f"  {r.run_id}  {r.status}  {r.experiment_id}")


@runs_app.command("show")
def runs_show_cmd(
    run_id: str = typer.Argument(..., help="Run ID."),
) -> None:
    """Show run details."""
    from metrics_lie.db.crud import get_run
    from metrics_lie.db.session import get_session

    with get_session() as session:
        try:
            r = get_run(session, run_id)
        except ValueError:
            typer.echo(f"Run {run_id} not found.")
            raise typer.Exit(code=1)
    typer.echo(
        json.dumps(
            {
                "run_id": r.run_id,
                "experiment_id": r.experiment_id,
                "status": r.status,
            },
            indent=2,
        )
    )


@jobs_app.command("list")
def jobs_list_cmd(
    limit: int = typer.Option(20, "--limit", "-l", help="Max results."),
    status: Optional[str] = typer.Option(
        None, "--status", "-s", help="Filter by status."
    ),
) -> None:
    """List jobs."""
    from metrics_lie.db.crud import list_jobs
    from metrics_lie.db.session import get_session

    with get_session() as session:
        rows = list_jobs(session, limit=limit, status=status)
    for r in rows:
        typer.echo(f"  {r.job_id}  {r.status}  {r.kind}")


@jobs_app.command("show")
def jobs_show_cmd(
    job_id: str = typer.Argument(..., help="Job ID."),
) -> None:
    """Show job details."""
    from metrics_lie.db.crud import get_job
    from metrics_lie.db.session import get_session

    with get_session() as session:
        try:
            j = get_job(session, job_id)
        except ValueError:
            typer.echo(f"Job {job_id} not found.")
            raise typer.Exit(code=1)
    typer.echo(
        json.dumps(
            {"job_id": j.job_id, "status": j.status, "kind": j.kind},
            indent=2,
        )
    )


def _print_compare_table(report: dict) -> None:
    """Print a comparison report as a formatted table."""
    typer.echo(f"Metric: {report.get('metric_name', '?')}")
    bd = report.get("baseline_delta", {})
    typer.echo(
        f"Baseline delta: {bd.get('mean', 0):+.4f}"
        f" (A={bd.get('a', 0):.4f}, B={bd.get('b', 0):.4f})"
    )
    decision = report.get("decision", {})
    typer.echo(
        f"Winner: {decision.get('winner', '?')}"
        f" ({decision.get('confidence', '?')})"
    )
    typer.echo(f"Reasoning: {decision.get('reasoning', '?')}")


if __name__ == "__main__":
    app()
