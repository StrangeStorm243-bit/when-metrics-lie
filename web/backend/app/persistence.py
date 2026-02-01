"""File-based persistence for experiments and results."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .contracts import ExperimentCreateRequest, ExperimentSummary, ResultSummary


def get_experiments_dir() -> Path:
    """Get the experiments persistence directory."""
    repo_root = Path(__file__).parent.parent.parent.parent
    experiments_dir = repo_root / ".spectra_ui" / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    return experiments_dir


def get_experiment_dir(experiment_id: str) -> Path:
    """Get directory for a specific experiment."""
    experiments_dir = get_experiments_dir()
    exp_dir = experiments_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_experiment(experiment_id: str, create_req: ExperimentCreateRequest, summary: ExperimentSummary) -> None:
    """Save experiment metadata to disk."""
    exp_dir = get_experiment_dir(experiment_id)
    experiment_file = exp_dir / "experiment.json"

    data = {
        "create_request": create_req.model_dump(),
        "summary": summary.model_dump(mode="json"),
    }

    experiment_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_experiment(experiment_id: str) -> tuple[ExperimentCreateRequest, ExperimentSummary]:
    """Load experiment metadata from disk."""
    exp_dir = get_experiment_dir(experiment_id)
    experiment_file = exp_dir / "experiment.json"

    if not experiment_file.exists():
        raise FileNotFoundError(f"Experiment {experiment_id} not found")

    data = json.loads(experiment_file.read_text(encoding="utf-8"))
    create_req = ExperimentCreateRequest.model_validate(data["create_request"])
    summary = ExperimentSummary.model_validate(data["summary"])

    return create_req, summary


def list_experiments() -> list[ExperimentSummary]:
    """List all persisted experiments."""
    experiments_dir = get_experiments_dir()
    summaries = []

    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        experiment_file = exp_dir / "experiment.json"
        if not experiment_file.exists():
            continue

        try:
            _, summary = load_experiment(exp_dir.name)
            summaries.append(summary)
        except Exception:
            # Skip corrupted experiments
            continue

    return summaries


def save_result(experiment_id: str, run_id: str, result: ResultSummary) -> None:
    """Save result to disk."""
    exp_dir = get_experiment_dir(experiment_id)
    runs_dir = exp_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    result_file = run_dir / "result.json"
    result_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def load_latest_result(experiment_id: str) -> Optional[ResultSummary]:
    """Load the latest result for an experiment."""
    exp_dir = get_experiment_dir(experiment_id)
    runs_dir = exp_dir / "runs"

    if not runs_dir.exists():
        return None

    # Find the most recent run
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None

    # Sort by modification time, most recent first
    run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    for run_dir in run_dirs:
        result_file = run_dir / "result.json"
        if result_file.exists():
            try:
                data = json.loads(result_file.read_text(encoding="utf-8"))
                return ResultSummary.model_validate(data)
            except Exception:
                continue

    return None


def list_runs(experiment_id: str) -> list[dict]:
    """List all runs for an experiment."""
    exp_dir = get_experiment_dir(experiment_id)
    runs_dir = exp_dir / "runs"

    if not runs_dir.exists():
        return []

    runs = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        run_id = run_dir.name
        result_file = run_dir / "result.json"

        if not result_file.exists():
            continue

        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            result = ResultSummary.model_validate(data)
            runs.append({
                "run_id": run_id,
                "generated_at": result.generated_at.isoformat() if result.generated_at else None,
            })
        except Exception:
            # Skip corrupted results
            continue

    # Sort by generated_at descending (most recent first), then by run_id
    # Put None values at the end
    runs_with_dates = [r for r in runs if r["generated_at"] is not None]
    runs_without_dates = [r for r in runs if r["generated_at"] is None]
    
    # Sort runs with dates: descending by generated_at, then by run_id
    runs_with_dates.sort(key=lambda r: (r["generated_at"], r["run_id"]), reverse=True)
    
    # Sort runs without dates: by run_id
    runs_without_dates.sort(key=lambda r: r["run_id"])
    
    # Combine: dates first (descending), then None values
    runs[:] = runs_with_dates + runs_without_dates

    return runs


def load_result_for_run(experiment_id: str, run_id: str) -> ResultSummary:
    """Load result for a specific run."""
    exp_dir = get_experiment_dir(experiment_id)
    runs_dir = exp_dir / "runs"

    if not runs_dir.exists():
        raise FileNotFoundError(f"Experiment {experiment_id} has no runs")

    run_dir = runs_dir / run_id
    result_file = run_dir / "result.json"

    if not result_file.exists():
        raise FileNotFoundError(f"Run {run_id} not found for experiment {experiment_id}")

    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
        return ResultSummary.model_validate(data)
    except Exception as e:
        raise ValueError(f"Failed to load result for run {run_id}: {e}") from e

