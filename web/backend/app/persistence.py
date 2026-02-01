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

