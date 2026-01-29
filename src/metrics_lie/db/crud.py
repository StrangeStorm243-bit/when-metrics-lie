from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from metrics_lie.experiments.definition import ExperimentDefinition
from metrics_lie.experiments.runs import RunRecord
from metrics_lie.schema import Artifact

from .models import Experiment, Run, Artifact as ArtifactModel


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_experiment(session: Session, exp_def: ExperimentDefinition, spec_json_str: str) -> None:
    """Insert or update an experiment definition."""
    existing = session.scalar(select(Experiment).where(Experiment.experiment_id == exp_def.experiment_id))
    
    if existing:
        # Update existing
        existing.name = exp_def.name
        existing.task = exp_def.task
        existing.metric = exp_def.metric
        existing.n_trials = exp_def.n_trials
        existing.seed = exp_def.seed
        existing.dataset_fingerprint = exp_def.dataset["fingerprint"]
        existing.dataset_schema_json = json.dumps(exp_def.dataset)
        existing.scenarios_json = json.dumps(exp_def.scenarios)
        existing.spec_schema_version = exp_def.spec_schema_version
        existing.spec_json = spec_json_str
    else:
        # Insert new
        experiment = Experiment(
            experiment_id=exp_def.experiment_id,
            name=exp_def.name,
            task=exp_def.task,
            metric=exp_def.metric,
            n_trials=exp_def.n_trials,
            seed=exp_def.seed,
            dataset_fingerprint=exp_def.dataset["fingerprint"],
            dataset_schema_json=json.dumps(exp_def.dataset),
            scenarios_json=json.dumps(exp_def.scenarios),
            created_at=exp_def.created_at,
            spec_schema_version=exp_def.spec_schema_version,
            spec_json=spec_json_str,
        )
        session.add(experiment)


def insert_run(session: Session, run_record: RunRecord) -> None:
    """Insert a new run record."""
    run = Run(
        run_id=run_record.run_id,
        experiment_id=run_record.experiment_id,
        status=run_record.status,
        created_at=run_record.created_at,
        started_at=run_record.started_at,
        finished_at=run_record.finished_at,
        results_path=run_record.results_path,
        artifacts_dir=run_record.artifacts_dir,
        seed_used=run_record.seed_used,
        error=run_record.error,
        rerun_of=getattr(run_record, "rerun_of", None),
    )
    session.add(run)


def update_run(session: Session, run_record: RunRecord) -> None:
    """Update an existing run record (for status transitions)."""
    run = session.scalar(select(Run).where(Run.run_id == run_record.run_id))
    if run:
        run.status = run_record.status
        run.started_at = run_record.started_at
        run.finished_at = run_record.finished_at
        run.error = run_record.error


def insert_artifacts(session: Session, run_id: str, artifacts: list[Artifact]) -> None:
    """Insert artifact rows for a run."""
    for artifact in artifacts:
        artifact_model = ArtifactModel(
            run_id=run_id,
            kind=artifact.kind,
            path=artifact.path,
            meta_json=json.dumps(artifact.meta),
            created_at=_now_iso(),
        )
        session.add(artifact_model)


def get_run_by_id(session: Session, run_id: str) -> Run:
    run = session.scalar(select(Run).where(Run.run_id == run_id))
    if run is None:
        raise ValueError(f"Run not found: {run_id}")
    return run


def get_results_path_for_run(session: Session, run_id: str) -> str:
    run = get_run_by_id(session, run_id)
    return run.results_path


def get_experiment_by_id(session: Session, experiment_id: str) -> Experiment:
    experiment = session.scalar(select(Experiment).where(Experiment.experiment_id == experiment_id))
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_id}")
    return experiment


def get_experiment_spec_json(session: Session, experiment_id: str) -> str:
    experiment = get_experiment_by_id(session, experiment_id)
    return experiment.spec_json


def get_experiment_id_for_run(session: Session, run_id: str) -> str:
    run = get_run_by_id(session, run_id)
    if run.experiment_id is None:
        raise ValueError(f"Run has no experiment_id: {run_id}")
    return run.experiment_id

