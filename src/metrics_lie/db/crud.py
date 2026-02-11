from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from metrics_lie.experiments.definition import ExperimentDefinition
from metrics_lie.experiments.runs import RunRecord
from metrics_lie.schema import Artifact

from .models import Experiment, Run, Artifact as ArtifactModel, Job


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_experiment(
    session: Session, exp_def: ExperimentDefinition, spec_json_str: str
) -> None:
    """Insert or update an experiment definition."""
    existing = session.scalar(
        select(Experiment).where(Experiment.experiment_id == exp_def.experiment_id)
    )

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
    experiment = session.scalar(
        select(Experiment).where(Experiment.experiment_id == experiment_id)
    )
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


def enqueue_job_run_experiment(session: Session, experiment_id: str) -> str:
    """Enqueue a job to run an experiment. Returns the job_id."""
    import uuid

    job_id = uuid.uuid4().hex[:10].upper()
    job = Job(
        job_id=job_id,
        kind="run_experiment",
        experiment_id=experiment_id,
        run_id=None,
        status="queued",
        created_at=_now_iso(),
        started_at=None,
        finished_at=None,
        error=None,
        result_run_id=None,
    )
    session.add(job)
    return job_id


def enqueue_job_rerun(session: Session, run_id: str) -> str:
    """Enqueue a job to rerun a run. Returns the job_id."""
    import uuid

    job_id = uuid.uuid4().hex[:10].upper()
    job = Job(
        job_id=job_id,
        kind="rerun_run",
        experiment_id=None,
        run_id=run_id,
        status="queued",
        created_at=_now_iso(),
        started_at=None,
        finished_at=None,
        error=None,
        result_run_id=None,
    )
    session.add(job)
    return job_id


def claim_next_job(session: Session) -> Job | None:
    """Claim the oldest queued job and mark it as running. Returns the job or None."""
    job = session.scalar(
        select(Job).where(Job.status == "queued").order_by(Job.created_at).limit(1)
    )
    if job is None:
        return None

    job.status = "running"
    job.started_at = _now_iso()
    session.commit()
    return job


def mark_job_completed(session: Session, job_id: str, result_run_id: str) -> None:
    """Mark a job as completed with the resulting run_id."""
    job = session.scalar(select(Job).where(Job.job_id == job_id))
    if job is None:
        raise ValueError(f"Job not found: {job_id}")
    job.status = "completed"
    job.finished_at = _now_iso()
    job.result_run_id = result_run_id


def mark_job_failed(session: Session, job_id: str, error_msg: str) -> None:
    """Mark a job as failed with an error message."""
    job = session.scalar(select(Job).where(Job.job_id == job_id))
    if job is None:
        raise ValueError(f"Job not found: {job_id}")
    job.status = "failed"
    job.finished_at = _now_iso()
    job.error = error_msg


# Phase 2.6: Read helpers for CLI queries


def list_experiments(session: Session, limit: int = 20) -> list[Experiment]:
    """List experiments, ordered by created_at descending."""
    stmt = select(Experiment).order_by(Experiment.created_at.desc()).limit(limit)
    return list(session.scalars(stmt).all())


def get_experiment(session: Session, experiment_id: str) -> Experiment:
    """Get an experiment by ID. Raises ValueError if not found."""
    return get_experiment_by_id(session, experiment_id)


def list_runs(
    session: Session,
    limit: int = 50,
    status: str | None = None,
    experiment_id: str | None = None,
) -> list[Run]:
    """List runs with optional filters, ordered by created_at descending."""
    stmt = select(Run).order_by(Run.created_at.desc())
    if status is not None:
        stmt = stmt.where(Run.status == status)
    if experiment_id is not None:
        stmt = stmt.where(Run.experiment_id == experiment_id)
    stmt = stmt.limit(limit)
    return list(session.scalars(stmt).all())


def get_run(session: Session, run_id: str) -> Run:
    """Get a run by ID. Raises ValueError if not found."""
    return get_run_by_id(session, run_id)


def list_jobs(
    session: Session, limit: int = 50, status: str | None = None
) -> list[Job]:
    """List jobs with optional status filter, ordered by created_at descending."""
    stmt = select(Job).order_by(Job.created_at.desc())
    if status is not None:
        stmt = stmt.where(Job.status == status)
    stmt = stmt.limit(limit)
    return list(session.scalars(stmt).all())


def get_job(session: Session, job_id: str) -> Job:
    """Get a job by ID. Raises ValueError if not found."""
    job = session.scalar(select(Job).where(Job.job_id == job_id))
    if job is None:
        raise ValueError(f"Job not found: {job_id}")
    return job


def list_artifacts_for_run(session: Session, run_id: str) -> list[ArtifactModel]:
    """List all artifacts for a run."""
    stmt = (
        select(ArtifactModel)
        .where(ArtifactModel.run_id == run_id)
        .order_by(ArtifactModel.created_at)
    )
    return list(session.scalars(stmt).all())
