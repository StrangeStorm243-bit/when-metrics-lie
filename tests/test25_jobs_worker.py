import json
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from metrics_lie.db.models import Base, Experiment, Job, Run
from metrics_lie.db.crud import (
    enqueue_job_run_experiment,
    get_experiment_spec_json,
)
from metrics_lie.experiments.definition import ExperimentDefinition
from metrics_lie.experiments.identity import canonical_json
from metrics_lie.spec import load_experiment_spec
from metrics_lie.experiments.datasets import dataset_fingerprint_csv
from metrics_lie.worker import process_one_job


def test_job_enqueue_and_process(tmp_path, monkeypatch):
    """
    Test that enqueueing a job and processing it creates a run.
    """
    # Use the real example spec as the source of truth.
    spec_path = Path("examples/experiment_minimal.json")
    spec_dict = json.loads(spec_path.read_text())
    spec = load_experiment_spec(spec_dict)
    
    # Create a temporary DB
    db_path = tmp_path / "test25_jobs.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    
    # Create experiment in DB
    dataset_fp = dataset_fingerprint_csv(spec.dataset.path)
    exp_def = ExperimentDefinition.from_spec(spec, dataset_fingerprint=dataset_fp)
    spec_json_str = canonical_json(spec_dict)
    
    session = SessionLocal()
    try:
        from metrics_lie.db.crud import upsert_experiment
        upsert_experiment(session, exp_def, spec_json_str)
        session.commit()
        
        # Enqueue a job
        job_id = enqueue_job_run_experiment(session, exp_def.experiment_id)
        session.commit()
        
        # Verify job is queued
        job = session.query(Job).filter(Job.job_id == job_id).one()
        assert job.status == "queued"
        assert job.kind == "run_experiment"
        assert job.experiment_id == exp_def.experiment_id
    finally:
        session.close()
    
    # Patch get_session to use our temporary DB
    def fake_get_session():
        class _Ctx:
            def __enter__(self_inner):
                self_inner.session = SessionLocal()
                return self_inner.session
            
            def __exit__(self_inner, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    self_inner.session.commit()
                else:
                    self_inner.session.rollback()
                self_inner.session.close()
        
        return _Ctx()
    
    # Patch execution, worker, and session modules
    import metrics_lie.execution as execution_mod
    import metrics_lie.worker as worker_mod
    import metrics_lie.db.session as session_mod
    
    monkeypatch.setattr(execution_mod, "get_session", fake_get_session)
    monkeypatch.setattr(worker_mod, "get_session", fake_get_session)
    monkeypatch.setattr(session_mod, "get_session", fake_get_session)
    
    # Process the job
    processed = process_one_job()
    assert processed == 1
    
    # Verify job is completed
    session = SessionLocal()
    try:
        job = session.query(Job).filter(Job.job_id == job_id).one()
        assert job.status == "completed"
        assert job.result_run_id is not None
        assert job.started_at is not None
        assert job.finished_at is not None
        
        # Verify run exists
        run = session.query(Run).filter(Run.run_id == job.result_run_id).one()
        assert run.experiment_id == exp_def.experiment_id
        assert run.status == "completed"
        
        # Verify results.json exists
        results_path = Path(run.results_path)
        assert results_path.exists()
    finally:
        session.close()

