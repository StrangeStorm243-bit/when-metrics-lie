"""Tests for Phase 2.6 DB read helpers."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from metrics_lie.db.models import Base, Experiment, Run, Job, Artifact as ArtifactModel
from metrics_lie.db.crud import (
    list_experiments,
    get_experiment,
    list_runs,
    get_run,
    list_jobs,
    get_job,
    list_artifacts_for_run,
)


def test_list_and_get_experiments(tmp_path):
    """Test list_experiments and get_experiment."""
    db_path = tmp_path / "test26_queries.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    try:
        # Insert test experiment
        exp = Experiment(
            experiment_id="exp_TEST26",
            name="test",
            task="binary_classification",
            metric="auc",
            n_trials=100,
            seed=42,
            dataset_fingerprint="fp123",
            dataset_schema_json="{}",
            scenarios_json="[]",
            created_at="2026-01-29T00:00:00Z",
            spec_schema_version=None,
            spec_json='{"test": "data"}',
        )
        session.add(exp)
        session.commit()
        
        # Test list_experiments
        experiments = list_experiments(session, limit=10)
        assert len(experiments) == 1
        assert experiments[0].experiment_id == "exp_TEST26"
        
        # Test get_experiment
        exp_found = get_experiment(session, "exp_TEST26")
        assert exp_found.experiment_id == "exp_TEST26"
        assert exp_found.name == "test"
        
        # Test get_experiment not found
        try:
            get_experiment(session, "exp_NOTFOUND")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
    finally:
        session.close()


def test_list_and_get_runs(tmp_path):
    """Test list_runs and get_run."""
    db_path = tmp_path / "test26_runs.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    try:
        # Insert experiment first
        exp = Experiment(
            experiment_id="exp_TEST26",
            name="test",
            task="binary_classification",
            metric="auc",
            n_trials=100,
            seed=42,
            dataset_fingerprint="fp123",
            dataset_schema_json="{}",
            scenarios_json="[]",
            created_at="2026-01-29T00:00:00Z",
            spec_schema_version=None,
            spec_json="{}",
        )
        session.add(exp)
        
        # Insert runs
        run1 = Run(
            run_id="RUN001",
            experiment_id="exp_TEST26",
            status="completed",
            created_at="2026-01-29T00:00:00Z",
            started_at="2026-01-29T00:01:00Z",
            finished_at="2026-01-29T00:02:00Z",
            results_path="runs/RUN001/results.json",
            artifacts_dir="runs/RUN001/artifacts",
            seed_used=42,
            error=None,
            rerun_of=None,
        )
        run2 = Run(
            run_id="RUN002",
            experiment_id="exp_TEST26",
            status="running",
            created_at="2026-01-29T00:03:00Z",
            started_at="2026-01-29T00:04:00Z",
            finished_at=None,
            results_path="runs/RUN002/results.json",
            artifacts_dir="runs/RUN002/artifacts",
            seed_used=42,
            error=None,
            rerun_of="RUN001",
        )
        session.add(run1)
        session.add(run2)
        session.commit()
        
        # Test list_runs (all)
        runs = list_runs(session, limit=10)
        assert len(runs) == 2
        # Should be ordered by created_at desc, so RUN002 first
        assert runs[0].run_id == "RUN002"
        assert runs[1].run_id == "RUN001"
        
        # Test list_runs with status filter
        runs_completed = list_runs(session, limit=10, status="completed")
        assert len(runs_completed) == 1
        assert runs_completed[0].run_id == "RUN001"
        
        # Test list_runs with experiment filter
        runs_exp = list_runs(session, limit=10, experiment_id="exp_TEST26")
        assert len(runs_exp) == 2
        
        # Test get_run
        run_found = get_run(session, "RUN001")
        assert run_found.run_id == "RUN001"
        assert run_found.status == "completed"
        
        # Test get_run not found
        try:
            get_run(session, "RUN_NOTFOUND")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
    finally:
        session.close()


def test_list_and_get_jobs(tmp_path):
    """Test list_jobs and get_job."""
    db_path = tmp_path / "test26_jobs.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    try:
        # Insert experiment
        exp = Experiment(
            experiment_id="exp_TEST26",
            name="test",
            task="binary_classification",
            metric="auc",
            n_trials=100,
            seed=42,
            dataset_fingerprint="fp123",
            dataset_schema_json="{}",
            scenarios_json="[]",
            created_at="2026-01-29T00:00:00Z",
            spec_schema_version=None,
            spec_json="{}",
        )
        session.add(exp)
        
        # Insert jobs
        job1 = Job(
            job_id="JOB001",
            kind="run_experiment",
            experiment_id="exp_TEST26",
            run_id=None,
            status="completed",
            created_at="2026-01-29T00:00:00Z",
            started_at="2026-01-29T00:01:00Z",
            finished_at="2026-01-29T00:02:00Z",
            error=None,
            result_run_id="RUN001",
        )
        job2 = Job(
            job_id="JOB002",
            kind="rerun_run",
            experiment_id=None,
            run_id="RUN001",
            status="queued",
            created_at="2026-01-29T00:03:00Z",
            started_at=None,
            finished_at=None,
            error=None,
            result_run_id=None,
        )
        session.add(job1)
        session.add(job2)
        session.commit()
        
        # Test list_jobs (all)
        jobs = list_jobs(session, limit=10)
        assert len(jobs) == 2
        # Should be ordered by created_at desc, so JOB002 first
        assert jobs[0].job_id == "JOB002"
        assert jobs[1].job_id == "JOB001"
        
        # Test list_jobs with status filter
        jobs_queued = list_jobs(session, limit=10, status="queued")
        assert len(jobs_queued) == 1
        assert jobs_queued[0].job_id == "JOB002"
        
        # Test get_job
        job_found = get_job(session, "JOB001")
        assert job_found.job_id == "JOB001"
        assert job_found.kind == "run_experiment"
        assert job_found.result_run_id == "RUN001"
        
        # Test get_job not found
        try:
            get_job(session, "JOB_NOTFOUND")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
    finally:
        session.close()


def test_list_artifacts_for_run(tmp_path):
    """Test list_artifacts_for_run."""
    db_path = tmp_path / "test26_artifacts.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    try:
        # Insert experiment and run
        exp = Experiment(
            experiment_id="exp_TEST26",
            name="test",
            task="binary_classification",
            metric="auc",
            n_trials=100,
            seed=42,
            dataset_fingerprint="fp123",
            dataset_schema_json="{}",
            scenarios_json="[]",
            created_at="2026-01-29T00:00:00Z",
            spec_schema_version=None,
            spec_json="{}",
        )
        session.add(exp)
        
        run = Run(
            run_id="RUN001",
            experiment_id="exp_TEST26",
            status="completed",
            created_at="2026-01-29T00:00:00Z",
            started_at=None,
            finished_at=None,
            results_path="runs/RUN001/results.json",
            artifacts_dir="runs/RUN001/artifacts",
            seed_used=42,
            error=None,
            rerun_of=None,
        )
        session.add(run)
        
        # Insert artifacts
        art1 = ArtifactModel(
            run_id="RUN001",
            kind="plot",
            path="artifacts/plot1.png",
            meta_json='{"type": "calibration"}',
            created_at="2026-01-29T00:00:00Z",
        )
        art2 = ArtifactModel(
            run_id="RUN001",
            kind="plot",
            path="artifacts/plot2.png",
            meta_json='{"type": "distribution"}',
            created_at="2026-01-29T00:01:00Z",
        )
        session.add(art1)
        session.add(art2)
        session.commit()
        
        # Test list_artifacts_for_run
        artifacts = list_artifacts_for_run(session, "RUN001")
        assert len(artifacts) == 2
        assert artifacts[0].path == "artifacts/plot1.png"
        assert artifacts[1].path == "artifacts/plot2.png"
        
        # Test empty list for non-existent run
        artifacts_empty = list_artifacts_for_run(session, "RUN_NOTFOUND")
        assert len(artifacts_empty) == 0
    finally:
        session.close()

