import json
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from metrics_lie.db.models import Base, Experiment, Run
from metrics_lie.db.crud import (
    get_experiment_id_for_run,
    get_experiment_spec_json,
)
from metrics_lie.cli import run_from_spec_dict


def _make_temp_session(tmp_path):
    db_path = tmp_path / "test_rerun.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_experiment_spec_json_and_run_linkage(tmp_path):
    # Minimal smoke test for helper functions using an isolated in-memory-like SQLite DB.
    session = _make_temp_session(tmp_path)

    # Insert an experiment and a run linked to it.
    exp = Experiment(
        experiment_id="exp_TEST12345",
        name="test",
        task="binary_classification",
        metric="auc",
        n_trials=1,
        seed=42,
        dataset_fingerprint="fp",
        dataset_schema_json="{}",
        scenarios_json="[]",
        created_at="2026-01-01T00:00:00Z",
        spec_schema_version=None,
        spec_json='{"foo":"bar"}',
    )
    session.add(exp)
    run = Run(
        run_id="RUN123",
        experiment_id=exp.experiment_id,
        status="completed",
        created_at="2026-01-01T00:00:00Z",
        started_at=None,
        finished_at=None,
        results_path="runs/RUN123/results.json",
        artifacts_dir="runs/RUN123/artifacts",
        seed_used=42,
        error=None,
        rerun_of=None,
    )
    session.add(run)
    session.commit()

    # Helpers should round-trip IDs and spec_json correctly.
    assert get_experiment_id_for_run(session, "RUN123") == "exp_TEST12345"
    assert get_experiment_spec_json(session, "exp_TEST12345") == '{"foo":"bar"}'


def test_run_from_spec_dict_uses_same_experiment_id(tmp_path, monkeypatch):
    """
    Ensure that re-running from the same spec dict produces a run
    with an experiment linked to the same experiment_id semantics.
    """
    # Use the real example spec as the source of truth.
    spec_path = Path("examples/experiment_minimal.json")
    spec_dict = json.loads(spec_path.read_text())

    created_runs = []

    def fake_get_session():
        # Use an isolated SQLite file for this test.
        engine = create_engine(f"sqlite:///{tmp_path/'phase24_rerun.db'}")
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)

        class _Ctx:
            def __enter__(self_inner):
                self_inner.session = SessionLocal()
                return self_inner.session

            def __exit__(self_inner, exc_type, exc_val, exc_tb):
                self_inner.session.close()

        return _Ctx()

    # Patch the CLI to use the temporary DB instead of the global one.
    import metrics_lie.cli as cli_mod

    monkeypatch.setattr(cli_mod, "get_session", fake_get_session)

    # First run establishes the experiment in the temp DB.
    run_id_1 = run_from_spec_dict(spec_dict, spec_path_for_notes=str(spec_path))
    created_runs.append(run_id_1)

    # Second run from the same dict should reuse the same experiment_id semantics.
    run_id_2 = run_from_spec_dict(spec_dict, spec_path_for_notes=str(spec_path))
    created_runs.append(run_id_2)

    # Inspect the DB to confirm both runs point at the same experiment_id.
    engine = create_engine(f"sqlite:///{tmp_path/'phase24_rerun.db'}")
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        runs = session.query(Run).all()
        assert {r.run_id for r in runs} == set(created_runs)
        exp_ids = {r.experiment_id for r in runs}
        assert len(exp_ids) == 1
    finally:
        session.close()


