"""Persistence layer for experiments and results.

Dispatches between two backends based on SPECTRA_STORAGE_BACKEND env var:
- 'local' (default): file-based persistence under .spectra_ui/
- 'supabase': Supabase Postgres for metadata + Supabase Storage for artifacts

When running locally, the owner_id parameter is accepted but ignored,
preserving Phase 4 behavior exactly.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .contracts import ExperimentCreateRequest, ExperimentSummary, ResultSummary


# ---------------------------------------------------------------------------
# Configuration helper
# ---------------------------------------------------------------------------


def _is_hosted() -> bool:
    from .config import get_settings

    return get_settings().is_hosted


# ---------------------------------------------------------------------------
# Local filesystem implementation (Phase 4 behavior, unchanged)
# ---------------------------------------------------------------------------


def _get_experiments_dir() -> Path:
    """Get the experiments persistence directory."""
    repo_root = Path(__file__).parent.parent.parent.parent
    experiments_dir = repo_root / ".spectra_ui" / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    return experiments_dir


def _get_experiment_dir(experiment_id: str) -> Path:
    """Get directory for a specific experiment."""
    experiments_dir = _get_experiments_dir()
    exp_dir = experiments_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def _save_experiment_local(
    experiment_id: str, create_req: ExperimentCreateRequest, summary: ExperimentSummary
) -> None:
    exp_dir = _get_experiment_dir(experiment_id)
    experiment_file = exp_dir / "experiment.json"
    data = {
        "create_request": create_req.model_dump(),
        "summary": summary.model_dump(mode="json"),
    }
    experiment_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_experiment_local(
    experiment_id: str,
) -> tuple[ExperimentCreateRequest, ExperimentSummary]:
    exp_dir = _get_experiment_dir(experiment_id)
    experiment_file = exp_dir / "experiment.json"
    if not experiment_file.exists():
        raise FileNotFoundError(f"Experiment {experiment_id} not found")
    data = json.loads(experiment_file.read_text(encoding="utf-8"))
    create_req = ExperimentCreateRequest.model_validate(data["create_request"])
    summary = ExperimentSummary.model_validate(data["summary"])
    return create_req, summary


def _list_experiments_local() -> list[ExperimentSummary]:
    experiments_dir = _get_experiments_dir()
    summaries = []
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        experiment_file = exp_dir / "experiment.json"
        if not experiment_file.exists():
            continue
        try:
            _, summary = _load_experiment_local(exp_dir.name)
            summaries.append(summary)
        except Exception:
            continue
    return summaries


def _save_result_local(experiment_id: str, run_id: str, result: ResultSummary) -> None:
    exp_dir = _get_experiment_dir(experiment_id)
    runs_dir = exp_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result_file = run_dir / "result.json"
    result_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def _load_latest_result_local(experiment_id: str) -> Optional[ResultSummary]:
    exp_dir = _get_experiment_dir(experiment_id)
    runs_dir = exp_dir / "runs"
    if not runs_dir.exists():
        return None
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
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


def _list_runs_local(experiment_id: str) -> list[dict]:
    exp_dir = _get_experiment_dir(experiment_id)
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
            runs.append(
                {
                    "run_id": run_id,
                    "generated_at": result.generated_at.isoformat()
                    if result.generated_at
                    else None,
                }
            )
        except Exception:
            continue
    runs_with_dates = [r for r in runs if r["generated_at"] is not None]
    runs_without_dates = [r for r in runs if r["generated_at"] is None]
    runs_with_dates.sort(key=lambda r: (r["generated_at"], r["run_id"]), reverse=True)
    runs_without_dates.sort(key=lambda r: r["run_id"])
    return runs_with_dates + runs_without_dates


def _load_result_for_run_local(experiment_id: str, run_id: str) -> ResultSummary:
    exp_dir = _get_experiment_dir(experiment_id)
    runs_dir = exp_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Experiment {experiment_id} has no runs")
    run_dir = runs_dir / run_id
    result_file = run_dir / "result.json"
    if not result_file.exists():
        raise FileNotFoundError(
            f"Run {run_id} not found for experiment {experiment_id}"
        )
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
        return ResultSummary.model_validate(data)
    except Exception as e:
        raise ValueError(f"Failed to load result for run {run_id}: {e}") from e


def _save_bundle_local(
    experiment_id: str, run_id: str, bundle_json: str, owner_id: str
) -> None:
    exp_dir = _get_experiment_dir(experiment_id)
    runs_dir = exp_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "bundle.json").write_text(bundle_json, encoding="utf-8")


def _load_bundle_local(
    experiment_id: str, run_id: str, owner_id: str
) -> dict | None:
    exp_dir = _get_experiment_dir(experiment_id)
    run_dir = exp_dir / "runs" / run_id
    bundle_file = run_dir / "bundle.json"
    if not bundle_file.exists():
        return None
    return json.loads(bundle_file.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Hosted (Supabase) implementation
# ---------------------------------------------------------------------------


def _save_experiment_hosted(
    experiment_id: str,
    create_req: ExperimentCreateRequest,
    summary: ExperimentSummary,
    owner_id: str,
) -> None:
    from . import supabase_db

    config = {
        "create_request": create_req.model_dump(),
        "summary": summary.model_dump(mode="json"),
    }

    existing = supabase_db.get_experiment(experiment_id, owner_id)
    if existing:
        supabase_db.update_experiment(
            experiment_id,
            owner_id,
            {
                "name": summary.name,
                "config": config,
            },
        )
    else:
        from supabase import create_client
        from .config import get_settings

        settings = get_settings()
        client = create_client(
            settings.supabase_url, settings.supabase_service_role_key
        )
        row = {
            "id": experiment_id,
            "owner_id": owner_id,
            "name": create_req.name,
            "created_at": summary.created_at.isoformat()
            if summary.created_at
            else datetime.now(timezone.utc).isoformat(),
            "config": config,
        }
        client.table("experiments").insert(row).execute()


def _load_experiment_hosted(
    experiment_id: str, owner_id: str
) -> tuple[ExperimentCreateRequest, ExperimentSummary]:
    from . import supabase_db

    row = supabase_db.get_experiment(experiment_id, owner_id)
    if not row:
        raise FileNotFoundError(f"Experiment {experiment_id} not found")

    config = row.get("config", {})
    create_req = ExperimentCreateRequest.model_validate(
        config.get("create_request", {})
    )
    summary = ExperimentSummary.model_validate(config.get("summary", {}))
    return create_req, summary


def _list_experiments_hosted(owner_id: str) -> list[ExperimentSummary]:
    from . import supabase_db

    rows = supabase_db.list_experiments_for_owner(owner_id)
    summaries = []
    for row in rows:
        config = row.get("config", {})
        summary_data = config.get("summary")
        if summary_data:
            try:
                summaries.append(ExperimentSummary.model_validate(summary_data))
            except Exception:
                continue
    return summaries


def _save_result_hosted(
    experiment_id: str,
    run_id: str,
    result: ResultSummary,
    owner_id: str,
) -> None:
    from . import supabase_db
    from .storage_backend import get_storage_backend, storage_key

    backend = get_storage_backend()
    result_json = result.model_dump_json(indent=2)
    results_key = storage_key(owner_id, experiment_id, run_id, "results.json")
    backend.upload(
        results_key, result_json.encode("utf-8"), content_type="application/json"
    )

    # Extract analysis artifacts key if present
    analysis_key = None
    if result.analysis_artifacts:
        analysis_key = storage_key(owner_id, experiment_id, run_id, "analysis.json")
        analysis_json = json.dumps(result.analysis_artifacts, indent=2)
        backend.upload(
            analysis_key, analysis_json.encode("utf-8"), content_type="application/json"
        )

    supabase_db.create_run(
        run_id=run_id,
        experiment_id=experiment_id,
        owner_id=owner_id,
        status="succeeded",
        results_key=results_key,
        analysis_key=analysis_key,
    )


def _load_latest_result_hosted(
    experiment_id: str, owner_id: str
) -> Optional[ResultSummary]:
    from . import supabase_db
    from .storage_backend import get_storage_backend

    runs = supabase_db.list_runs_for_experiment(experiment_id, owner_id)
    if not runs:
        return None

    backend = get_storage_backend()
    for run_row in runs:
        results_key = run_row.get("results_key")
        if results_key:
            try:
                data = json.loads(backend.download(results_key).decode("utf-8"))
                return ResultSummary.model_validate(data)
            except Exception:
                continue
    return None


def _list_runs_hosted(experiment_id: str, owner_id: str) -> list[dict]:
    from . import supabase_db

    runs = supabase_db.list_runs_for_experiment(experiment_id, owner_id)
    return [
        {
            "run_id": row["id"],
            "generated_at": row.get("created_at"),
        }
        for row in runs
    ]


def _load_result_for_run_hosted(
    experiment_id: str, run_id: str, owner_id: str
) -> ResultSummary:
    from . import supabase_db
    from .storage_backend import get_storage_backend

    run_row = supabase_db.get_run(run_id, owner_id)
    if not run_row:
        raise FileNotFoundError(f"Run {run_id} not found")

    backend = get_storage_backend()
    results_key = run_row.get("results_key")
    if not results_key:
        raise FileNotFoundError(f"Run {run_id} has no results_key")

    try:
        data = json.loads(backend.download(results_key).decode("utf-8"))
        return ResultSummary.model_validate(data)
    except Exception as e:
        raise ValueError(f"Failed to load result for run {run_id}: {e}") from e


def _save_bundle_hosted(
    experiment_id: str, run_id: str, bundle_json: str, owner_id: str
) -> None:
    from .storage_backend import get_storage_backend, storage_key

    backend = get_storage_backend()
    key = storage_key(owner_id, experiment_id, run_id, "bundle.json")
    backend.upload(key, bundle_json.encode("utf-8"), content_type="application/json")


def _load_bundle_hosted(
    experiment_id: str, run_id: str, owner_id: str
) -> dict | None:
    from .storage_backend import get_storage_backend, storage_key

    backend = get_storage_backend()
    key = storage_key(owner_id, experiment_id, run_id, "bundle.json")
    if not backend.exists(key):
        return None
    return json.loads(backend.download(key).decode("utf-8"))


# ---------------------------------------------------------------------------
# Public dispatch functions
# ---------------------------------------------------------------------------


def save_experiment(
    experiment_id: str,
    create_req: ExperimentCreateRequest,
    summary: ExperimentSummary,
    *,
    owner_id: str = "anonymous",
) -> None:
    """Save experiment metadata."""
    if _is_hosted():
        _save_experiment_hosted(experiment_id, create_req, summary, owner_id)
    else:
        _save_experiment_local(experiment_id, create_req, summary)


def load_experiment(
    experiment_id: str,
    *,
    owner_id: str = "anonymous",
) -> tuple[ExperimentCreateRequest, ExperimentSummary]:
    """Load experiment metadata."""
    if _is_hosted():
        return _load_experiment_hosted(experiment_id, owner_id)
    return _load_experiment_local(experiment_id)


def list_experiments(*, owner_id: str = "anonymous") -> list[ExperimentSummary]:
    """List all experiments (scoped to owner in hosted mode)."""
    if _is_hosted():
        return _list_experiments_hosted(owner_id)
    return _list_experiments_local()


def save_result(
    experiment_id: str,
    run_id: str,
    result: ResultSummary,
    *,
    owner_id: str = "anonymous",
) -> None:
    """Save run result."""
    if _is_hosted():
        _save_result_hosted(experiment_id, run_id, result, owner_id)
    else:
        _save_result_local(experiment_id, run_id, result)


def load_latest_result(
    experiment_id: str,
    *,
    owner_id: str = "anonymous",
) -> Optional[ResultSummary]:
    """Load the latest result for an experiment."""
    if _is_hosted():
        return _load_latest_result_hosted(experiment_id, owner_id)
    return _load_latest_result_local(experiment_id)


def list_runs(
    experiment_id: str,
    *,
    owner_id: str = "anonymous",
) -> list[dict]:
    """List all runs for an experiment."""
    if _is_hosted():
        return _list_runs_hosted(experiment_id, owner_id)
    return _list_runs_local(experiment_id)


def load_result_for_run(
    experiment_id: str,
    run_id: str,
    *,
    owner_id: str = "anonymous",
) -> ResultSummary:
    """Load result for a specific run."""
    if _is_hosted():
        return _load_result_for_run_hosted(experiment_id, run_id, owner_id)
    return _load_result_for_run_local(experiment_id, run_id)


def save_bundle(
    experiment_id: str,
    run_id: str,
    bundle_json: str,
    *,
    owner_id: str = "anonymous",
) -> None:
    """Save full ResultBundle JSON for comparison (Phase 7)."""
    if _is_hosted():
        _save_bundle_hosted(experiment_id, run_id, bundle_json, owner_id)
    else:
        _save_bundle_local(experiment_id, run_id, bundle_json, owner_id)


def load_bundle(
    experiment_id: str,
    run_id: str,
    *,
    owner_id: str = "anonymous",
) -> dict | None:
    """Load full ResultBundle as dict for comparison. Returns None if not found."""
    if _is_hosted():
        return _load_bundle_hosted(experiment_id, run_id, owner_id)
    return _load_bundle_local(experiment_id, run_id, owner_id)


# ---------------------------------------------------------------------------
# R1 Share token (public share links)
# ---------------------------------------------------------------------------


def _save_share_token_local(
    experiment_id: str, run_id: str, share_token: str, owner_id: str
) -> None:
    exp_dir = _get_experiment_dir(experiment_id)
    runs_dir = exp_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "share_token": share_token,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "owner_id": owner_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "share.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def _save_share_token_hosted(
    experiment_id: str, run_id: str, share_token: str, owner_id: str
) -> None:
    from .storage_backend import get_storage_backend, storage_key

    backend = get_storage_backend()
    data = {
        "share_token": share_token,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "owner_id": owner_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    key = storage_key(owner_id, experiment_id, run_id, "share.json")
    backend.upload(
        key, json.dumps(data, indent=2).encode("utf-8"), content_type="application/json"
    )


def save_share_token(
    experiment_id: str,
    run_id: str,
    share_token: str,
    *,
    owner_id: str = "anonymous",
) -> None:
    """Save share metadata (token) for a run. Only owner should call this."""
    if _is_hosted():
        _save_share_token_hosted(experiment_id, run_id, share_token, owner_id)
    else:
        _save_share_token_local(experiment_id, run_id, share_token, owner_id)


def _validate_share_token_local(
    experiment_id: str, run_id: str, token: str
) -> str | None:
    exp_dir = _get_experiment_dir(experiment_id)
    share_file = exp_dir / "runs" / run_id / "share.json"
    if not share_file.exists():
        return None
    try:
        data = json.loads(share_file.read_text(encoding="utf-8"))
        if data.get("share_token") == token and data.get("run_id") == run_id:
            return data.get("owner_id")
    except Exception:
        pass
    return None


def _validate_share_token_hosted(
    experiment_id: str, run_id: str, token: str
) -> str | None:
    from . import supabase_db
    from .storage_backend import get_storage_backend, storage_key

    run_row = supabase_db.get_run_by_id(run_id)
    if not run_row:
        return None
    owner_id = run_row.get("owner_id")
    if not owner_id or run_row.get("experiment_id") != experiment_id:
        return None
    backend = get_storage_backend()
    key = storage_key(owner_id, experiment_id, run_id, "share.json")
    if not backend.exists(key):
        return None
    try:
        raw = backend.download(key)
        data = json.loads(raw.decode("utf-8"))
        if data.get("share_token") == token and data.get("run_id") == run_id:
            return owner_id
    except Exception:
        pass
    return None


def validate_share_token(
    experiment_id: str, run_id: str, token: str
) -> str | None:
    """Validate share token; return owner_id if valid, None otherwise."""
    if _is_hosted():
        return _validate_share_token_hosted(experiment_id, run_id, token)
    return _validate_share_token_local(experiment_id, run_id, token)


def load_result_for_shared_run(
    experiment_id: str, run_id: str, owner_id: str
) -> ResultSummary:
    """Load result for a run using owner_id from share metadata (no auth check)."""
    return load_result_for_run(experiment_id, run_id, owner_id=owner_id)
