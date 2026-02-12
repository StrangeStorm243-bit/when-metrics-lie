"""Supabase Postgres client for experiment/run metadata.

Used when SPECTRA_STORAGE_BACKEND=supabase. Communicates with Supabase Postgres
via the PostgREST API using the service role key.

All queries explicitly filter by owner_id to enforce user scoping at the
application layer (in addition to RLS policies on the database).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid


def _get_client():
    """Get a Supabase client (lazy import to avoid hard dependency)."""
    try:
        from supabase import create_client
    except ImportError:
        raise ImportError(
            "supabase package required for hosted mode. "
            "Install with: pip install supabase"
        )

    from .config import get_settings

    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY required for hosted mode"
        )
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def create_experiment(owner_id: str, name: str, config: dict[str, Any]) -> dict:
    """Insert a new experiment row. Returns the inserted row as dict."""
    client = _get_client()
    experiment_id = str(uuid.uuid4())
    row = {
        "id": experiment_id,
        "owner_id": owner_id,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
    }
    result = client.table("experiments").insert(row).execute()
    return result.data[0]


def get_experiment(experiment_id: str, owner_id: str) -> dict | None:
    """Get an experiment by ID, scoped to owner."""
    client = _get_client()
    result = (
        client.table("experiments")
        .select("*")
        .eq("id", experiment_id)
        .eq("owner_id", owner_id)
        .execute()
    )
    return result.data[0] if result.data else None


def list_experiments_for_owner(owner_id: str) -> list[dict]:
    """List all experiments for an owner, newest first."""
    client = _get_client()
    result = (
        client.table("experiments")
        .select("*")
        .eq("owner_id", owner_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data


def update_experiment(experiment_id: str, owner_id: str, updates: dict) -> dict | None:
    """Update an experiment row, scoped to owner."""
    client = _get_client()
    result = (
        client.table("experiments")
        .update(updates)
        .eq("id", experiment_id)
        .eq("owner_id", owner_id)
        .execute()
    )
    return result.data[0] if result.data else None


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def create_run(
    run_id: str,
    experiment_id: str,
    owner_id: str,
    status: str,
    results_key: str,
    analysis_key: str | None = None,
) -> dict:
    """Insert a new run row. Returns the inserted row as dict."""
    client = _get_client()
    row = {
        "id": run_id,
        "experiment_id": experiment_id,
        "owner_id": owner_id,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results_key": results_key,
        "analysis_key": analysis_key,
    }
    result = client.table("runs").insert(row).execute()
    return result.data[0]


def list_runs_for_experiment(experiment_id: str, owner_id: str) -> list[dict]:
    """List all runs for an experiment, scoped to owner, newest first."""
    client = _get_client()
    result = (
        client.table("runs")
        .select("*")
        .eq("experiment_id", experiment_id)
        .eq("owner_id", owner_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data


def get_run(run_id: str, owner_id: str) -> dict | None:
    """Get a run by ID, scoped to owner."""
    client = _get_client()
    result = (
        client.table("runs")
        .select("*")
        .eq("id", run_id)
        .eq("owner_id", owner_id)
        .execute()
    )
    return result.data[0] if result.data else None


def get_run_by_id(run_id: str) -> dict | None:
    """Get a run by ID only (service-role bypasses RLS). Used for share token validation."""
    client = _get_client()
    result = client.table("runs").select("*").eq("id", run_id).execute()
    return result.data[0] if result.data else None


def update_run(run_id: str, owner_id: str, updates: dict) -> dict | None:
    """Update a run row, scoped to owner."""
    client = _get_client()
    result = (
        client.table("runs")
        .update(updates)
        .eq("id", run_id)
        .eq("owner_id", owner_id)
        .execute()
    )
    return result.data[0] if result.data else None
