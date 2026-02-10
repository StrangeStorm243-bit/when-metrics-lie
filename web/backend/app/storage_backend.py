"""Storage backend abstraction for Spectra artifact files.

Provides a Protocol-based interface with two implementations:
- LocalFSStorage (default): stores artifacts on local filesystem
- SupabaseStorage: stores artifacts in Supabase Storage buckets

The storage backend handles results.json, analysis.json, and plot files.
Experiment/run metadata is handled by the persistence layer.

Storage path convention (when hosted):
  artifacts/{owner_id}/experiments/{experiment_id}/runs/{run_id}/results.json
  artifacts/{owner_id}/experiments/{experiment_id}/runs/{run_id}/analysis.json
  artifacts/{owner_id}/experiments/{experiment_id}/runs/{run_id}/plots/*
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for artifact storage operations."""

    def upload(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Upload data to storage. Returns the storage key."""
        ...

    def download(self, key: str) -> bytes:
        """Download data from storage by key."""
        ...

    def exists(self, key: str) -> bool:
        """Check if a key exists in storage."""
        ...

    def list_keys(self, prefix: str) -> list[str]:
        """List all keys under a given prefix."""
        ...

    def delete(self, key: str) -> None:
        """Delete a key from storage."""
        ...


class LocalFSStorage:
    """Local filesystem storage backend.

    Default for development. Stores artifacts under .spectra_ui/artifacts/.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        if base_dir is None:
            repo_root = Path(__file__).parent.parent.parent.parent
            base_dir = repo_root / ".spectra_ui" / "artifacts"
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        return self._base_dir / Path(key)

    def upload(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def download(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.exists():
            raise FileNotFoundError(f"Storage key not found: {key}")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def list_keys(self, prefix: str) -> list[str]:
        base = self._resolve(prefix)
        if not base.exists():
            return []
        keys = []
        for p in base.rglob("*"):
            if p.is_file():
                rel = p.relative_to(self._base_dir)
                keys.append(rel.as_posix())
        return sorted(keys)

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        if path.exists():
            path.unlink()


class SupabaseStorage:
    """Supabase Storage backend for hosted deployments."""

    def __init__(self, url: str, service_role_key: str, bucket: str = "artifacts") -> None:
        try:
            from supabase import create_client
        except ImportError:
            raise ImportError(
                "supabase package required for SupabaseStorage. "
                "Install with: pip install supabase"
            )
        self._client = create_client(url, service_role_key)
        self._bucket = bucket

    def upload(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self._client.storage.from_(self._bucket).upload(
            path=key,
            file=data,
            file_options={"content-type": content_type, "upsert": "true"},
        )
        return key

    def download(self, key: str) -> bytes:
        return self._client.storage.from_(self._bucket).download(key)

    def exists(self, key: str) -> bool:
        try:
            self.download(key)
            return True
        except Exception:
            return False

    def list_keys(self, prefix: str) -> list[str]:
        parts = prefix.rstrip("/").rsplit("/", 1)
        folder = parts[0] if len(parts) > 1 else ""
        result = self._client.storage.from_(self._bucket).list(folder)
        keys = []
        for item in result:
            name = item.get("name", "")
            if name:
                full_key = f"{folder}/{name}" if folder else name
                keys.append(full_key)
        return sorted(keys)

    def delete(self, key: str) -> None:
        self._client.storage.from_(self._bucket).remove([key])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_backend: StorageBackend | None = None


def get_storage_backend() -> StorageBackend:
    """Get the configured storage backend (singleton)."""
    global _backend
    if _backend is not None:
        return _backend

    from .config import get_settings

    settings = get_settings()

    if settings.storage_backend == "supabase":
        url = settings.supabase_url
        key = settings.supabase_service_role_key
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required "
                "when SPECTRA_STORAGE_BACKEND=supabase"
            )
        _backend = SupabaseStorage(url, key)
    else:
        _backend = LocalFSStorage()

    return _backend


def storage_key(owner_id: str, experiment_id: str, run_id: str, filename: str) -> str:
    """Build a deterministic storage key following the required convention."""
    return f"artifacts/{owner_id}/experiments/{experiment_id}/runs/{run_id}/{filename}"
