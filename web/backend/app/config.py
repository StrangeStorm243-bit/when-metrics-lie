"""Configuration for Spectra web backend.

All configuration is read from environment variables.
When no cloud env vars are set, the system defaults to local filesystem behavior,
preserving Phase 4 behavior exactly.
"""
from __future__ import annotations

import os
from functools import lru_cache


class Settings:
    """Application settings derived from environment variables."""

    @property
    def storage_backend(self) -> str:
        """'local' or 'supabase'. Default: 'local'."""
        return os.getenv("SPECTRA_STORAGE_BACKEND", "local")

    @property
    def supabase_url(self) -> str | None:
        return os.getenv("SUPABASE_URL")

    @property
    def supabase_service_role_key(self) -> str | None:
        return os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    @property
    def clerk_issuer_url(self) -> str | None:
        """Clerk frontend API / issuer URL, e.g. https://my-app.clerk.accounts.dev"""
        return os.getenv("CLERK_ISSUER_URL")

    @property
    def clerk_jwks_url(self) -> str | None:
        """JWKS endpoint for Clerk JWT verification."""
        issuer = self.clerk_issuer_url
        if issuer:
            return f"{issuer.rstrip('/')}/.well-known/jwks.json"
        return os.getenv("CLERK_JWKS_URL")

    @property
    def is_hosted(self) -> bool:
        """True when running in hosted mode (Supabase backend)."""
        return self.storage_backend == "supabase"

    @property
    def auth_enabled(self) -> bool:
        """True when Clerk auth is configured."""
        return bool(self.clerk_issuer_url or os.getenv("CLERK_JWKS_URL"))

    @property
    def cors_origins(self) -> list[str]:
        """Allowed CORS origins. Always includes localhost for dev."""
        origins = [
            "http://localhost:5173",
            "http://localhost:3000",
        ]
        extra = os.getenv("SPECTRA_CORS_ORIGINS", "")
        if extra:
            origins.extend(o.strip() for o in extra.split(",") if o.strip())
        return origins

    def validate_startup(self) -> None:
        """Fail fast with clear messages for invalid hosted/auth configuration."""
        missing: list[str] = []
        if self.storage_backend == "supabase":
            if not self.supabase_url:
                missing.append("SUPABASE_URL")
            if not self.supabase_service_role_key:
                missing.append("SUPABASE_SERVICE_ROLE_KEY")

        if missing:
            raise RuntimeError(
                "Hosted mode configuration error: missing required environment variables: "
                + ", ".join(missing)
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
