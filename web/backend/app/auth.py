"""Authentication middleware for Spectra backend.

When Clerk is configured (CLERK_ISSUER_URL set):
  - Verifies Clerk RS256 JWTs via JWKS endpoint
  - Extracts user ID from the 'sub' claim

When Clerk is NOT configured (local dev):
  - Returns 'anonymous' for all requests
  - No auth enforcement

This preserves Phase 4 local-dev behavior while enabling auth in hosted mode.
"""
from __future__ import annotations

from fastapi import Header, HTTPException, status

from .config import get_settings

# Lazy-loaded JWKS client
_jwks_client = None

ANONYMOUS_USER_ID = "anonymous"


def _get_jwks_client():
    """Get or create the JWKS client for Clerk JWT verification."""
    global _jwks_client
    if _jwks_client is not None:
        return _jwks_client

    try:
        import jwt as pyjwt  # noqa: F811
    except ImportError:
        raise ImportError(
            "PyJWT[crypto] required for Clerk auth. "
            "Install with: pip install 'PyJWT[crypto]'"
        )

    settings = get_settings()
    jwks_url = settings.clerk_jwks_url
    if not jwks_url:
        raise RuntimeError(
            "CLERK_ISSUER_URL (or CLERK_JWKS_URL) required when auth is enabled"
        )

    _jwks_client = pyjwt.PyJWKClient(jwks_url)
    return _jwks_client


def verify_clerk_token(token: str) -> str:
    """Verify a Clerk JWT and return the user ID (sub claim).

    Returns:
        The user_id (Clerk 'sub' claim).

    Raises:
        HTTPException: If token is invalid or expired.
    """
    import jwt as pyjwt

    try:
        jwks_client = _get_jwks_client()
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        payload = pyjwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing 'sub' claim",
            )
        return user_id
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        )
    except pyjwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
        )


async def get_current_user(authorization: str | None = Header(None)) -> str:
    """FastAPI dependency: extract and verify user identity.

    In hosted mode (Clerk configured): verifies the Bearer token and returns user_id.
    In local mode (no Clerk): returns 'anonymous'.
    """
    settings = get_settings()

    if not settings.auth_enabled:
        return ANONYMOUS_USER_ID

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
        )

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must be 'Bearer <token>'",
        )

    token = parts[1]
    return verify_clerk_token(token)
