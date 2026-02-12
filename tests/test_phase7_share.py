"""Phase 7 R1: Share token persistence and validation tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
from web.backend.app.persistence import (
    save_share_token,
    validate_share_token,
)
from web.backend.app.routers.share import _generate_share_token


def test_share_token_roundtrip_local(tmp_path: Path) -> None:
    """save_share_token -> validate_share_token returns correct owner_id."""
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    with patch("web.backend.app.persistence._get_experiments_dir", return_value=experiments_dir):
        save_share_token("exp1", "run1", "secret-token-xyz", owner_id="alice")
        owner_id = validate_share_token("exp1", "run1", "secret-token-xyz")

    assert owner_id == "alice"
    share_file = experiments_dir / "exp1" / "runs" / "run1" / "share.json"
    assert share_file.exists()


def test_share_token_invalid_rejected(tmp_path: Path) -> None:
    """validate_share_token with wrong token returns None."""
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    with patch("web.backend.app.persistence._get_experiments_dir", return_value=experiments_dir):
        save_share_token("exp1", "run1", "correct-token", owner_id="alice")
        owner_id = validate_share_token("exp1", "run1", "wrong-token")

    assert owner_id is None


def test_share_token_wrong_run_rejected(tmp_path: Path) -> None:
    """validate_share_token with correct token but wrong run_id returns None."""
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    with patch("web.backend.app.persistence._get_experiments_dir", return_value=experiments_dir):
        save_share_token("exp1", "run1", "secret-token", owner_id="alice")
        owner_id = validate_share_token("exp1", "run2", "secret-token")

    assert owner_id is None


def test_share_token_generation_strength() -> None:
    """Generated share tokens are strong enough for demo links."""
    token_a = _generate_share_token()
    token_b = _generate_share_token()

    assert token_a != token_b
    assert len(token_a) >= 43
    assert len(token_b) >= 43
