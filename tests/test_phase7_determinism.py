"""Phase 7: Determinism tests for upload -> run -> compare and multi-feature validation."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from metrics_lie.db.session import DB_PATH, engine, init_db

# Import web backend components for Phase 7 path
pytest.importorskip("fastapi")
from web.backend.app.contracts import ExperimentCreateRequest
from web.backend.app.engine_bridge import run_experiment
from web.backend.app.model_validation import validate_sklearn_pickle
from web.backend.app.persistence import load_bundle


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _fresh_registry_db() -> None:
    """Ensure local SQLite schema exists for engine runs."""
    engine.dispose()
    if DB_PATH.exists():
        DB_PATH.unlink()
    init_db()


def test_web_bridge_produces_valid_bundle(tmp_path: Path) -> None:
    """Web bridge run_experiment produces a valid, loadable bundle.

    Engine-level determinism (same seed = same result) is covered by
    test35_mvs_integration::test_mvs_determinism_golden, so this test
    only verifies that the web bridge path works end-to-end.
    """
    df = pd.DataFrame(
        {
            "feature1": [0, 1, 2, 3, 4, 5],
            "y_true": [0, 0, 0, 1, 1, 1],
            "y_score": [0.1, 0.2, 0.3, 0.8, 0.9, 0.7],
            "group": ["A", "A", "B", "B", "A", "B"],
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    X = df[["feature1"]].to_numpy(dtype=float)
    y = df["y_true"].to_numpy(dtype=int)
    model = LogisticRegression(random_state=0).fit(X, y)
    raw = pickle.dumps(model)
    model_id = hashlib.sha256(raw).hexdigest()

    result = validate_sklearn_pickle(raw)
    assert result.valid, result.error

    owner_id = "test_user_phase7"
    repo = _repo_root()
    models_dir = repo / ".spectra_ui" / "models" / owner_id
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"{model_id}.pkl").write_bytes(raw)
    meta = {
        "model_id": model_id,
        "original_filename": "model.pkl",
        "model_class": result.model_class,
        "capabilities": result.capabilities,
        "file_size_bytes": len(raw),
        "uploaded_at": "2025-01-01T00:00:00Z",
        "owner_id": owner_id,
    }
    (models_dir / f"{model_id}.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    create_req = ExperimentCreateRequest(
        name="phase7_web_bridge",
        metric_id="accuracy",
        stress_suite_id="default",
        config={
            "dataset_path": str(csv_path.resolve()),
            "model_id": model_id,
            "feature_cols": ["feature1"],
            "threshold": 0.5,
        },
    )

    summary = run_experiment(
        create_req,
        experiment_id="exp_phase7_bridge",
        run_id="run_bridge",
        seed=42,
        owner_id=owner_id,
    )

    assert summary is not None

    bundle = load_bundle("exp_phase7_bridge", "run_bridge", owner_id=owner_id)
    assert bundle is not None
    assert "baseline" in bundle


def test_multifeature_model_upload_validation() -> None:
    """Model with 5 features passes validation (uses n_features_in_ for probe shape)."""
    df = pd.DataFrame(
        {
            "f1": [0] * 4 + [1] * 4,
            "f2": [0, 0, 1, 1] * 2,
            "f3": [0, 1, 0, 1] * 2,
            "f4": [1, 0, 1, 0] * 2,
            "f5": [0.5] * 8,
            "y_true": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    X = df[["f1", "f2", "f3", "f4", "f5"]].to_numpy(dtype=float)
    y = df["y_true"].to_numpy(dtype=int)
    model = LogisticRegression(random_state=0).fit(X, y)
    raw = pickle.dumps(model)

    result = validate_sklearn_pickle(raw)
    assert result.valid, result.error or "expected valid"
    assert result.model_class
