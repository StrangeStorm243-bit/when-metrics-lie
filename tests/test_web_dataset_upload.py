"""Tests for POST /datasets CSV upload endpoint."""

from __future__ import annotations

import io

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from web.backend.app.main import app  # noqa: E402

client = TestClient(app)


def _csv_bytes(text: str) -> io.BytesIO:
    """Return a BytesIO CSV from a text string."""
    return io.BytesIO(text.strip().encode("utf-8"))


def test_upload_csv_returns_columns(tmp_path, monkeypatch):
    """Upload CSV with y_true, y_score, age, income — verify detection."""
    monkeypatch.setenv("SPECTRA_STORAGE_BACKEND", "local")
    csv_data = "y_true,y_score,age,income\n1,0.9,25,50000\n0,0.1,30,60000\n"

    response = client.post(
        "/datasets",
        files={"file": ("test_data.csv", _csv_bytes(csv_data), "text/csv")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["original_filename"] == "test_data.csv"
    assert body["columns"] == ["y_true", "y_score", "age", "income"]
    assert body["n_rows"] == 2
    assert body["detected_y_true_col"] == "y_true"
    assert body["detected_y_score_col"] == "y_score"
    assert "age" in body["detected_feature_cols"]
    assert "income" in body["detected_feature_cols"]
    assert body["dataset_id"]  # non-empty SHA256


def test_upload_csv_detects_label_prediction_columns(tmp_path, monkeypatch):
    """CSV with 'label' and 'prediction' columns detected correctly."""
    monkeypatch.setenv("SPECTRA_STORAGE_BACKEND", "local")
    csv_data = "label,prediction,feat_a\n1,0.8,42\n0,0.2,7\n"

    response = client.post(
        "/datasets",
        files={"file": ("alt.csv", _csv_bytes(csv_data), "text/csv")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["detected_y_true_col"] == "label"
    assert body["detected_y_score_col"] == "prediction"
    assert body["detected_feature_cols"] == ["feat_a"]


def test_upload_rejects_non_csv(tmp_path, monkeypatch):
    """Non-CSV file rejected with 422."""
    monkeypatch.setenv("SPECTRA_STORAGE_BACKEND", "local")

    response = client.post(
        "/datasets",
        files={"file": ("data.json", io.BytesIO(b'{"a":1}'), "application/json")},
    )

    assert response.status_code == 422
    assert "csv" in response.json()["detail"].lower()


def test_engine_bridge_resolves_dataset_id():
    """Engine bridge should resolve dataset_id to the uploaded CSV path."""
    pytest.importorskip("fastapi")
    from web.backend.app.engine_bridge import _get_dataset_path
    from web.backend.app.contracts import ExperimentCreateRequest
    from web.backend.app.routers.datasets import _datasets_dir_local

    local_dir = _datasets_dir_local("anonymous")
    fake_id = "abc123"
    csv_path = local_dir / f"{fake_id}.csv"
    csv_path.write_text("y_true,y_score\n1,0.9\n0,0.1\n")

    try:
        req = ExperimentCreateRequest(
            name="test",
            metric_id="auc",
            stress_suite_id="balanced",
            config={"dataset_id": fake_id},
        )
        result = _get_dataset_path(req, owner_id="anonymous")
        assert result == csv_path.resolve()
    finally:
        csv_path.unlink(missing_ok=True)
