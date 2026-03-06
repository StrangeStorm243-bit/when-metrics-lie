"""Tests for POST /auto-detect endpoint."""

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


def test_auto_detect_without_model(tmp_path, monkeypatch):
    """Upload CSV with y_true/y_score, call auto-detect with only dataset_id."""
    monkeypatch.setenv("SPECTRA_STORAGE_BACKEND", "local")
    csv_data = "y_true,y_score,age,income\n1,0.9,25,50000\n0,0.1,30,60000\n"

    # Upload dataset first
    upload_resp = client.post(
        "/datasets",
        files={"file": ("detect_test.csv", _csv_bytes(csv_data), "text/csv")},
    )
    assert upload_resp.status_code == 201
    dataset_id = upload_resp.json()["dataset_id"]

    # Call auto-detect
    detect_resp = client.post(
        "/auto-detect",
        json={"dataset_id": dataset_id},
    )
    assert detect_resp.status_code == 200
    body = detect_resp.json()

    assert body["task_type"] == "binary_classification"
    assert body["y_true_col"] == "y_true"
    assert body["y_score_col"] == "y_score"
    assert "age" in body["feature_cols"]
    assert "income" in body["feature_cols"]
    assert body["recommended_metric"] == "auc"
    assert body["recommended_stress_suite"] == "balanced"
    assert body["n_rows"] == 2
    assert body["model_class"] is None
    assert body["confidence"] == "medium"


def test_auto_detect_column_patterns(tmp_path, monkeypatch):
    """Upload CSV with 'label'/'prediction' columns, verify detected correctly."""
    monkeypatch.setenv("SPECTRA_STORAGE_BACKEND", "local")
    csv_data = "label,prediction,feat_a\n1,0.8,42\n0,0.2,7\n"

    # Upload dataset first
    upload_resp = client.post(
        "/datasets",
        files={"file": ("pattern_test.csv", _csv_bytes(csv_data), "text/csv")},
    )
    assert upload_resp.status_code == 201
    dataset_id = upload_resp.json()["dataset_id"]

    # Call auto-detect
    detect_resp = client.post(
        "/auto-detect",
        json={"dataset_id": dataset_id},
    )
    assert detect_resp.status_code == 200
    body = detect_resp.json()

    assert body["task_type"] == "binary_classification"
    assert body["y_true_col"] == "label"
    assert body["y_score_col"] == "prediction"
    assert body["feature_cols"] == ["feat_a"]
    assert body["recommended_metric"] == "auc"
    assert body["confidence"] == "medium"
