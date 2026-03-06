"""End-to-end integration test for Quick Test flow."""
from __future__ import annotations

import io
from pathlib import Path

import pytest


@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from web.backend.app.main import app
    return TestClient(app)


@pytest.fixture
def demo_csv_bytes():
    """CSV compatible with the demo model."""
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    for csv_file in sorted(data_dir.glob("*.csv")):
        content = csv_file.read_text()
        if "y_true" in content and "y_score" in content:
            return csv_file.read_bytes()
    pytest.skip("No suitable demo CSV found")


def test_quick_test_full_flow(client, demo_csv_bytes):
    """Full Quick Test flow: upload dataset -> auto-detect -> create+run experiment."""
    # Step 1: Upload dataset
    ds_resp = client.post(
        "/datasets",
        files={"file": ("demo.csv", io.BytesIO(demo_csv_bytes), "text/csv")},
    )
    assert ds_resp.status_code == 201
    ds_data = ds_resp.json()
    dataset_id = ds_data["dataset_id"]
    assert ds_data["detected_y_true_col"] is not None

    # Step 2: Auto-detect
    detect_resp = client.post(
        "/auto-detect",
        json={"dataset_id": dataset_id},
    )
    assert detect_resp.status_code == 200
    detect_data = detect_resp.json()
    assert detect_data["task_type"]
    assert detect_data["recommended_metric"]
    assert detect_data["recommended_stress_suite"]

    # Step 3: Create experiment with detected config
    config = {
        "task_type": detect_data["task_type"],
        "y_true_col": detect_data["y_true_col"],
        "y_score_col": detect_data["y_score_col"],
        "dataset_id": dataset_id,
    }
    exp_resp = client.post(
        "/experiments",
        json={
            "name": "Quick Test - demo.csv",
            "metric_id": detect_data["recommended_metric"],
            "stress_suite_id": detect_data["recommended_stress_suite"],
            "config": config,
        },
    )
    assert exp_resp.status_code == 201
    experiment_id = exp_resp.json()["id"]

    # Step 4: Run experiment
    run_resp = client.post(
        f"/experiments/{experiment_id}/run",
        json={},
    )
    assert run_resp.status_code == 200
    assert run_resp.json()["status"] == "completed"


def test_auto_detect_column_patterns(client):
    """Verify auto-detect handles various column naming patterns."""
    csv = b"label,prediction,age\n1,0.9,25\n0,0.1,30\n1,0.7,35\n"
    ds_resp = client.post(
        "/datasets",
        files={"file": ("data.csv", io.BytesIO(csv), "text/csv")},
    )
    assert ds_resp.status_code == 201
    ds = ds_resp.json()

    detect_resp = client.post(
        "/auto-detect",
        json={"dataset_id": ds["dataset_id"]},
    )
    assert detect_resp.status_code == 200
    data = detect_resp.json()
    assert data["y_true_col"] == "label"
    assert data["y_score_col"] == "prediction"
    assert "age" in data["feature_cols"]
