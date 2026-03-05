"""Tests for Phase 5 preset task-type filtering."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from web.backend.app.storage import METRIC_PRESETS


def test_all_presets_have_task_types_field():
    """Every preset must have a task_types list."""
    for p in METRIC_PRESETS:
        assert "task_types" in p, f"Preset {p['id']} missing task_types"
        assert isinstance(p["task_types"], list)
        assert len(p["task_types"]) > 0


def test_binary_presets_include_auc():
    binary = [p for p in METRIC_PRESETS if "binary_classification" in p["task_types"]]
    ids = [p["id"] for p in binary]
    assert "auc" in ids
    assert "brier_score" in ids


def test_regression_presets_include_mae():
    regression = [p for p in METRIC_PRESETS if "regression" in p["task_types"]]
    ids = [p["id"] for p in regression]
    assert "mae" in ids
    assert "mse" in ids
    assert "rmse" in ids
    assert "r_squared" in ids


def test_multiclass_presets_include_weighted_f1():
    multiclass = [p for p in METRIC_PRESETS if "multiclass_classification" in p["task_types"]]
    ids = [p["id"] for p in multiclass]
    assert "weighted_f1" in ids
    assert "macro_f1" in ids
    assert "cohens_kappa" in ids


def test_auc_not_in_regression():
    regression = [p for p in METRIC_PRESETS if "regression" in p["task_types"]]
    ids = [p["id"] for p in regression]
    assert "auc" not in ids


def test_mae_not_in_binary():
    binary = [p for p in METRIC_PRESETS if "binary_classification" in p["task_types"]]
    ids = [p["id"] for p in binary]
    assert "mae" not in ids


def test_shared_metrics_appear_in_both_classification_types():
    """accuracy, f1, precision, recall should be in both binary and multiclass."""
    shared = ["accuracy", "f1", "precision", "recall"]
    binary = {p["id"] for p in METRIC_PRESETS if "binary_classification" in p["task_types"]}
    multi = {p["id"] for p in METRIC_PRESETS if "multiclass_classification" in p["task_types"]}
    for m in shared:
        assert m in binary, f"{m} missing from binary"
        assert m in multi, f"{m} missing from multiclass"


def test_filter_simulation():
    """Simulate the API filter: only return presets matching task_type."""
    task_type = "regression"
    filtered = [p for p in METRIC_PRESETS if task_type in p.get("task_types", [])]
    assert all("regression" in p["task_types"] for p in filtered)
    assert len(filtered) >= 4  # mae, mse, rmse, r_squared at minimum
