"""Tests for Phase 5 engine bridge multi-task routing."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from web.backend.app.engine_bridge import _get_default_scenarios


# --- scenario routing by task type ---

def test_binary_scenarios_include_class_imbalance():
    scenarios = _get_default_scenarios("default", "binary_classification")
    ids = [s["id"] for s in scenarios]
    assert "class_imbalance" in ids
    assert "label_noise" in ids
    assert "score_noise" in ids


def test_regression_scenarios_exclude_class_imbalance():
    scenarios = _get_default_scenarios("default", "regression")
    ids = [s["id"] for s in scenarios]
    assert "class_imbalance" not in ids
    assert "label_noise" in ids
    assert "score_noise" in ids


def test_ranking_scenarios_exclude_class_imbalance():
    scenarios = _get_default_scenarios("default", "ranking")
    ids = [s["id"] for s in scenarios]
    assert "class_imbalance" not in ids


def test_multiclass_scenarios_include_class_imbalance():
    scenarios = _get_default_scenarios("default", "multiclass_classification")
    ids = [s["id"] for s in scenarios]
    assert "class_imbalance" in ids


def test_default_task_type_is_binary():
    """When no task_type given, defaults to binary (has class_imbalance)."""
    scenarios = _get_default_scenarios("default")
    ids = [s["id"] for s in scenarios]
    assert "class_imbalance" in ids
