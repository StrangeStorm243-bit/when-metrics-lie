"""Tests for ResultBundle schema extension."""
from __future__ import annotations

from metrics_lie.schema import ResultBundle


def test_result_bundle_has_task_type():
    bundle = ResultBundle(
        run_id="test",
        experiment_name="test",
        metric_name="auc",
    )
    assert bundle.task_type == "binary_classification"


def test_result_bundle_accepts_regression():
    bundle = ResultBundle(
        run_id="test",
        experiment_name="test",
        metric_name="mae",
        task_type="regression",
    )
    assert bundle.task_type == "regression"


def test_result_bundle_schema_version():
    bundle = ResultBundle(
        run_id="test",
        experiment_name="test",
        metric_name="auc",
    )
    assert bundle.schema_version == "0.2"
