"""Tests for Fairlearn-powered fairness analysis."""
from __future__ import annotations
import numpy as np
import pytest

fairlearn = pytest.importorskip("fairlearn")

from metrics_lie.diagnostics.fairness import compute_fairness_report  # noqa: E402

def test_compute_fairness_report_binary():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 0, 0, 1])
    sensitive = np.array(["A","A","A","A","A","B","B","B","B","B"])
    report = compute_fairness_report(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive,
        metric_fns={"accuracy": lambda yt, yp: float(np.mean(yt == yp))},
    )
    assert "group_metrics" in report
    assert "A" in report["group_metrics"]
    assert "B" in report["group_metrics"]
    assert "gaps" in report
    assert "accuracy" in report["gaps"]
    assert "demographic_parity_difference" in report

def test_compute_fairness_report_multiclass():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 1, 1, 0])
    sensitive = np.array(["X","X","X","Y","Y","Y"])
    report = compute_fairness_report(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive,
        metric_fns={"accuracy": lambda yt, yp: float(np.mean(yt == yp))},
    )
    assert "group_metrics" in report
    assert "gaps" in report
