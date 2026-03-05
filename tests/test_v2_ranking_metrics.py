from __future__ import annotations

import numpy as np
import pytest


def test_metric_ndcg_at_k():
    from metrics_lie.metrics.core import metric_ndcg

    y_true = np.array([3, 2, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    result = metric_ndcg(y_true, y_score)
    assert 0.0 <= result <= 1.0


def test_metric_ndcg_perfect_ranking():
    from metrics_lie.metrics.core import metric_ndcg

    y_true = np.array([3, 2, 1, 0])
    y_score = np.array([3.0, 2.0, 1.0, 0.0])
    result = metric_ndcg(y_true, y_score)
    assert result == pytest.approx(1.0, abs=1e-6)


def test_ndcg_in_metrics_dict():
    from metrics_lie.metrics.core import METRICS

    assert "ndcg" in METRICS


def test_ndcg_in_ranking_set():
    from metrics_lie.metrics.core import RANKING_METRICS

    assert "ndcg" in RANKING_METRICS


def test_ranking_metrics_in_registry():
    from metrics_lie.metrics.registry import METRIC_REQUIREMENTS

    ranking_ids = {
        r.metric_id
        for r in METRIC_REQUIREMENTS
        if r.task_types and "ranking" in r.task_types
    }
    assert "ndcg" in ranking_ids
