from __future__ import annotations

import pytest


def test_rouge_l_metric():
    pytest.importorskip("evaluate")
    from metrics_lie.metrics.nlp import metric_rouge_l

    y_true = ["the cat sat on the mat", "hello world"]
    y_pred = ["the cat sat on a mat", "hello there world"]
    result = metric_rouge_l(y_true, y_pred)
    assert 0.0 <= result <= 1.0


def test_bleu_metric():
    pytest.importorskip("evaluate")
    from metrics_lie.metrics.nlp import metric_bleu

    y_true = ["the cat sat on the mat"]
    y_pred = ["the cat sat on a mat"]
    result = metric_bleu(y_true, y_pred)
    assert 0.0 <= result <= 1.0


def test_nlp_metrics_registered():
    from metrics_lie.metrics.core import METRICS

    try:
        import evaluate  # noqa: F401
        assert "rouge_l" in METRICS
        assert "bleu" in METRICS
    except ImportError:
        pass
