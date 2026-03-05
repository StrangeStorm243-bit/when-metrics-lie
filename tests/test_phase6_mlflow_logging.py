"""Tests for Phase 6 MLflow logging integration."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from metrics_lie.schema import MetricSummary, ResultBundle, ScenarioResult


def _metric(mean: float = 0.85) -> MetricSummary:
    return MetricSummary(mean=mean, std=0.02, q05=0.81, q50=0.85, q95=0.89, n=200)


def _bundle(
    *,
    metric_name: str = "auc",
    baseline_mean: float = 0.90,
    task_type: str = "binary_classification",
) -> ResultBundle:
    return ResultBundle(
        run_id="TEST_RUN_01",
        experiment_name="test_exp",
        metric_name=metric_name,
        task_type=task_type,
        baseline=_metric(baseline_mean),
        scenarios=[
            ScenarioResult(scenario_id="label_noise", params={"p": 0.1}, metric=_metric(0.85)),
            ScenarioResult(scenario_id="score_noise", params={"sigma": 0.05}, metric=_metric(0.88)),
        ],
        metric_results={"auc": _metric(0.90), "f1": _metric(0.82)},
        created_at="2026-01-01T00:00:00+00:00",
    )


@patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.pyfunc": MagicMock()})
def test_log_to_mlflow_creates_run():
    """log_to_mlflow creates a new MLflow run and logs params/metrics."""
    import sys
    mock_mlflow = sys.modules["mlflow"]

    mock_run = MagicMock()
    mock_run.info.run_id = "mlflow_run_123"
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.active_run.return_value = mock_run

    from metrics_lie.integrations.mlflow import log_to_mlflow

    bundle = _bundle()
    rid = log_to_mlflow(bundle)

    assert rid == "mlflow_run_123"
    mock_mlflow.log_param.assert_any_call("spectra.experiment_name", "test_exp")
    mock_mlflow.log_param.assert_any_call("spectra.metric_name", "auc")
    mock_mlflow.log_param.assert_any_call("spectra.task_type", "binary_classification")
    mock_mlflow.log_metric.assert_any_call("spectra.auc.baseline_mean", 0.90)


@patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.pyfunc": MagicMock()})
def test_log_to_mlflow_logs_scenario_deltas():
    """Scenario deltas are logged as metrics."""
    import sys
    mock_mlflow = sys.modules["mlflow"]

    mock_run = MagicMock()
    mock_run.info.run_id = "run_456"
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.active_run.return_value = mock_run

    from metrics_lie.integrations.mlflow import log_to_mlflow

    bundle = _bundle(baseline_mean=0.90)
    log_to_mlflow(bundle)

    # label_noise: 0.85 - 0.90 = -0.05
    mock_mlflow.log_metric.assert_any_call("spectra.scenario.label_noise.delta", pytest.approx(-0.05))
    mock_mlflow.log_metric.assert_any_call("spectra.scenario.score_noise.mean", pytest.approx(0.88))


@patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.pyfunc": MagicMock()})
def test_log_to_mlflow_logs_artifact():
    """Full ResultBundle is logged as JSON artifact."""
    import sys
    mock_mlflow = sys.modules["mlflow"]

    mock_run = MagicMock()
    mock_run.info.run_id = "run_789"
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.active_run.return_value = mock_run

    from metrics_lie.integrations.mlflow import log_to_mlflow

    bundle = _bundle()
    log_to_mlflow(bundle)

    mock_mlflow.log_artifact.assert_called_once()
    args = mock_mlflow.log_artifact.call_args
    assert args.kwargs.get("artifact_path") == "spectra" or args[1].get("artifact_path") == "spectra"


@patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.pyfunc": MagicMock()})
def test_log_to_mlflow_with_tracking_uri():
    """tracking_uri is set when provided."""
    import sys
    mock_mlflow = sys.modules["mlflow"]

    mock_run = MagicMock()
    mock_run.info.run_id = "run_uri"
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.active_run.return_value = mock_run

    from metrics_lie.integrations.mlflow import log_to_mlflow

    bundle = _bundle()
    log_to_mlflow(bundle, tracking_uri="http://mlflow:5000")

    mock_mlflow.set_tracking_uri.assert_called_once_with("http://mlflow:5000")


def test_log_to_mlflow_import_error_without_mlflow():
    """Raises ImportError with install hint when mlflow is missing."""
    # This test works when mlflow is NOT installed
    try:
        import mlflow  # noqa: F401
        pytest.skip("mlflow is installed -- cannot test missing import")
    except ImportError:
        pass

    from metrics_lie.integrations.mlflow import log_to_mlflow

    bundle = _bundle()
    with pytest.raises(ImportError, match="pip install"):
        log_to_mlflow(bundle)
