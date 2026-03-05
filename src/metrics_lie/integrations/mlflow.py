"""MLflow integration — log Spectra results to MLflow tracking."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def log_to_mlflow(
    result: Any,
    *,
    run_id: str | None = None,
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
) -> str:
    """Log a Spectra ResultBundle to MLflow.

    Logs metrics, scenario results (as JSON artifact), experiment params,
    and any matplotlib plot artifacts.

    Args:
        result: A ResultBundle instance.
        run_id: Existing MLflow run ID to log to. If None, creates a new run.
        experiment_name: MLflow experiment name (used only when creating a new run).
        tracking_uri: MLflow tracking server URI. If None, uses MLflow default.

    Returns:
        The MLflow run ID that was logged to.
    """
    try:
        import mlflow
    except ImportError as e:
        raise ImportError(
            "MLflow integration requires: pip install metrics_lie[mlflow]"
        ) from e

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if experiment_name and run_id is None:
        mlflow.set_experiment(experiment_name)

    def _do_log(active_run: Any) -> str:
        rid = active_run.info.run_id

        # Log params
        mlflow.log_param("spectra.experiment_name", result.experiment_name)
        mlflow.log_param("spectra.metric_name", result.metric_name)
        mlflow.log_param("spectra.task_type", getattr(result, "task_type", "binary_classification"))
        mlflow.log_param("spectra.run_id", result.run_id)

        # Log baseline metric
        if result.baseline:
            mlflow.log_metric(f"spectra.{result.metric_name}.baseline_mean", result.baseline.mean)
            mlflow.log_metric(f"spectra.{result.metric_name}.baseline_std", result.baseline.std)
            mlflow.log_metric(f"spectra.{result.metric_name}.baseline_q05", result.baseline.q05)
            mlflow.log_metric(f"spectra.{result.metric_name}.baseline_q95", result.baseline.q95)

        # Log per-metric results
        for metric_name, summary in result.metric_results.items():
            mlflow.log_metric(f"spectra.{metric_name}.mean", summary.mean)
            mlflow.log_metric(f"spectra.{metric_name}.std", summary.std)

        # Log scenario deltas as metrics
        for scenario in result.scenarios:
            delta = scenario.metric.mean - (result.baseline.mean if result.baseline else 0.0)
            safe_id = scenario.scenario_id.replace(" ", "_")
            mlflow.log_metric(f"spectra.scenario.{safe_id}.mean", scenario.metric.mean)
            mlflow.log_metric(f"spectra.scenario.{safe_id}.delta", delta)

        # Log full ResultBundle as JSON artifact
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="spectra_result_"
        ) as f:
            f.write(result.to_pretty_json())
            temp_path = f.name
        try:
            mlflow.log_artifact(temp_path, artifact_path="spectra")
        finally:
            Path(temp_path).unlink(missing_ok=True)

        logger.info("Logged Spectra result to MLflow run %s", rid)
        return rid

    if run_id:
        with mlflow.start_run(run_id=run_id):
            return _do_log(mlflow.active_run())
    else:
        with mlflow.start_run() as active_run:
            return _do_log(active_run)
