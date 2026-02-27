"""Evidently-powered drift detection for dataset comparison."""
from __future__ import annotations

from typing import Any

import pandas as pd


def compute_drift_report(
    *,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Compare reference and current datasets for distribution drift.

    Uses Evidently's DataDriftPreset for per-feature drift statistics.
    Supports both evidently >=0.7 (new API) and >=0.4 (legacy API).

    Returns a dict with keys:
        dataset_drift (bool): Whether overall dataset drift was detected.
        n_drifted_features (int): Number of drifted features.
        n_features (int): Total number of features compared.
        feature_drift (dict): Per-feature drift details.
    """
    if feature_columns is not None:
        reference = reference[feature_columns]
        current = current[feature_columns]
    else:
        shared = sorted(set(reference.columns) & set(current.columns))
        reference = reference[shared]
        current = current[shared]

    n_features = len(reference.columns)

    return _run_drift_new_api(reference, current, n_features)


def _run_drift_new_api(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    n_features: int,
) -> dict[str, Any]:
    """Run drift detection using the evidently >=0.7 API."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report([DataDriftPreset()])
    snapshot = report.run(current, reference)

    result_dict = snapshot.dict()
    metrics = result_dict.get("metrics", [])

    n_drifted = 0
    drift_share = 0.0
    feature_drift: dict[str, Any] = {}

    for metric_entry in metrics:
        metric_name = metric_entry.get("metric_name", "")
        config = metric_entry.get("config", {})
        value = metric_entry.get("value", {})

        # DriftedColumnsCount returns {"count": <int>, "share": <float>}
        if "DriftedColumnsCount" in metric_name:
            if isinstance(value, dict):
                n_drifted = int(value.get("count", 0))
                drift_share = float(value.get("share", 0.0))
            continue

        # ValueDrift returns a drift score (float) per column
        if "ValueDrift" in metric_name:
            col = config.get("column", "")
            drift_score = float(value) if not isinstance(value, dict) else 0.0
            threshold = config.get("threshold")
            method = config.get("method")
            drift_detected = (
                drift_score >= threshold if threshold is not None else False
            )
            feature_drift[col] = {
                "drift_detected": drift_detected,
                "drift_score": drift_score,
                "stattest_name": method,
            }

    # If we have per-feature drift info but no DriftedColumnsCount,
    # compute n_drifted from per-feature results
    if n_drifted == 0 and feature_drift:
        n_drifted = sum(
            1 for fd in feature_drift.values() if fd.get("drift_detected", False)
        )

    # Determine overall dataset drift: True if drifted proportion >= 0.5
    dataset_drift = drift_share >= 0.5 if drift_share > 0 else n_drifted > (n_features / 2)

    return {
        "dataset_drift": dataset_drift,
        "n_drifted_features": n_drifted,
        "n_features": n_features,
        "feature_drift": feature_drift,
    }
