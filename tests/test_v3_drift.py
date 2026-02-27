"""Tests for Evidently-powered drift detection."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# Evidently may fail to import on unsupported Python versions (e.g. 3.14)
# due to pydantic v1 compat issues. Handle both missing and broken imports.
try:
    import evidently  # noqa: F401
    from evidently.presets import DataDriftPreset  # noqa: F401
    HAS_EVIDENTLY = True
except Exception:
    HAS_EVIDENTLY = False

from metrics_lie.diagnostics.drift import compute_drift_report  # noqa: E402


# ---------- Mock-based tests (always run) ----------

def test_drift_report_parsing_no_drift():
    """Verify parsing logic with mocked evidently output (no drift case)."""
    mock_snapshot = MagicMock()
    mock_snapshot.dict.return_value = {
        "metrics": [
            {
                "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                "config": {"drift_share": 0.5},
                "value": {"count": 0, "share": 0.0},
            },
            {
                "metric_name": "ValueDrift(column=f0)",
                "config": {"column": "f0", "method": "ks", "threshold": 0.05},
                "value": 0.01,
            },
            {
                "metric_name": "ValueDrift(column=f1)",
                "config": {"column": "f1", "method": "ks", "threshold": 0.05},
                "value": 0.02,
            },
        ],
        "tests": [],
    }

    mock_report_cls = MagicMock()
    mock_report_cls.return_value.run.return_value = mock_snapshot

    mock_preset_cls = MagicMock()

    with patch("metrics_lie.diagnostics.drift.Report", mock_report_cls, create=True), \
         patch("metrics_lie.diagnostics.drift.DataDriftPreset", mock_preset_cls, create=True):
        # Patch the imports inside the function
        import metrics_lie.diagnostics.drift as drift_mod
        original_fn = drift_mod._run_drift_new_api

        def patched_run(reference, current, n_features):
            import sys
            mock_evidently = MagicMock()
            mock_evidently.Report = mock_report_cls
            mock_evidently.presets.DataDriftPreset = mock_preset_cls
            sys.modules["evidently"] = mock_evidently
            sys.modules["evidently.presets"] = mock_evidently.presets
            try:
                return original_fn(reference, current, n_features)
            finally:
                sys.modules.pop("evidently", None)
                sys.modules.pop("evidently.presets", None)

        with patch.object(drift_mod, "_run_drift_new_api", patched_run):
            report = compute_drift_report(
                reference=pd.DataFrame({"f0": [1.0, 2.0], "f1": [3.0, 4.0]}),
                current=pd.DataFrame({"f0": [1.1, 2.1], "f1": [3.1, 4.1]}),
            )

    assert report["dataset_drift"] is False
    assert report["n_drifted_features"] == 0
    assert report["n_features"] == 2
    assert "f0" in report["feature_drift"]
    assert "f1" in report["feature_drift"]
    assert report["feature_drift"]["f0"]["drift_detected"] is False
    assert report["feature_drift"]["f0"]["drift_score"] == 0.01


def test_drift_report_parsing_with_drift():
    """Verify parsing logic with mocked evidently output (drift detected)."""
    mock_snapshot = MagicMock()
    mock_snapshot.dict.return_value = {
        "metrics": [
            {
                "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                "config": {"drift_share": 0.5},
                "value": {"count": 2, "share": 1.0},
            },
            {
                "metric_name": "ValueDrift(column=f0)",
                "config": {"column": "f0", "method": "ks", "threshold": 0.05},
                "value": 0.98,
            },
            {
                "metric_name": "ValueDrift(column=f1)",
                "config": {"column": "f1", "method": "ks", "threshold": 0.05},
                "value": 0.97,
            },
        ],
        "tests": [],
    }

    mock_report_cls = MagicMock()
    mock_report_cls.return_value.run.return_value = mock_snapshot
    mock_preset_cls = MagicMock()

    import metrics_lie.diagnostics.drift as drift_mod
    original_fn = drift_mod._run_drift_new_api

    def patched_run(reference, current, n_features):
        import sys
        mock_evidently = MagicMock()
        mock_evidently.Report = mock_report_cls
        mock_evidently.presets.DataDriftPreset = mock_preset_cls
        sys.modules["evidently"] = mock_evidently
        sys.modules["evidently.presets"] = mock_evidently.presets
        try:
            return original_fn(reference, current, n_features)
        finally:
            sys.modules.pop("evidently", None)
            sys.modules.pop("evidently.presets", None)

    with patch.object(drift_mod, "_run_drift_new_api", patched_run):
        report = compute_drift_report(
            reference=pd.DataFrame({"f0": [1.0, 2.0], "f1": [3.0, 4.0]}),
            current=pd.DataFrame({"f0": [10.0, 20.0], "f1": [30.0, 40.0]}),
        )

    assert report["dataset_drift"] is True
    assert report["n_drifted_features"] == 2
    assert report["feature_drift"]["f0"]["drift_detected"] is True
    assert report["feature_drift"]["f0"]["drift_score"] == 0.98


# ---------- Integration tests (require evidently) ----------

@pytest.mark.skipif(not HAS_EVIDENTLY, reason="evidently not available")
def test_drift_report_no_drift():
    rng = np.random.default_rng(42)
    reference = pd.DataFrame({"f0": rng.normal(0, 1, 200), "f1": rng.normal(0, 1, 200)})
    current = pd.DataFrame({"f0": rng.normal(0, 1, 200), "f1": rng.normal(0, 1, 200)})
    report = compute_drift_report(reference=reference, current=current)
    assert "dataset_drift" in report
    assert "n_drifted_features" in report
    assert "feature_drift" in report


@pytest.mark.skipif(not HAS_EVIDENTLY, reason="evidently not available")
def test_drift_report_with_drift():
    rng = np.random.default_rng(42)
    reference = pd.DataFrame({"f0": rng.normal(0, 1, 200), "f1": rng.normal(0, 1, 200)})
    current = pd.DataFrame({"f0": rng.normal(5, 1, 200), "f1": rng.normal(5, 1, 200)})
    report = compute_drift_report(reference=reference, current=current)
    assert report["dataset_drift"] is True
    assert report["n_drifted_features"] > 0
