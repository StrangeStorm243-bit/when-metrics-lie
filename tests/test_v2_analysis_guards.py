"""Tests for analysis module task-type guards."""
from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.model.surface import PredictionSurface, SurfaceType, CalibrationState


def test_threshold_sweep_rejects_continuous():
    from metrics_lie.analysis.threshold_sweep import run_threshold_sweep
    surface = PredictionSurface(
        surface_type=SurfaceType.CONTINUOUS,
        values=np.array([1.0, 2.0, 3.0]),
        dtype=np.dtype("float64"),
        n_samples=3,
        class_names=(),
        positive_label=None,
        threshold=None,
        calibration_state=CalibrationState.UNKNOWN,
        model_hash=None,
        is_deterministic=True,
    )
    with pytest.raises(ValueError, match="PROBABILITY|SCORE"):
        run_threshold_sweep(
            y_true=np.array([1.0, 2.0, 3.0]),
            surface=surface,
            metrics=["mae"],
        )


def test_disagreement_returns_empty_for_continuous():
    from metrics_lie.analysis.disagreement import analyze_metric_disagreements
    surface = PredictionSurface(
        surface_type=SurfaceType.CONTINUOUS,
        values=np.array([1.0, 2.0, 3.0]),
        dtype=np.dtype("float64"),
        n_samples=3,
        class_names=(),
        positive_label=None,
        threshold=None,
        calibration_state=CalibrationState.UNKNOWN,
        model_hash=None,
        is_deterministic=True,
    )
    result = analyze_metric_disagreements(
        y_true=np.array([1.0, 2.0, 3.0]),
        surface=surface,
        thresholds={},
        metrics=["mae"],
    )
    assert result == []


def test_disagreement_uses_core_threshold_metrics():
    """disagreement.py should import THRESHOLD_METRICS from core, not redefine."""
    from metrics_lie.analysis import disagreement
    from metrics_lie.metrics.core import THRESHOLD_METRICS as CORE_TM
    assert disagreement.THRESHOLD_METRICS is CORE_TM
