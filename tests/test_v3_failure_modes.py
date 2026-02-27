"""Tests for failure_modes with multiclass and regression surfaces."""
from __future__ import annotations

import numpy as np

from metrics_lie.analysis.failure_modes import locate_failure_modes
from metrics_lie.model.surface import CalibrationState, PredictionSurface, SurfaceType


def _make_surface(surface_type, values, **kwargs):
    return PredictionSurface(
        surface_type=surface_type,
        values=np.array(values),
        dtype=np.array(values).dtype,
        n_samples=len(values),
        class_names=kwargs.get("class_names", ("neg", "pos")),
        positive_label=kwargs.get("positive_label", 1),
        threshold=kwargs.get("threshold", 0.5),
        calibration_state=CalibrationState.UNKNOWN,
        model_hash=None,
        is_deterministic=True,
    )


def test_failure_modes_multiclass_probability():
    y_true = np.array([0, 1, 2, 0, 1])
    proba = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.2, 0.2, 0.6],
        [0.3, 0.4, 0.3],
        [0.5, 0.3, 0.2],
    ])
    surface = _make_surface(
        SurfaceType.PROBABILITY, proba, class_names=("a", "b", "c"), threshold=None
    )
    report = locate_failure_modes(
        y_true=y_true, surface=surface, metrics=["macro_f1"], top_k=3
    )
    assert report.total_samples == 5
    assert len(report.failure_samples) == 3
    assert 3 in report.failure_samples
    assert 4 in report.failure_samples


def test_failure_modes_regression_continuous():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    preds = np.array([1.1, 2.0, 5.0, 3.5, 5.2])
    surface = _make_surface(SurfaceType.CONTINUOUS, preds, threshold=None)
    report = locate_failure_modes(
        y_true=y_true, surface=surface, metrics=["mae"], top_k=2
    )
    assert report.total_samples == 5
    assert 2 in report.failure_samples


def test_failure_modes_multiclass_label():
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 2, 2, 1, 1])
    surface = _make_surface(SurfaceType.LABEL, y_pred, class_names=("a", "b", "c"))
    report = locate_failure_modes(
        y_true=y_true, surface=surface, metrics=["macro_f1"], top_k=2
    )
    assert 1 in report.failure_samples
    assert 3 in report.failure_samples


def test_failure_modes_binary_unchanged():
    """Binary PROBABILITY still works as before."""
    y_true = np.array([0, 1, 0, 1, 0])
    proba = np.array([0.9, 0.8, 0.3, 0.4, 0.1])
    surface = _make_surface(SurfaceType.PROBABILITY, proba)
    report = locate_failure_modes(
        y_true=y_true, surface=surface, metrics=["auc"], top_k=2
    )
    assert report.total_samples == 5
    assert len(report.failure_samples) == 2
