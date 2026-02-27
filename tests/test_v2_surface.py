"""Tests for multi-task PredictionSurface."""
from __future__ import annotations

import numpy as np

from metrics_lie.model.surface import (
    CalibrationState,
    PredictionSurface,
    SurfaceType,
    validate_surface,
)


def test_surface_accepts_empty_class_names():
    """Regression surfaces should accept empty class_names."""
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
    assert surface.class_names == ()


def test_surface_accepts_multiclass_names():
    """Multiclass surfaces should accept 3+ class names."""
    surface = PredictionSurface(
        surface_type=SurfaceType.PROBABILITY,
        values=np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]),
        dtype=np.dtype("float64"),
        n_samples=2,
        class_names=("class_0", "class_1", "class_2"),
        positive_label=None,
        threshold=None,
        calibration_state=CalibrationState.UNKNOWN,
        model_hash=None,
        is_deterministic=True,
    )
    assert len(surface.class_names) == 3


def test_validate_surface_2d_probability_multiclass():
    """2D probability array with 3 classes should validate."""
    values = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]])
    result = validate_surface(
        surface_type=SurfaceType.PROBABILITY,
        values=values,
        expected_n_samples=3,
        threshold=None,
        enforce_binary=False,
    )
    assert result.shape == (3, 3)


def test_validate_surface_2d_binary_still_works():
    """2D probability array with 2 classes should still validate."""
    values = np.array([[0.7, 0.3], [0.4, 0.6]])
    result = validate_surface(
        surface_type=SurfaceType.PROBABILITY,
        values=values,
        expected_n_samples=2,
        threshold=None,
        enforce_binary=False,
    )
    assert result.shape == (2, 2)
