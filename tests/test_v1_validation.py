from __future__ import annotations

import numpy as np
import pytest

from metrics_lie.validation import validate_labels, validate_binary_labels
from metrics_lie.model.surface import SurfaceType, validate_surface


def test_validate_binary_labels_still_works():
    arr = np.array([0, 1, 0, 1])
    validate_binary_labels(arr, "test")


def test_validate_binary_labels_rejects_multiclass():
    arr = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="binary"):
        validate_binary_labels(arr, "test")


def test_validate_labels_accepts_multiclass():
    arr = np.array([0, 1, 2, 3, 0])
    validate_labels(arr, "test")


def test_validate_labels_accepts_binary():
    arr = np.array([0, 1, 0, 1])
    validate_labels(arr, "test")


def test_validate_labels_rejects_nan():
    arr = np.array([0.0, np.nan, 1.0])
    with pytest.raises(ValueError, match="NaN"):
        validate_labels(arr, "test")


def test_validate_surface_label_multiclass():
    arr = np.array([0, 1, 2, 3])
    result = validate_surface(
        surface_type=SurfaceType.LABEL,
        values=arr,
        expected_n_samples=4,
        threshold=None,
        enforce_binary=False,
    )
    assert len(result) == 4


def test_validate_surface_label_binary_default():
    arr = np.array([0, 1, 0, 1])
    result = validate_surface(
        surface_type=SurfaceType.LABEL,
        values=arr,
        expected_n_samples=4,
        threshold=None,
    )
    assert len(result) == 4


def test_validate_surface_regression():
    arr = np.array([1.5, 2.3, -0.7, 100.0])
    result = validate_surface(
        surface_type=SurfaceType.CONTINUOUS,
        values=arr,
        expected_n_samples=4,
        threshold=None,
    )
    assert len(result) == 4


def test_surface_type_continuous_exists():
    assert SurfaceType.CONTINUOUS == "continuous"
