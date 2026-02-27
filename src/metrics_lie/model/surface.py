from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from metrics_lie.validation import validate_binary_labels, validate_labels, validate_no_inf, validate_no_nan, validate_numeric_dtype
from .errors import SurfaceValidationError


class SurfaceType(str, Enum):
    LABEL = "label"
    PROBABILITY = "probability"
    SCORE = "score"
    CONTINUOUS = "continuous"


class CalibrationState(str, Enum):
    UNKNOWN = "unknown"
    CALIBRATED = "calibrated"
    UNCALIBRATED = "uncalibrated"


@dataclass(frozen=True)
class PredictionSurface:
    surface_type: SurfaceType
    values: np.ndarray
    dtype: np.dtype
    n_samples: int
    class_names: tuple[str, ...]
    positive_label: int | str
    threshold: float | None
    calibration_state: CalibrationState
    model_hash: str | None
    is_deterministic: bool

    def to_jsonable(self) -> dict[str, Any]:
        vals = np.asarray(self.values)
        stats = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "q05": float(np.quantile(vals, 0.05)),
            "q95": float(np.quantile(vals, 0.95)),
        }
        value_range = {"min": float(np.min(vals)), "max": float(np.max(vals))}
        return {
            "surface_type": self.surface_type.value,
            "shape": list(vals.shape),
            "dtype": str(self.dtype),
            "n_samples": int(self.n_samples),
            "class_names": list(self.class_names),
            "positive_label": self.positive_label,
            "threshold": self.threshold,
            "calibration_state": self.calibration_state.value,
            "model_hash": self.model_hash,
            "is_deterministic": self.is_deterministic,
            "value_range": value_range,
            "statistics": stats,
        }


def validate_surface(
    *,
    surface_type: SurfaceType,
    values: np.ndarray,
    expected_n_samples: int,
    threshold: float | None,
    enforce_binary: bool = True,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim not in (1, 2):
        raise SurfaceValidationError(
            f"surface values must be 1d or 2d. Got shape {arr.shape}"
        )
    if arr.shape[0] != expected_n_samples:
        raise SurfaceValidationError(
            f"surface length mismatch: expected {expected_n_samples}, got {arr.shape[0]}"
        )
    try:
        validate_numeric_dtype(arr, "surface")
        validate_no_nan(arr, "surface values")
        validate_no_inf(arr, "surface values")
    except ValueError as e:
        raise SurfaceValidationError(str(e)) from e

    if surface_type == SurfaceType.PROBABILITY:
        if arr.ndim == 2 and arr.shape[1] < 2:
            raise SurfaceValidationError(
                f"probability surface must have at least 2 columns. Got {arr.shape}"
            )
        # Range checks for probabilities
        out_of_range = (arr < 0) | (arr > 1)
        if np.any(out_of_range):
            ratio = float(np.mean(out_of_range))
            if ratio > 0.05:
                raise SurfaceValidationError(
                    f"probability surface has {ratio:.2%} values out of [0,1]"
                )
            arr = np.clip(arr, 0.0, 1.0)
    elif surface_type == SurfaceType.LABEL:
        if arr.ndim != 1:
            raise SurfaceValidationError(f"label surface must be 1d. Got {arr.shape}")
        if enforce_binary:
            try:
                validate_binary_labels(arr, "label surface")
            except ValueError as e:
                raise SurfaceValidationError(str(e)) from e
        else:
            try:
                validate_labels(arr, "label surface")
            except ValueError as e:
                raise SurfaceValidationError(str(e)) from e
    elif surface_type == SurfaceType.CONTINUOUS:
        if arr.ndim != 1:
            raise SurfaceValidationError(f"continuous surface must be 1d. Got {arr.shape}")
    else:
        if arr.ndim != 1:
            raise SurfaceValidationError(f"score surface must be 1d. Got {arr.shape}")

    if surface_type not in (SurfaceType.PROBABILITY, SurfaceType.CONTINUOUS) and threshold is not None:
        raise SurfaceValidationError("threshold is only valid for probability surfaces")

    return arr
