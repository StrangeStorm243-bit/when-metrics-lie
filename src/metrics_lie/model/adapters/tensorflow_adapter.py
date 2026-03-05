"""TensorFlow/Keras model adapter."""
from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np

from metrics_lie.model.metadata import ModelMetadata
from metrics_lie.model.surface import (
    CalibrationState,
    PredictionSurface,
    SurfaceType,
    validate_surface,
)
from metrics_lie.task_types import TaskType


class TensorFlowAdapter:
    """Model adapter for TensorFlow/Keras saved models.

    Loads Keras models (``*.keras``, ``*.h5``) via
    ``tf.keras.models.load_model`` and runs inference with
    ``model.predict()``.
    """

    def __init__(
        self,
        *,
        path: str,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        threshold: float = 0.5,
        positive_label: int = 1,
    ) -> None:
        try:
            import tensorflow  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for TensorFlowAdapter. "
                "Install it with: pip install 'metrics_lie[tensorflow]'"
            ) from exc

        import tensorflow as _tf

        self._path = path
        self._task_type = task_type
        self._threshold = threshold
        self._positive_label = positive_label

        self._model = _tf.keras.models.load_model(path)
        self._model_hash = self._compute_hash(path)

    @staticmethod
    def _compute_hash(path: str) -> str:
        """Compute SHA-256 hash of the model file or directory."""
        h = hashlib.sha256()
        if os.path.isdir(path):
            # SavedModel format is a directory; hash all files
            for root, _dirs, files in sorted(os.walk(path)):
                for fname in sorted(files):
                    fpath = os.path.join(root, fname)
                    with open(fpath, "rb") as f:
                        for chunk in iter(lambda: f.read(1024 * 1024), b""):
                            h.update(chunk)
        else:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
        return f"sha256:{h.hexdigest()}"

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_class=self._model.__class__.__name__,
            model_module="tensorflow",
            model_format="keras",
            model_hash=self._model_hash,
            capabilities={"predict", "predict_proba"},
        )

    def _inference(self, X: np.ndarray) -> np.ndarray:
        """Run inference and return numpy output."""
        return np.asarray(self._model.predict(X, verbose=0))

    def predict(self, X: np.ndarray) -> PredictionSurface:
        """Return LABEL surface from model output."""
        raw = self._inference(X)
        if raw.ndim == 2 and raw.shape[1] > 1:
            labels = np.argmax(raw, axis=1)
        else:
            labels = (raw.flatten() >= self._threshold).astype(int)
        arr = validate_surface(
            surface_type=SurfaceType.LABEL,
            values=labels,
            expected_n_samples=X.shape[0],
            threshold=None,
            enforce_binary=self._task_type == TaskType.BINARY_CLASSIFICATION,
        )
        return PredictionSurface(
            surface_type=SurfaceType.LABEL,
            values=arr.astype(int),
            dtype=arr.dtype,
            n_samples=int(arr.shape[0]),
            class_names=("negative", "positive"),
            positive_label=self._positive_label,
            threshold=None,
            calibration_state=CalibrationState.UNKNOWN,
            model_hash=self._model_hash,
            is_deterministic=True,
        )

    def predict_proba(self, X: np.ndarray) -> PredictionSurface | None:
        """Return PROBABILITY surface from softmax output."""
        raw = self._inference(X)
        if raw.ndim == 2 and raw.shape[1] > 1:
            if self._task_type == TaskType.BINARY_CLASSIFICATION:
                proba = raw[:, 1]
            else:
                proba = np.max(raw, axis=1)
        else:
            proba = raw.flatten()

        arr = validate_surface(
            surface_type=SurfaceType.PROBABILITY,
            values=proba,
            expected_n_samples=X.shape[0],
            threshold=self._threshold,
        )
        return PredictionSurface(
            surface_type=SurfaceType.PROBABILITY,
            values=arr.astype(float),
            dtype=arr.dtype,
            n_samples=int(arr.shape[0]),
            class_names=("negative", "positive"),
            positive_label=self._positive_label,
            threshold=self._threshold,
            calibration_state=CalibrationState.UNKNOWN,
            model_hash=self._model_hash,
            is_deterministic=True,
        )

    def predict_raw(self, X: np.ndarray) -> dict[str, Any]:
        """Return raw model outputs as a dict."""
        raw = self._inference(X)
        result: dict[str, Any] = {}
        if raw.ndim == 2 and raw.shape[1] > 1:
            result["labels"] = np.argmax(raw, axis=1)
            result["probabilities"] = raw
        else:
            flat = raw.flatten()
            result["labels"] = (flat >= self._threshold).astype(int)
            result["probabilities"] = flat
        return result

    def get_all_surfaces(
        self, X: np.ndarray
    ) -> dict[SurfaceType, PredictionSurface]:
        """Return all available prediction surfaces."""
        surfaces: dict[SurfaceType, PredictionSurface] = {}
        surfaces[SurfaceType.LABEL] = self.predict(X)
        proba = self.predict_proba(X)
        if proba is not None:
            surfaces[SurfaceType.PROBABILITY] = proba
        return surfaces
