from __future__ import annotations

import hashlib
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


class ONNXAdapter:
    """Model adapter for ONNX format models via onnxruntime."""

    def __init__(
        self,
        *,
        path: str,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        threshold: float = 0.5,
        positive_label: int = 1,
    ) -> None:
        import onnxruntime as ort

        self._path = path
        self._task_type = task_type
        self._threshold = threshold
        self._positive_label = positive_label

        self._session = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._model_hash = self._compute_hash(path)

    @staticmethod
    def _compute_hash(path: str) -> str:
        h = hashlib.sha256()
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
            model_class="ONNXModel",
            model_module="onnxruntime",
            model_format="onnx",
            model_hash=self._model_hash,
            capabilities={"predict", "predict_proba"},
        )

    @staticmethod
    def _coerce_proba_output(raw: Any) -> np.ndarray:
        """Convert ONNX probability output to a numeric NumPy array.

        sklearn-converted ONNX classifiers often emit probabilities as a list
        of dicts (e.g. ``[{0: 0.6, 1: 0.4}, ...]``).  This helper normalises
        that to a float64 2-D array ``(n_samples, n_classes)``.
        """
        arr = np.asarray(raw)
        if arr.dtype == object and arr.ndim == 1:
            # list-of-dicts format from ZipMap operator
            dicts: list[dict[Any, float]] = list(raw)
            keys = sorted(dicts[0].keys())
            arr = np.array([[d[k] for k in keys] for d in dicts], dtype=np.float64)
        return arr

    def _run(self, X: np.ndarray) -> list[Any]:
        X = np.asarray(X, dtype=np.float32)
        return self._session.run(self._output_names, {self._input_name: X})

    def predict(self, X: np.ndarray) -> PredictionSurface:
        outputs = self._run(X)
        labels = np.asarray(outputs[0]).flatten()
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
        outputs = self._run(X)
        if len(outputs) < 2:
            return None
        proba = self._coerce_proba_output(outputs[1])
        if proba.ndim == 2 and proba.shape[1] > 1:
            pos_idx = 1 if proba.shape[1] > 1 else 0
            proba_1d = proba[:, pos_idx]
        else:
            proba_1d = proba.flatten()
        arr = validate_surface(
            surface_type=SurfaceType.PROBABILITY,
            values=proba_1d,
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
        outputs = self._run(X)
        result: dict[str, Any] = {}
        if len(outputs) >= 1:
            result["labels"] = np.asarray(outputs[0]).flatten()
        if len(outputs) >= 2:
            result["probabilities"] = self._coerce_proba_output(outputs[1])
        return result

    def get_all_surfaces(self, X: np.ndarray) -> dict[SurfaceType, PredictionSurface]:
        surfaces: dict[SurfaceType, PredictionSurface] = {}
        surfaces[SurfaceType.LABEL] = self.predict(X)
        proba = self.predict_proba(X)
        if proba is not None:
            surfaces[SurfaceType.PROBABILITY] = proba
        return surfaces
