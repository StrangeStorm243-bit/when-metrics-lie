"""PyTorch TorchScript model adapter."""
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


class PyTorchAdapter:
    """Model adapter for PyTorch TorchScript models.

    Loads TorchScript models (``*.pt``, ``*.pth``) via ``torch.jit.load``
    and runs inference on CPU with ``torch.no_grad()``.
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
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for PyTorchAdapter. "
                "Install it with: pip install 'metrics_lie[pytorch]'"
            ) from exc

        import torch as _torch

        self._torch = _torch
        self._path = path
        self._task_type = task_type
        self._threshold = threshold
        self._positive_label = positive_label

        self._model = _torch.jit.load(path, map_location="cpu")
        self._model.eval()
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
            model_class=self._model.__class__.__name__,
            model_module="torch",
            model_format="torchscript",
            model_hash=self._model_hash,
            capabilities={"predict", "predict_proba"},
        )

    def _inference(self, X: np.ndarray) -> np.ndarray:
        """Run inference: convert to float32 tensor, forward pass, return numpy."""
        tensor = self._torch.tensor(X, dtype=self._torch.float32)
        with self._torch.no_grad():
            output = self._model(tensor)
        return output.cpu().numpy()

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
                # Multiclass: return positive-class column (col 1) for surface validation
                # Surface is 1-D for compatibility with the rest of the engine
                proba = raw[:, 1] if raw.shape[1] == 2 else raw[:, 0]
                # Store full matrix in a separate attribute if needed later;
                # for now, return the max probability per sample for the surface
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
