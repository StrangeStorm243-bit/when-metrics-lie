from __future__ import annotations

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


class HTTPAdapter:
    """Adapter for models served via HTTP endpoints."""

    def __init__(
        self,
        *,
        endpoint: str,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        threshold: float = 0.5,
        positive_label: int = 1,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._task_type = task_type
        self._threshold = threshold
        self._positive_label = positive_label
        self._headers = headers or {"Content-Type": "application/json"}

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_class="HTTPModel",
            model_module="http",
            model_format="http",
            model_hash=None,
            capabilities={"predict", "predict_proba"},
        )

    def _call_endpoint(self, X: np.ndarray) -> list[dict[str, Any]]:
        import requests

        payload = {"instances": X.tolist()}
        resp = requests.post(
            self._endpoint,
            json=payload,
            headers=self._headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("predictions", [])

    def predict(self, X: np.ndarray) -> PredictionSurface:
        preds = self._call_endpoint(X)
        labels = np.array([p["label"] for p in preds])
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
            model_hash=None,
            is_deterministic=False,
        )

    def predict_proba(self, X: np.ndarray) -> PredictionSurface | None:
        preds = self._call_endpoint(X)
        if not preds or "probability" not in preds[0]:
            return None
        raw = np.array([p["probability"] for p in preds])
        if raw.ndim == 2 and raw.shape[1] > 1:
            proba = raw[:, 1]
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
            model_hash=None,
            is_deterministic=False,
        )

    def predict_raw(self, X: np.ndarray) -> dict[str, Any]:
        preds = self._call_endpoint(X)
        return {"predictions": preds}

    def get_all_surfaces(self, X: np.ndarray) -> dict[SurfaceType, PredictionSurface]:
        surfaces: dict[SurfaceType, PredictionSurface] = {}
        surfaces[SurfaceType.LABEL] = self.predict(X)
        proba = self.predict_proba(X)
        if proba is not None:
            surfaces[SurfaceType.PROBABILITY] = proba
        return surfaces
