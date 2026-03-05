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
        protocol: str = "custom",
        model_name: str = "",
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._task_type = task_type
        self._threshold = threshold
        self._positive_label = positive_label
        self._headers = headers or {"Content-Type": "application/json"}
        self._model_name = model_name

        # Auto-detect KServe V2 from URL
        if protocol == "custom" and "/v2/" in endpoint:
            self._protocol = "kserve_v2"
        else:
            self._protocol = protocol

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

    def _build_kserve_url(self) -> str:
        """Build KServe V2 inference URL."""
        if self._model_name:
            return f"{self._endpoint}/v2/models/{self._model_name}/infer"
        # If endpoint already has /v2/ path, use as-is
        return self._endpoint

    def _format_kserve_request(self, X: np.ndarray) -> dict[str, Any]:
        """Format input as KServe V2 tensor request."""
        return {
            "inputs": [
                {
                    "name": "input",
                    "shape": list(X.shape),
                    "datatype": "FP64",
                    "data": X.flatten().tolist(),
                }
            ]
        }

    def _parse_kserve_response(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse KServe V2 tensor response into prediction dicts.

        KServe V2 returns: {"outputs": [{"name": "output", "shape": [...], "data": [...]}]}
        We convert to our internal format: [{"label": ..., "probability": ...}, ...]
        """
        outputs = data.get("outputs", [])
        if not outputs:
            return []

        # Find the primary output
        primary = outputs[0]
        values = primary.get("data", [])
        shape = primary.get("shape", [])

        n_samples = shape[0] if shape else len(values)

        # If shape is [n, classes] — treat as probabilities
        if len(shape) == 2 and shape[1] > 1:
            n_classes = shape[1]
            preds = []
            for i in range(n_samples):
                row = values[i * n_classes : (i + 1) * n_classes]
                label = int(np.argmax(row))
                preds.append({"label": label, "probability": row})
            return preds

        # If shape is [n] — treat as labels or scores
        preds = []
        for i in range(n_samples):
            val = values[i] if i < len(values) else 0
            if isinstance(val, float) and 0.0 <= val <= 1.0:
                label = int(val >= self._threshold)
                preds.append({"label": label, "probability": [1.0 - val, val]})
            else:
                preds.append({"label": int(val)})
        return preds

    def _call_endpoint(self, X: np.ndarray) -> list[dict[str, Any]]:
        import requests

        if self._protocol == "kserve_v2":
            url = self._build_kserve_url()
            payload = self._format_kserve_request(X)
        else:
            url = self._endpoint
            payload = {"instances": X.tolist()}

        resp = requests.post(
            url,
            json=payload,
            headers=self._headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        if self._protocol == "kserve_v2":
            return self._parse_kserve_response(data)
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
