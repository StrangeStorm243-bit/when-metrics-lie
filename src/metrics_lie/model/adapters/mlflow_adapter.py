"""MLflow model adapter — load models from MLflow registry."""
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


class MLflowAdapter:
    """Adapter for models loaded from MLflow model registry.

    Supports MLflow model URIs:
      - runs:/run_id/model
      - models:/model_name/version
      - models:/model_name/stage
    """

    def __init__(
        self,
        *,
        uri: str,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        threshold: float = 0.5,
        positive_label: int = 1,
    ) -> None:
        try:
            import mlflow.pyfunc
        except ImportError as e:
            raise ImportError(
                "MLflow adapter requires: pip install metrics_lie[mlflow]"
            ) from e

        self._uri = uri
        self._task_type = task_type
        self._threshold = threshold
        self._positive_label = positive_label
        self._model = mlflow.pyfunc.load_model(uri)
        self._model_info = self._model.metadata

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    @property
    def metadata(self) -> ModelMetadata:
        model_class = "mlflow.pyfunc"
        if self._model_info and hasattr(self._model_info, "flavors"):
            flavors = self._model_info.flavors or {}
            if "sklearn" in flavors:
                model_class = "mlflow.sklearn"
            elif "xgboost" in flavors:
                model_class = "mlflow.xgboost"
            elif "lightgbm" in flavors:
                model_class = "mlflow.lightgbm"
        return ModelMetadata(
            model_class=model_class,
            model_module="mlflow",
            model_format="mlflow",
            model_hash=None,
            capabilities={"predict", "predict_proba"},
        )

    def predict(self, X: np.ndarray) -> PredictionSurface:
        import pandas as pd

        df = pd.DataFrame(X)
        raw = self._model.predict(df)
        preds = np.asarray(raw)

        if self._task_type.is_regression:
            arr = validate_surface(
                surface_type=SurfaceType.CONTINUOUS,
                values=preds.flatten(),
                expected_n_samples=X.shape[0],
                threshold=None,
            )
            return PredictionSurface(
                surface_type=SurfaceType.CONTINUOUS,
                values=arr.astype(float),
                dtype=arr.dtype,
                n_samples=int(arr.shape[0]),
                class_names=None,
                positive_label=None,
                threshold=None,
                calibration_state=CalibrationState.UNKNOWN,
                model_hash=None,
                is_deterministic=True,
            )

        # Classification: predict returns labels
        if preds.ndim == 2:
            labels = np.argmax(preds, axis=1)
        else:
            labels = preds.astype(int)

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
            class_names=None,
            positive_label=self._positive_label,
            threshold=None,
            calibration_state=CalibrationState.UNKNOWN,
            model_hash=None,
            is_deterministic=True,
        )

    def predict_proba(self, X: np.ndarray) -> PredictionSurface | None:
        """Attempt probabilistic prediction via MLflow model.

        MLflow pyfunc.predict() may return probabilities if the underlying
        model supports them. We detect this from output shape.
        """
        import pandas as pd

        df = pd.DataFrame(X)
        raw = self._model.predict(df)
        preds = np.asarray(raw)

        # If output is 2D with >1 columns, treat as class probabilities
        if preds.ndim == 2 and preds.shape[1] > 1:
            if self._task_type == TaskType.BINARY_CLASSIFICATION:
                proba = preds[:, 1]
            else:
                # Multiclass: return full probability array
                proba = preds
        elif preds.ndim == 1:
            vals = preds.astype(float)
            if np.all((vals >= 0) & (vals <= 1)):
                proba = vals
            else:
                return None
        else:
            return None

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
            class_names=None,
            positive_label=self._positive_label,
            threshold=self._threshold,
            calibration_state=CalibrationState.UNKNOWN,
            model_hash=None,
            is_deterministic=True,
        )

    def predict_raw(self, X: np.ndarray) -> dict[str, Any]:
        import pandas as pd

        df = pd.DataFrame(X)
        raw = self._model.predict(df)
        return {"predictions": np.asarray(raw).tolist()}

    def get_all_surfaces(self, X: np.ndarray) -> dict[SurfaceType, PredictionSurface]:
        surfaces: dict[SurfaceType, PredictionSurface] = {}
        label_surface = self.predict(X)
        surfaces[label_surface.surface_type] = label_surface
        proba = self.predict_proba(X)
        if proba is not None:
            surfaces[SurfaceType.PROBABILITY] = proba
        return surfaces
