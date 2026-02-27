from __future__ import annotations

import hashlib
from typing import Any, Literal

import numpy as np

from metrics_lie.model.metadata import ModelMetadata
from metrics_lie.model.surface import (
    CalibrationState,
    PredictionSurface,
    SurfaceType,
    validate_surface,
)
from metrics_lie.task_types import TaskType


BoostingFramework = Literal["xgboost", "lightgbm", "catboost"]


class BoostingAdapter:
    """Adapter for gradient boosting models (XGBoost, LightGBM, CatBoost)."""

    def __init__(
        self,
        *,
        path: str,
        framework: BoostingFramework,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        threshold: float = 0.5,
        positive_label: int = 1,
    ) -> None:
        self._path = path
        self._framework = framework
        self._task_type = task_type
        self._threshold = threshold
        self._positive_label = positive_label
        self._model = self._load_model(path, framework)
        self._model_hash = self._compute_hash(path)

    @staticmethod
    def _compute_hash(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return f"sha256:{h.hexdigest()}"

    @staticmethod
    def _load_model(path: str, framework: BoostingFramework) -> Any:
        if framework == "xgboost":
            import xgboost as xgb

            model = xgb.XGBClassifier()
            model.load_model(path)
            return model
        elif framework == "lightgbm":
            import lightgbm as lgb

            return lgb.Booster(model_file=path)
        elif framework == "catboost":
            from catboost import CatBoostClassifier

            model = CatBoostClassifier()
            model.load_model(path)
            return model
        raise ValueError(f"Unknown boosting framework: {framework}")

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_class=self._model.__class__.__name__,
            model_module=self._framework,
            model_format=self._framework,
            model_hash=self._model_hash,
            capabilities={"predict", "predict_proba"},
        )

    def predict(self, X: np.ndarray) -> PredictionSurface:
        labels = np.asarray(self._model.predict(X)).flatten()
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
        if not callable(getattr(self._model, "predict_proba", None)):
            return None
        raw = np.asarray(self._model.predict_proba(X))
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
            model_hash=self._model_hash,
            is_deterministic=True,
        )

    def predict_raw(self, X: np.ndarray) -> dict[str, Any]:
        result: dict[str, Any] = {"labels": self._model.predict(X)}
        if callable(getattr(self._model, "predict_proba", None)):
            result["probabilities"] = self._model.predict_proba(X)
        return result

    def get_all_surfaces(self, X: np.ndarray) -> dict[SurfaceType, PredictionSurface]:
        surfaces: dict[SurfaceType, PredictionSurface] = {}
        surfaces[SurfaceType.LABEL] = self.predict(X)
        proba = self.predict_proba(X)
        if proba is not None:
            surfaces[SurfaceType.PROBABILITY] = proba
        return surfaces
