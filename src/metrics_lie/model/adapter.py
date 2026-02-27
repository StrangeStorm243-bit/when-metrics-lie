from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from metrics_lie.model.errors import CapabilityError, ModelNotFittedError
from metrics_lie.model.metadata import ModelMetadata
from metrics_lie.model.sources import ModelSource, ModelSourceCallable, load_model
from metrics_lie.model.surface import CalibrationState, PredictionSurface, SurfaceType, validate_surface
from metrics_lie.task_types import TaskType


@dataclass(frozen=True)
class ModelAdapterReport:
    model_class: str
    model_module: str
    model_hash: str | None
    capabilities: dict[str, bool]


class SklearnAdapter:
    def __init__(
        self,
        source: ModelSource,
        *,
        threshold: float = 0.5,
        positive_label: int = 1,
        calibration_state: CalibrationState = CalibrationState.UNKNOWN,
    ) -> None:
        self._source = source
        self._model, self._model_hash = load_model(source)
        self._threshold = float(threshold)
        self._positive_label = positive_label
        self._calibration_state = calibration_state
        self._enforce_cpu_only()
        self._ensure_fitted()

    def _enforce_cpu_only(self) -> None:
        module = getattr(self._model, "__module__", "")
        if module.startswith("torch") or module.startswith("tensorflow"):
            raise CapabilityError(
                "GPU-backed frameworks are not supported in Phase 5. Use CPU-only sklearn models."
            )

    def _ensure_fitted(self) -> None:
        # Only attempt sklearn check if it looks like a sklearn estimator.
        try:
            from sklearn.utils.validation import check_is_fitted  # type: ignore
        except Exception:
            return
        model = self._model
        module = getattr(model, "__module__", "")
        if module.startswith("sklearn"):
            try:
                check_is_fitted(model)
            except Exception as exc:
                raise ModelNotFittedError(
                    "Model appears to be unfitted. Call fit() before evaluation."
                ) from exc

    def detect_capabilities(self) -> dict[str, bool]:
        if isinstance(self._source, ModelSourceCallable):
            # Callables are treated as probability predictors.
            return {"predict": False, "predict_proba": True, "decision_function": False}
        model = self._model
        return {
            "predict": callable(getattr(model, "predict", None)),
            "predict_proba": callable(getattr(model, "predict_proba", None)),
            "decision_function": callable(getattr(model, "decision_function", None)),
        }

    def report(self) -> ModelAdapterReport:
        model = self._model
        return ModelAdapterReport(
            model_class=model.__class__.__name__,
            model_module=getattr(model, "__module__", "unknown"),
            model_hash=self._model_hash,
            capabilities=self.detect_capabilities(),
        )

    def _call(self, fn: Callable[[Any], Any], X: np.ndarray) -> np.ndarray:
        out = fn(X)
        return np.asarray(out)

    def predict(self, X: np.ndarray) -> PredictionSurface:
        model = self._model
        if isinstance(self._source, ModelSourceCallable):
            raise CapabilityError("Callable sources do not support predict()")
        if not callable(getattr(model, "predict", None)):
            raise CapabilityError("Model does not implement predict()")
        raw = self._call(model.predict, X)
        arr = validate_surface(
            surface_type=SurfaceType.LABEL,
            values=raw,
            expected_n_samples=X.shape[0],
            threshold=None,
        )
        return PredictionSurface(
            surface_type=SurfaceType.LABEL,
            values=arr.astype(int),
            dtype=arr.dtype,
            n_samples=int(arr.shape[0]),
            class_names=("negative", "positive"),
            positive_label=self._positive_label,
            threshold=None,
            calibration_state=self._calibration_state,
            model_hash=self._model_hash,
            is_deterministic=True,
        )

    def predict_proba(self, X: np.ndarray) -> PredictionSurface:
        if isinstance(self._source, ModelSourceCallable):
            raw = self._call(self._model, X)
        else:
            model = self._model
            if not callable(getattr(model, "predict_proba", None)):
                raise CapabilityError("Model does not implement predict_proba()")
            raw = self._call(model.predict_proba, X)

        arr = validate_surface(
            surface_type=SurfaceType.PROBABILITY,
            values=raw,
            expected_n_samples=X.shape[0],
            threshold=self._threshold,
        )
        if arr.ndim == 2:
            # Select column for positive label (default: column 1)
            pos_idx = 1 if arr.shape[1] > 1 else 0
            arr = arr[:, pos_idx]
        return PredictionSurface(
            surface_type=SurfaceType.PROBABILITY,
            values=arr.astype(float),
            dtype=arr.dtype,
            n_samples=int(arr.shape[0]),
            class_names=("negative", "positive"),
            positive_label=self._positive_label,
            threshold=self._threshold,
            calibration_state=self._calibration_state,
            model_hash=self._model_hash,
            is_deterministic=True,
        )

    def decision_function(self, X: np.ndarray) -> PredictionSurface:
        model = self._model
        if isinstance(self._source, ModelSourceCallable):
            raise CapabilityError("Callable sources do not support decision_function()")
        if not callable(getattr(model, "decision_function", None)):
            raise CapabilityError("Model does not implement decision_function()")
        raw = self._call(model.decision_function, X)
        arr = validate_surface(
            surface_type=SurfaceType.SCORE,
            values=raw,
            expected_n_samples=X.shape[0],
            threshold=None,
        )
        return PredictionSurface(
            surface_type=SurfaceType.SCORE,
            values=arr.astype(float),
            dtype=arr.dtype,
            n_samples=int(arr.shape[0]),
            class_names=("negative", "positive"),
            positive_label=self._positive_label,
            threshold=None,
            calibration_state=self._calibration_state,
            model_hash=self._model_hash,
            is_deterministic=True,
        )

    def get_all_surfaces(self, X: np.ndarray) -> dict[SurfaceType, PredictionSurface]:
        surfaces: dict[SurfaceType, PredictionSurface] = {}
        caps = self.detect_capabilities()
        if caps.get("predict_proba"):
            surfaces[SurfaceType.PROBABILITY] = self.predict_proba(X)
        if caps.get("predict"):
            surfaces[SurfaceType.LABEL] = self.predict(X)
        if caps.get("decision_function"):
            surfaces[SurfaceType.SCORE] = self.decision_function(X)
        return surfaces

    @property
    def task_type(self) -> TaskType:
        return TaskType.BINARY_CLASSIFICATION

    @property
    def metadata(self) -> ModelMetadata:
        caps = self.detect_capabilities()
        cap_set = {k for k, v in caps.items() if v}
        fmt = "callable" if isinstance(self._source, ModelSourceCallable) else "pickle"
        return ModelMetadata(
            model_class=self._model.__class__.__name__,
            model_module=getattr(self._model, "__module__", "unknown"),
            model_format=fmt,
            model_hash=self._model_hash,
            capabilities=cap_set,
        )

    def predict_raw(self, X: np.ndarray) -> dict[str, Any]:
        surfaces = self.get_all_surfaces(X)
        result: dict[str, Any] = {}
        if SurfaceType.PROBABILITY in surfaces:
            result["probabilities"] = surfaces[SurfaceType.PROBABILITY].values
        if SurfaceType.LABEL in surfaces:
            result["labels"] = surfaces[SurfaceType.LABEL].values
        if SurfaceType.SCORE in surfaces:
            result["scores"] = surfaces[SurfaceType.SCORE].values
        return result


# Backward compatibility alias
ModelAdapter = SklearnAdapter
