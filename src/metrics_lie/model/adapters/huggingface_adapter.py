"""HuggingFace transformers pipeline adapter."""
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

try:
    import transformers
except ImportError:
    transformers = None  # type: ignore[assignment]


class HuggingFaceAdapter:
    """Model adapter for HuggingFace text-classification pipelines.

    Unlike other adapters, input is ``list[str]`` (text), not a numpy array.
    The adapter wraps a ``transformers.pipeline`` and extracts labels/scores.
    """

    def __init__(
        self,
        *,
        pipeline: Any,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        threshold: float = 0.5,
        positive_label: int = 1,
        positive_label_name: str = "POSITIVE",
    ) -> None:
        if transformers is None:
            raise ImportError(
                "transformers is required for HuggingFaceAdapter. "
                "Install it with: pip install 'metrics_lie[huggingface]'"
            )

        self._pipeline = pipeline
        self._task_type = task_type
        self._threshold = threshold
        self._positive_label = positive_label
        self._positive_label_name = positive_label_name

        # Extract model name from pipeline config if available
        self._model_name: str = "unknown"
        if hasattr(pipeline, "model") and hasattr(pipeline.model, "config"):
            config = pipeline.model.config
            if hasattr(config, "_name_or_path"):
                self._model_name = config._name_or_path

    @classmethod
    def from_model_path(
        cls,
        *,
        model_path: str,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        threshold: float = 0.5,
        positive_label: int = 1,
        positive_label_name: str = "POSITIVE",
    ) -> HuggingFaceAdapter:
        """Create adapter by loading a pipeline from a model path or name."""
        if transformers is None:
            raise ImportError(
                "transformers is required for HuggingFaceAdapter. "
                "Install it with: pip install 'metrics_lie[huggingface]'"
            )

        pipe = transformers.pipeline(
            "text-classification",
            model=model_path,
            return_all_scores=True,
        )
        return cls(
            pipeline=pipe,
            task_type=task_type,
            threshold=threshold,
            positive_label=positive_label,
            positive_label_name=positive_label_name,
        )

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_class="Pipeline",
            model_module="transformers",
            model_format="huggingface",
            model_hash=None,
            capabilities={"predict", "predict_proba"},
        )

    def _run_pipeline(self, texts: list[str]) -> list[list[dict[str, Any]]]:
        """Run the pipeline and return per-sample list of label-score dicts.

        The pipeline is called with ``return_all_scores=True`` (or the
        ``top_k=None`` equivalent) so that every sample returns a list
        of ``{"label": ..., "score": ...}`` dicts for all classes.
        """
        results = self._pipeline(texts, return_all_scores=True)
        # Ensure we always have list-of-lists structure
        if results and not isinstance(results[0], list):
            # Single-sample or already-flattened output
            results = [results]
        return results

    def predict(self, X: Any) -> PredictionSurface:
        """Return LABEL surface. X should be list[str]."""
        texts = list(X)
        results = self._run_pipeline(texts)

        labels = np.array([
            self._extract_top_label(sample_scores)
            for sample_scores in results
        ], dtype=int)

        arr = validate_surface(
            surface_type=SurfaceType.LABEL,
            values=labels,
            expected_n_samples=len(texts),
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
            is_deterministic=True,
        )

    def predict_proba(self, X: Any) -> PredictionSurface | None:
        """Return PROBABILITY surface. X should be list[str]."""
        texts = list(X)
        results = self._run_pipeline(texts)

        proba = np.array([
            self._extract_positive_score(sample_scores)
            for sample_scores in results
        ], dtype=float)

        arr = validate_surface(
            surface_type=SurfaceType.PROBABILITY,
            values=proba,
            expected_n_samples=len(texts),
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
            is_deterministic=True,
        )

    def predict_raw(self, X: Any) -> dict[str, Any]:
        """Return raw pipeline output."""
        texts = list(X)
        results = self._run_pipeline(texts)
        labels = [self._extract_top_label_name(s) for s in results]
        scores = [self._extract_positive_score(s) for s in results]
        return {"labels": labels, "scores": scores, "raw": results}

    def get_all_surfaces(self, X: Any) -> dict[SurfaceType, PredictionSurface]:
        """Return all available prediction surfaces."""
        surfaces: dict[SurfaceType, PredictionSurface] = {}
        surfaces[SurfaceType.LABEL] = self.predict(X)
        proba = self.predict_proba(X)
        if proba is not None:
            surfaces[SurfaceType.PROBABILITY] = proba
        return surfaces

    def _extract_top_label(self, sample_scores: list[dict[str, Any]]) -> int:
        """Extract the predicted class as 0/1 integer."""
        top = max(sample_scores, key=lambda x: x["score"])
        return self._positive_label if top["label"] == self._positive_label_name else 0

    def _extract_top_label_name(self, sample_scores: list[dict[str, Any]]) -> str:
        """Extract the top label name."""
        return max(sample_scores, key=lambda x: x["score"])["label"]

    def _extract_positive_score(self, sample_scores: list[dict[str, Any]]) -> float:
        """Extract the score for the positive class."""
        for entry in sample_scores:
            if entry["label"] == self._positive_label_name:
                return float(entry["score"])
        # Fallback: return score of top class
        return float(max(sample_scores, key=lambda x: x["score"])["score"])
