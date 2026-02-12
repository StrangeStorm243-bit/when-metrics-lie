"""Validate sklearn pickle models for Phase 7 upload.

Only binary classification models with predict_proba are accepted.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ModelValidationResult:
    """Result of validating an uploaded pickle model."""

    valid: bool
    model_class: str = ""
    capabilities: dict[str, bool] = field(
        default_factory=lambda: {"predict": False, "predict_proba": False, "decision_function": False}
    )
    error: str | None = None


def validate_sklearn_pickle(file_bytes: bytes) -> ModelValidationResult:
    """
    Validate a pickle file as a binary-classification sklearn-compatible model.

    1. Unpickle the bytes
    2. Check hasattr(model, 'predict_proba')
    3. Run model.predict_proba on a small test input
    4. Verify output shape is (n, 2) for binary classification

    Returns ModelValidationResult with valid, model_class, capabilities, and error if invalid.
    """
    try:
        model = pickle.loads(file_bytes)
    except Exception as e:
        return ModelValidationResult(
            valid=False,
            error=f"Invalid pickle file: {e!s}",
        )

    model_class = type(model).__name__
    module = type(model).__module__
    if module:
        model_class = f"{module}.{model_class}"

    capabilities = {
        "predict": hasattr(model, "predict") and callable(getattr(model, "predict")),
        "predict_proba": hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba")),
        "decision_function": hasattr(model, "decision_function")
        and callable(getattr(model, "decision_function")),
    }

    if not capabilities["predict_proba"]:
        return ModelValidationResult(
            valid=False,
            model_class=model_class,
            capabilities=capabilities,
            error="Model must support predict_proba for binary classification",
        )

    # Test with minimal binary-classification-style input: 2 samples, n_features
    n_features = getattr(model, "n_features_in_", 1)
    try:
        X = np.zeros((2, n_features), dtype=np.float64)
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
    except Exception as e:
        return ModelValidationResult(
            valid=False,
            model_class=model_class,
            capabilities=capabilities,
            error=f"predict_proba failed: {e!s}",
        )

    if proba.ndim != 2:
        return ModelValidationResult(
            valid=False,
            model_class=model_class,
            capabilities=capabilities,
            error="Model output must be 2D (n_samples, n_classes)",
        )
    if proba.shape[1] != 2:
        return ModelValidationResult(
            valid=False,
            model_class=model_class,
            capabilities=capabilities,
            error="Model output must be shape (n, 2) for binary classification",
        )

    return ModelValidationResult(
        valid=True,
        model_class=model_class,
        capabilities=capabilities,
    )
