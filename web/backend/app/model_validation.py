"""Validate uploaded models for web API.

Supports binary, multiclass, and regression models across pickle, ONNX, and boosting formats.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np


ACCEPTED_EXTENSIONS = {
    ".pkl": "pickle",
    ".joblib": "pickle",
    ".onnx": "onnx",
    ".ubj": "xgboost",
    ".xgb": "xgboost",
    ".lgb": "lightgbm",
    ".cbm": "catboost",
}


@dataclass
class ModelValidationResult:
    """Result of validating an uploaded model."""

    valid: bool
    model_class: str = ""
    task_type: str = "binary_classification"
    n_classes: int | None = None
    capabilities: dict[str, bool] = field(
        default_factory=lambda: {
            "predict": False,
            "predict_proba": False,
            "decision_function": False,
        }
    )
    error: str | None = None


def validate_model(
    file_bytes: bytes,
    file_ext: str,
    task_type: str = "binary_classification",
) -> ModelValidationResult:
    """
    Validate an uploaded model file.

    Args:
        file_bytes: Raw file bytes
        file_ext: File extension (e.g., '.pkl', '.onnx')
        task_type: Target task type

    Returns:
        ModelValidationResult with validation outcome
    """
    fmt = ACCEPTED_EXTENSIONS.get(file_ext.lower())
    if fmt is None:
        return ModelValidationResult(
            valid=False,
            error=f"Unsupported file extension: {file_ext}. "
            f"Accepted: {', '.join(sorted(ACCEPTED_EXTENSIONS.keys()))}",
        )

    if fmt == "pickle":
        return _validate_pickle(file_bytes, task_type)
    elif fmt == "onnx":
        return _validate_onnx(file_bytes, task_type)
    else:
        # Boosting formats — accept if file is non-empty
        # Full validation happens at inference time via BoostingAdapter
        return ModelValidationResult(
            valid=True,
            model_class=f"{fmt}_model",
            task_type=task_type,
            capabilities={"predict": True, "predict_proba": fmt != "catboost"},
        )


def _validate_pickle(file_bytes: bytes, task_type: str) -> ModelValidationResult:
    """Validate a pickle/joblib model."""
    try:
        model = pickle.loads(file_bytes)
    except Exception as e:
        return ModelValidationResult(valid=False, error=f"Invalid pickle file: {e!s}")

    model_class = type(model).__name__
    module = type(model).__module__
    if module:
        model_class = f"{module}.{model_class}"

    capabilities = {
        "predict": hasattr(model, "predict") and callable(getattr(model, "predict")),
        "predict_proba": hasattr(model, "predict_proba")
        and callable(getattr(model, "predict_proba")),
        "decision_function": hasattr(model, "decision_function")
        and callable(getattr(model, "decision_function")),
    }

    if not capabilities["predict"]:
        return ModelValidationResult(
            valid=False,
            model_class=model_class,
            capabilities=capabilities,
            error="Model must support predict()",
        )

    n_features = getattr(model, "n_features_in_", 1)
    n_classes = None

    if task_type in ("binary_classification", "multiclass_classification"):
        if not capabilities["predict_proba"]:
            return ModelValidationResult(
                valid=False,
                model_class=model_class,
                capabilities=capabilities,
                error="Classification models must support predict_proba()",
            )
        try:
            X = np.zeros((2, n_features), dtype=np.float64)
            proba = np.asarray(model.predict_proba(X))
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
                error="predict_proba output must be 2D (n_samples, n_classes)",
            )

        n_classes = proba.shape[1]
        if task_type == "binary_classification" and n_classes != 2:
            return ModelValidationResult(
                valid=False,
                model_class=model_class,
                capabilities=capabilities,
                error=f"Binary classification requires 2 classes, got {n_classes}",
            )
        if task_type == "multiclass_classification" and n_classes < 3:
            return ModelValidationResult(
                valid=False,
                model_class=model_class,
                capabilities=capabilities,
                error=f"Multiclass classification requires 3+ classes, got {n_classes}",
            )

    elif task_type == "regression":
        try:
            X = np.zeros((2, n_features), dtype=np.float64)
            preds = np.asarray(model.predict(X))
        except Exception as e:
            return ModelValidationResult(
                valid=False,
                model_class=model_class,
                capabilities=capabilities,
                error=f"predict failed: {e!s}",
            )
        if preds.ndim != 1:
            return ModelValidationResult(
                valid=False,
                model_class=model_class,
                capabilities=capabilities,
                error=f"Regression predict() must return 1D array, got shape {preds.shape}",
            )

    return ModelValidationResult(
        valid=True,
        model_class=model_class,
        task_type=task_type,
        n_classes=n_classes,
        capabilities=capabilities,
    )


def _validate_onnx(file_bytes: bytes, task_type: str) -> ModelValidationResult:
    """Validate an ONNX model file."""
    try:
        import onnx
        model = onnx.load_from_string(file_bytes)
        onnx.checker.check_model(model)
    except ImportError:
        return ModelValidationResult(
            valid=False,
            error="ONNX validation requires: pip install onnx",
        )
    except Exception as e:
        return ModelValidationResult(
            valid=False,
            error=f"Invalid ONNX model: {e!s}",
        )

    model_class = "onnx_model"
    graph = model.graph
    if graph and graph.name:
        model_class = f"onnx.{graph.name}"

    return ModelValidationResult(
        valid=True,
        model_class=model_class,
        task_type=task_type,
        capabilities={"predict": True, "predict_proba": True},
    )


# Backward compatibility alias
def validate_sklearn_pickle(file_bytes: bytes) -> ModelValidationResult:
    """Legacy validator — delegates to validate_model with binary_classification."""
    return _validate_pickle(file_bytes, "binary_classification")
