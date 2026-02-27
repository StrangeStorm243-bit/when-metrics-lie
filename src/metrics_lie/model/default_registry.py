from __future__ import annotations

from metrics_lie.model.adapter_registry import AdapterRegistry


def _sklearn_factory(**kwargs):
    from metrics_lie.model.adapter import SklearnAdapter
    return SklearnAdapter(**kwargs)


def _onnx_factory(**kwargs):
    from metrics_lie.model.adapters.onnx_adapter import ONNXAdapter
    return ONNXAdapter(**kwargs)


def _boosting_factory(**kwargs):
    from metrics_lie.model.adapters.boosting_adapter import BoostingAdapter
    return BoostingAdapter(**kwargs)


def _http_factory(**kwargs):
    from metrics_lie.model.adapters.http_adapter import HTTPAdapter
    return HTTPAdapter(**kwargs)


_DEFAULT_REGISTRY: AdapterRegistry | None = None


def get_default_registry() -> AdapterRegistry:
    """Return the default adapter registry, creating it on first call.

    The registry is cached as a module-level singleton.  Optional formats
    (ONNX, XGBoost, LightGBM, CatBoost) are registered only when their
    runtime dependencies are importable.
    """
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is not None:
        return _DEFAULT_REGISTRY

    reg = AdapterRegistry()

    # sklearn (always available)
    reg.register("sklearn", factory=_sklearn_factory, extensions={".pkl", ".pickle", ".joblib"})

    # HTTP (always available)
    reg.register("http", factory=_http_factory, extensions=set())

    # ONNX (optional)
    try:
        import onnxruntime  # noqa: F401
        reg.register("onnx", factory=_onnx_factory, extensions={".onnx"})
    except ImportError:
        pass

    # XGBoost (optional)
    try:
        import xgboost  # noqa: F401
        reg.register("xgboost", factory=_boosting_factory, extensions={".ubj", ".xgb"})
    except ImportError:
        pass

    # LightGBM (optional)
    try:
        import lightgbm  # noqa: F401
        reg.register("lightgbm", factory=_boosting_factory, extensions={".lgb"})
    except ImportError:
        pass

    # CatBoost (optional)
    try:
        import catboost  # noqa: F401
        reg.register("catboost", factory=_boosting_factory, extensions={".cbm"})
    except ImportError:
        pass

    _DEFAULT_REGISTRY = reg
    return reg


def _reset_registry() -> None:
    """Reset the cached registry (for testing only)."""
    global _DEFAULT_REGISTRY
    _DEFAULT_REGISTRY = None
