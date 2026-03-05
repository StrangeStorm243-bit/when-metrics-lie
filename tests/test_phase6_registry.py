"""Tests for Phase 6 default registry MLflow integration."""
from __future__ import annotations

from unittest.mock import patch


from metrics_lie.model.default_registry import get_default_registry, _reset_registry


def test_registry_includes_sklearn_and_http():
    """Default registry always has sklearn and http."""
    _reset_registry()
    reg = get_default_registry()
    formats = reg.list_formats()
    assert "sklearn" in formats
    assert "http" in formats


def test_registry_mlflow_when_available():
    """MLflow format is registered when mlflow is importable."""
    _reset_registry()
    with patch.dict("sys.modules", {"mlflow": __import__("types")}):
        _reset_registry()
        reg = get_default_registry()
        formats = reg.list_formats()
        assert "mlflow" in formats
    _reset_registry()


def test_registry_no_mlflow_when_missing():
    """MLflow format is NOT registered when mlflow is not importable."""
    _reset_registry()
    # Force mlflow import to fail
    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("no mlflow")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        _reset_registry()
        reg = get_default_registry()
        formats = reg.list_formats()
        # mlflow should NOT be in the list
        assert "mlflow" not in formats
    _reset_registry()
