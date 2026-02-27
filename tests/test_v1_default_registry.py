from __future__ import annotations

from metrics_lie.model.default_registry import get_default_registry, _reset_registry


def setup_function():
    _reset_registry()


def test_default_registry_has_sklearn():
    reg = get_default_registry()
    formats = reg.list_formats()
    assert "sklearn" in formats


def test_default_registry_resolves_pkl():
    reg = get_default_registry()
    factory = reg.resolve_extension(".pkl")
    assert factory is not None


def test_default_registry_resolves_pickle():
    reg = get_default_registry()
    factory = reg.resolve_extension(".pickle")
    assert factory is not None


def test_default_registry_has_http():
    reg = get_default_registry()
    formats = reg.list_formats()
    assert "http" in formats


def test_default_registry_is_cached():
    reg1 = get_default_registry()
    reg2 = get_default_registry()
    assert reg1 is reg2
