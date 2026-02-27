from __future__ import annotations

import pytest

from metrics_lie.model.adapter_registry import AdapterRegistry


def _fake_factory(**kwargs):
    return "fake_adapter"


def test_register_and_resolve_by_format():
    reg = AdapterRegistry()
    reg.register("sklearn", factory=_fake_factory, extensions={".pkl", ".pickle"})
    assert reg.resolve_format("sklearn") == _fake_factory


def test_resolve_by_extension():
    reg = AdapterRegistry()
    reg.register("sklearn", factory=_fake_factory, extensions={".pkl", ".pickle"})
    assert reg.resolve_extension(".pkl") == _fake_factory
    assert reg.resolve_extension(".pickle") == _fake_factory


def test_resolve_unknown_format_raises():
    reg = AdapterRegistry()
    with pytest.raises(KeyError, match="Unknown model format"):
        reg.resolve_format("unknown_format")


def test_resolve_unknown_extension_raises():
    reg = AdapterRegistry()
    with pytest.raises(KeyError, match="No adapter registered for extension"):
        reg.resolve_extension(".xyz")


def test_list_formats():
    reg = AdapterRegistry()
    reg.register("sklearn", factory=_fake_factory, extensions={".pkl"})
    reg.register("onnx", factory=_fake_factory, extensions={".onnx"})
    formats = reg.list_formats()
    assert "sklearn" in formats
    assert "onnx" in formats


def test_duplicate_format_raises():
    reg = AdapterRegistry()
    reg.register("sklearn", factory=_fake_factory, extensions={".pkl"})
    with pytest.raises(ValueError, match="already registered"):
        reg.register("sklearn", factory=_fake_factory, extensions={".pkl"})
