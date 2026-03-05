"""Integration tests for security scanning in the execution pipeline."""
from __future__ import annotations

from metrics_lie.spec import ModelSourceSpec


class TestModelSourceSpecTrustPickle:
    """Tests for the trust_pickle field on ModelSourceSpec."""

    def test_default_is_false(self) -> None:
        spec = ModelSourceSpec(kind="pickle", path="model.pkl")
        assert spec.trust_pickle is False

    def test_can_set_true(self) -> None:
        spec = ModelSourceSpec(kind="pickle", path="model.pkl", trust_pickle=True)
        assert spec.trust_pickle is True

    def test_onnx_does_not_need_trust_pickle(self) -> None:
        spec = ModelSourceSpec(kind="onnx", path="model.onnx")
        assert spec.trust_pickle is False
        # ONNX specs work fine without trust_pickle

    def test_import_kind_default(self) -> None:
        spec = ModelSourceSpec(kind="import", import_path="module:attr")
        assert spec.trust_pickle is False
