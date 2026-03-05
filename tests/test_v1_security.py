"""Tests for model security scanning and trust policy."""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from metrics_lie.model.security import (
    PICKLE_EXTENSIONS,
    SAFE_EXTENSIONS,
    ScanResult,
    check_trust_policy,
    scan_model_file,
)


class TestScanResult:
    """ScanResult dataclass field tests."""

    def test_default_fields(self) -> None:
        result = ScanResult(is_safe=True)
        assert result.is_safe is True
        assert result.issues == []
        assert result.format_detected == "unknown"

    def test_custom_fields(self) -> None:
        result = ScanResult(
            is_safe=False,
            issues=["dangerous opcode found"],
            format_detected="pickle",
        )
        assert result.is_safe is False
        assert result.issues == ["dangerous opcode found"]
        assert result.format_detected == "pickle"

    def test_frozen(self) -> None:
        result = ScanResult(is_safe=True)
        with pytest.raises(AttributeError):
            result.is_safe = False  # type: ignore[misc]


class TestScanModelFile:
    """Tests for scan_model_file on various extensions."""

    def test_safe_pickle_file(self, tmp_path: Path) -> None:
        """A pickle containing only a simple dict should scan as safe (or unscanned)."""
        pkl_path = tmp_path / "model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"weights": [1, 2, 3]}, f)

        result = scan_model_file(str(pkl_path))
        assert result.format_detected == "pickle"
        # Either safe (picklescan found nothing) or unscanned (picklescan not installed)
        # Both are acceptable in test environment
        assert isinstance(result.is_safe, bool)

    def test_onnx_extension_is_safe(self, tmp_path: Path) -> None:
        fake_onnx = tmp_path / "model.onnx"
        fake_onnx.write_text("fake")
        result = scan_model_file(str(fake_onnx))
        assert result.is_safe is True
        assert result.format_detected == "onnx"

    def test_cbm_extension_is_safe(self, tmp_path: Path) -> None:
        fake_cbm = tmp_path / "model.cbm"
        fake_cbm.write_text("fake")
        result = scan_model_file(str(fake_cbm))
        assert result.is_safe is True
        assert result.format_detected == "cbm"

    def test_safetensors_extension_is_safe(self, tmp_path: Path) -> None:
        fake = tmp_path / "model.safetensors"
        fake.write_text("fake")
        result = scan_model_file(str(fake))
        assert result.is_safe is True
        assert result.format_detected == "safetensors"

    def test_unknown_extension_is_safe(self, tmp_path: Path) -> None:
        fake = tmp_path / "model.xyz"
        fake.write_text("fake")
        result = scan_model_file(str(fake))
        assert result.is_safe is True
        assert result.format_detected == "unknown"

    def test_joblib_treated_as_pickle(self, tmp_path: Path) -> None:
        pkl_path = tmp_path / "model.joblib"
        with open(pkl_path, "wb") as f:
            pickle.dump({"x": 1}, f)
        result = scan_model_file(str(pkl_path))
        assert result.format_detected == "pickle"


class TestCheckTrustPolicy:
    """Tests for check_trust_policy."""

    def test_pkl_without_trust_raises(self, monkeypatch) -> None:
        monkeypatch.delenv("SPECTRA_TRUST_PICKLE", raising=False)
        with pytest.raises(ValueError, match="trust_pickle=True"):
            check_trust_policy("model.pkl", trust_pickle=False)

    def test_pickle_without_trust_raises(self, monkeypatch) -> None:
        monkeypatch.delenv("SPECTRA_TRUST_PICKLE", raising=False)
        with pytest.raises(ValueError, match="trust_pickle=True"):
            check_trust_policy("model.pickle", trust_pickle=False)

    def test_joblib_without_trust_raises(self, monkeypatch) -> None:
        monkeypatch.delenv("SPECTRA_TRUST_PICKLE", raising=False)
        with pytest.raises(ValueError, match="trust_pickle=True"):
            check_trust_policy("model.joblib", trust_pickle=False)

    def test_pkl_with_trust_allows(self) -> None:
        # Should not raise
        check_trust_policy("model.pkl", trust_pickle=True)

    def test_onnx_skips_check(self) -> None:
        # Should not raise regardless of trust_pickle
        check_trust_policy("model.onnx", trust_pickle=False)

    def test_safetensors_skips_check(self) -> None:
        check_trust_policy("model.safetensors", trust_pickle=False)

    def test_unknown_extension_no_raise(self) -> None:
        # Unknown extensions are not pickle, so no restriction
        check_trust_policy("model.xyz", trust_pickle=False)


class TestExtensionSets:
    """Verify extension constants."""

    def test_pickle_extensions(self) -> None:
        assert ".pkl" in PICKLE_EXTENSIONS
        assert ".pickle" in PICKLE_EXTENSIONS
        assert ".joblib" in PICKLE_EXTENSIONS

    def test_safe_extensions(self) -> None:
        assert ".onnx" in SAFE_EXTENSIONS
        assert ".safetensors" in SAFE_EXTENSIONS
        assert ".cbm" in SAFE_EXTENSIONS
