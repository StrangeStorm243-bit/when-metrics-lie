"""Model security scanning and trust policy enforcement.

Scans model files for known dangerous patterns (pickle deserialization attacks)
and enforces opt-in trust for formats that can execute arbitrary code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PICKLE_EXTENSIONS = {".pkl", ".pickle", ".joblib"}
SAFE_EXTENSIONS = {
    ".onnx",
    ".ubj",
    ".xgb",
    ".lgb",
    ".cbm",
    ".safetensors",
    ".pt",
    ".pth",
    ".keras",
    ".h5",
}


@dataclass(frozen=True)
class ScanResult:
    """Result of scanning a model file for security issues."""

    is_safe: bool
    issues: list[str] = field(default_factory=list)
    format_detected: str = "unknown"


def scan_model_file(path: str) -> ScanResult:
    """Scan a model file and return a ScanResult.

    Safe formats (.onnx, .safetensors, etc.) are returned immediately.
    Pickle formats are scanned with picklescan if available.
    Unknown formats are treated as safe (no scan needed).
    """
    ext = Path(path).suffix.lower()
    if ext in SAFE_EXTENSIONS:
        return ScanResult(is_safe=True, issues=[], format_detected=ext.lstrip("."))
    if ext in PICKLE_EXTENSIONS:
        return _scan_pickle(path)
    return ScanResult(is_safe=True, issues=[], format_detected="unknown")


def _scan_pickle(path: str) -> ScanResult:
    """Scan a pickle file using picklescan if available."""
    try:
        from picklescan.scanner import scan_file_path

        result = scan_file_path(path)
        issues: list[str] = []
        if result.infected_count > 0:
            for scan in result.scans:
                if scan.issues:
                    for issue in scan.issues:
                        issues.append(str(issue))
        return ScanResult(
            is_safe=result.infected_count == 0,
            issues=issues,
            format_detected="pickle",
        )
    except ImportError:
        return ScanResult(
            is_safe=True,
            issues=["picklescan not installed; pickle not scanned"],
            format_detected="pickle",
        )


def check_trust_policy(path: str, *, trust_pickle: bool = False) -> None:
    """Enforce trust policy for model files.

    Safe formats (ONNX, safetensors, etc.) are always allowed.
    Pickle formats require explicit opt-in via trust_pickle=True.
    Unknown formats are allowed (no restriction).

    Raises:
        ValueError: If a pickle file is loaded without trust_pickle=True.
    """
    ext = Path(path).suffix.lower()
    if ext in SAFE_EXTENSIONS:
        return
    if ext in PICKLE_EXTENSIONS and not trust_pickle:
        raise ValueError(
            f"Loading pickle files requires trust_pickle=True. "
            f"File: {path}. Pickle files can execute arbitrary code. "
            f"Use --trust-pickle on CLI or trust_pickle=True in SDK."
        )
