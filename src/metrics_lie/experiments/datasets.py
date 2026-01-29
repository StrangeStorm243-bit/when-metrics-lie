from __future__ import annotations

from pathlib import Path

from .identity import sha256_hex


def dataset_fingerprint_csv(path: str) -> str:
    """
    Compute a SHA-256 fingerprint for a CSV file based on its raw bytes.
    """
    p = Path(path)
    data = p.read_bytes()
    return sha256_hex(data)


