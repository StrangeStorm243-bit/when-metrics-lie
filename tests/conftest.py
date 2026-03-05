from __future__ import annotations

import os


def pytest_configure(config):
    """Set SPECTRA_TRUST_PICKLE=1 for all tests so pickle model loading works."""
    os.environ.setdefault("SPECTRA_TRUST_PICKLE", "1")
