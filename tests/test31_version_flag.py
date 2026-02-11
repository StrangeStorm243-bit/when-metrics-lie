"""Tests for Phase 2.8.2 version flag."""

import sys

import pytest

import metrics_lie
from metrics_lie import cli


def test_version_constant_exists():
    """Test that __version__ is defined in metrics_lie."""
    assert hasattr(metrics_lie, "__version__")
    assert isinstance(metrics_lie.__version__, str)
    assert len(metrics_lie.__version__) > 0


def test_version_flag_output(monkeypatch, capsys):
    """Test that --version flag prints version and exits cleanly."""
    # Patch sys.argv to simulate: spectra --version
    monkeypatch.setattr(sys, "argv", ["spectra", "--version"])

    # Should exit with code 0 (argparse's version action calls sys.exit(0))
    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    # SystemExit(0) is expected for --version
    assert exc_info.value.code == 0

    # Capture stdout
    captured = capsys.readouterr()
    output = captured.out

    # Should contain version string
    assert "Spectra" in output
    assert metrics_lie.__version__ in output


def test_version_flag_format(monkeypatch, capsys):
    """Test that --version output format is correct."""
    monkeypatch.setattr(sys, "argv", ["spectra", "--version"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    output = captured.out.strip()

    # Should be exactly "Spectra <version>"
    expected = f"Spectra {metrics_lie.__version__}"
    assert output == expected
