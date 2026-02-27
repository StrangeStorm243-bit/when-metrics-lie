"""Tests for Phase 2.8.1 CLI entrypoint wiring."""

import sys

import pytest

import metrics_lie.cli


def test_main_is_callable():
    """Test that metrics_lie.cli.main is callable."""
    assert callable(metrics_lie.cli.main)



def test_main_help_output(monkeypatch):
    """Test that main() can be called with --help and produces output."""
    # Patch sys.argv to simulate: spectra --help
    monkeypatch.setattr(sys, "argv", ["spectra", "--help"])

    # Should not raise an exception
    # Note: argparse will call sys.exit(0) on --help, so we catch SystemExit
    with pytest.raises(SystemExit) as exc_info:
        metrics_lie.cli.main()

    # SystemExit(0) is expected for --help
    assert exc_info.value.code == 0
