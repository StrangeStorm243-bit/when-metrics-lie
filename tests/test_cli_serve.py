"""Tests for spectra serve command."""
from __future__ import annotations

from typer.testing import CliRunner
from metrics_lie.cli_app import app


runner = CliRunner()


def test_serve_help():
    """Serve command exists and shows help."""
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "Launch" in result.output or "web" in result.output.lower() or "API" in result.output


def test_serve_registers_as_command():
    """Serve is registered as a top-level command."""
    result = runner.invoke(app, ["--help"])
    assert "serve" in result.output
