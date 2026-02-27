"""Tests for Typer-based CLI."""
from __future__ import annotations

from typer.testing import CliRunner

from metrics_lie.cli_app import app

runner = CliRunner()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "Spectra" in result.stdout


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "compare" in result.stdout
    assert "score" in result.stdout
    assert "metrics" in result.stdout
    assert "scenarios" in result.stdout


def test_metrics_list():
    result = runner.invoke(app, ["metrics", "list"])
    assert result.exit_code == 0
    assert "auc" in result.stdout


def test_metrics_list_with_task():
    result = runner.invoke(app, ["metrics", "list", "--task", "regression"])
    assert result.exit_code == 0
    assert "mae" in result.stdout
    assert "auc" not in result.stdout


def test_scenarios_list():
    result = runner.invoke(app, ["scenarios", "list"])
    assert result.exit_code == 0
    assert "label_noise" in result.stdout


def test_models_list():
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "pickle" in result.stdout
    assert "onnx" in result.stdout


def test_evaluate_command_help():
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "--dataset" in result.stdout
    assert "--metric" in result.stdout
