"""Nox automation for Spectra."""
from __future__ import annotations

import nox


@nox.session(python=["3.11", "3.12"])
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.install("-e", ".[dev]")
    session.run("pytest", "--tb=short", "-q")


@nox.session
def lint(session: nox.Session) -> None:
    """Run linting."""
    session.install("ruff>=0.3")
    session.run("ruff", "check", "src", "tests")


@nox.session
def docs(session: nox.Session) -> None:
    """Build documentation."""
    session.install("-e", ".[docs]")
    session.run("mkdocs", "build", "--strict")
