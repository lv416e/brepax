"""Nox automation for BRepAX."""

from __future__ import annotations

import nox

nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = True

PYTHON_VERSIONS = ["3.11", "3.12", "3.13"]


@nox.session(python=PYTHON_VERSIONS)
def test(session: nox.Session) -> None:
    """Run the test suite."""
    session.install(".[dev]")
    session.run(
        "pytest",
        "--cov=brepax",
        "--cov-report=term-missing",
        "-x",
        *session.posargs,
    )


@nox.session
def lint(session: nox.Session) -> None:
    """Run linters."""
    session.install("ruff")
    session.run("ruff", "check", "src/", "tests/")
    session.run("ruff", "format", "--check", "src/", "tests/")


@nox.session
def typecheck(session: nox.Session) -> None:
    """Run mypy type checking."""
    session.install(".[dev]")
    session.run("mypy", "src/")


@nox.session
def docs(session: nox.Session) -> None:
    """Build documentation."""
    session.install(".[docs]")
    session.run("mkdocs", "build", "--strict")


@nox.session
def benchmark(session: nox.Session) -> None:
    """Run benchmarks."""
    session.install(".[dev]")
    session.run(
        "pytest",
        "tests/benchmarks/",
        "--benchmark-only",
        *session.posargs,
    )
