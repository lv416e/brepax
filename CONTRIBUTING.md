# Contributing to BRepAX

Thank you for your interest in contributing to BRepAX.

## Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/lv416e/brepax.git
   cd brepax
   ```

2. Install dependencies:

   ```bash
   uv sync --extra dev
   ```

3. Install pre-commit hooks:

   ```bash
   pre-commit install
   pre-commit install --hook-type commit-msg
   ```

## Development Workflow

- Run tests: `uv run pytest`
- Run linter: `uv run ruff check src/ tests/`
- Format code: `uv run ruff format src/ tests/`
- Type check: `uv run mypy src/`
- Run all via nox: `uv run nox`

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/). Examples:

- `feat: add cylinder primitive`
- `fix: correct SDF sign at boundary`
- `docs: add vmap tutorial`
- `test: add property tests for boolean associativity`

## Pull Requests

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure all checks pass
4. Submit a PR with a clear description

## Architectural Decisions

If your change involves a significant design choice, add an ADR in `docs/architecture/adr/` following the existing template.
