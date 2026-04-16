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

3. Install git hooks:

   ```bash
   lefthook install
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

## Release Process

### TestPyPI dry run

Before publishing to PyPI, verify the package on TestPyPI:

- **Manual trigger**: Run the "Publish to TestPyPI" workflow via `workflow_dispatch` on GitHub Actions.
- **Pre-release tag**: Push a tag matching `v*.*.*-rc*` or `v*.*.*-beta*` (e.g., `v0.2.0-rc1`) to trigger the workflow automatically.

Install and verify the TestPyPI package:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple brepax==<version>
```

The `--extra-index-url` flag ensures runtime dependencies (JAX, equinox, etc.) resolve from the real PyPI.

### Production release

Production releases are managed by [release-please](https://github.com/googleapis/release-please):

1. Conventional Commits on `main` cause release-please to auto-create a release PR.
2. The release PR stays open until the maintainer decides to release.
3. Merging the release PR creates a semver tag (e.g., `v0.2.0`).
4. The tag push triggers the publish workflow, uploading to PyPI via Trusted Publishing.

**Important**: Release-please PRs must only be merged on the maintainer's explicit decision. They represent a strategic release choice, not a routine code change.

### Trusted Publisher setup

Both TestPyPI and PyPI use [Trusted Publishers](https://docs.pypi.org/trusted-publishers/) (OIDC) for authentication. No API tokens are stored in secrets. Each environment (`testpypi` and `pypi`) must be configured as a Trusted Publisher on the respective PyPI project settings page, specifying this repository and the corresponding workflow file.
