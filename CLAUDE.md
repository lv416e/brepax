<!-- markdownlint-disable MD013 -->
# Project Guidelines

## Language and Style

**English** -- all repository content: code, comments, docstrings, commit messages, log messages,
PR titles/bodies, issue titles/bodies, markdown documentation.
No emojis in code, comments, or documentation.

## Project Mission

BRepAX is a JAX-native library providing a differentiable rasterizer for CAD Boolean operations. It translates contact dynamics formulations from differentiable physics into the CAD domain, enabling gradient-based optimization through B-Rep topology changes. BRepAX is NOT a general CAD kernel, mesh library, or physics simulator.

## Technical Thesis

The gradient discontinuities at topological boundaries in CAD Boolean operations (where edges/faces appear or disappear) are mathematically isomorphic to contact events in differentiable physics simulation. By mapping LCP/convex/compliant/PBD contact formulations to stratum tracking / boundary smoothing / event-based correction / penalty strategies, we build a differentiable B-Rep kernel without inventing new theory.

## Comments

Comments explain **why**, not what or how. Keep them minimal and concise.

Prohibited:

- Development stage markers: "Phase 1", "Step 2", "TODO Phase 3", "WIP", "FIXME"
- Commented-out code or debug print statements
- Restated logic ("loop through users", "compute distance")
- Personal notes: "will fix later", "temp solution"

## Repository Conventions

- **src layout**: All library code lives under `src/brepax/`
- **Diataxis documentation**: `docs/` follows tutorials / how-to / reference / explanation separation
- **ADR pattern**: All significant architectural decisions are recorded in `docs/architecture/adr/`
- **Test-first development**: New features must include property tests and unit tests simultaneously
- **Conventional Commits**: All commit messages follow the Conventional Commits specification

## Development Workflow

```bash
# Install dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run linter
uv run ruff check src/ tests/

# Format code
uv run ruff format src/ tests/

# Type checking
uv run mypy src/

# Run all checks via nox
uv run nox -s test
uv run nox -s lint
uv run nox -s typecheck

# Install git hooks (lefthook, managed via mise)
lefthook install

# Build documentation
uv run mkdocs serve
```

### Branch and PR Workflow

All changes go through pull requests. Direct pushes to `main` are blocked.

1. Create a feature branch: `git checkout -b feat/my-feature`
2. Implement and commit (with quality checks)
3. Push: `git push origin feat/my-feature`
4. Create PR: `gh pr create --title "feat: ..." --body "..."`
5. Wait for CI (ci-gate must pass)
6. Wait for maintainer review and merge
7. After merge: `git checkout main && git pull`

### Commit and Merge Conventions

- **Conventional Commits**: `feat:`, `fix:`, `chore:`, `refactor:`, `docs:`, `ci:`, `test:`
- **Squash merge only**: All PRs are squash-merged to keep `main` history clean
- **Squash commit message**: PR title + `(#PR_number)` -- e.g. `feat(io): add STEP reader (#7)`
- **PR title = commit message**: PR title must be a valid Conventional Commits message (this becomes the squash commit on `main`)
- **Co-Authored-By**: When Claude contributed, include `Co-Authored-By: Claude <noreply@anthropic.com>` in the squash commit body
- **No merge comments**: Do not leave "LGTM" or similar comments when self-merging
- **PR body template** (must follow this format):

  ```markdown
  ## Summary
  - Concise bullet points describing changes (3-5 items)

  ## Verification
  - `uv run pytest` all passed
  - `uv run mypy src/` 0 errors
  - Additional manual verification steps if any

  ## Related
  - Related issues or ADRs if applicable (omit section if none)
  ```

### PR Checklist

Before creating a PR, verify locally:

1. All tests pass (`uv run pytest`)
2. Lint clean (`uv run ruff check src/ tests/`)
3. Format applied (`uv run ruff format src/ tests/`)
4. Mypy passes (`uv run mypy src/`)
5. New public API has docstrings with Examples section
6. ADR added if an architectural decision was made

Before asking for merge, verify on GitHub:

7. CI is green (ci-gate passed)
8. No merge conflicts with main
9. PR title is a valid Conventional Commit
10. PR body follows the template

### When to Pause

Do NOT proceed autonomously in the following cases:

1. **Release-please PRs**: Never merge automatically. Wait for maintainer's explicit
   instruction before merging `chore(main): release x.y.z` PRs.
2. **Strategic decisions**: When a task involves architectural decisions (e.g.,
   dropping a dependency, changing public API), pause and ask before implementing.
3. **Branch protection changes**: Do not modify repository settings (protection
   rules, secrets, deploy keys) without explicit maintainer approval.
4. **Destructive operations**: Force-push, rebase of shared branches, history
   rewriting require explicit maintainer approval.
5. **Admin bypass**: Never use admin privileges to bypass branch protection
   rules, even if technically possible.

### Release Process

Releases are managed by release-please. The workflow:

1. Conventional Commits on `main` accumulate a changelog in a release-please PR
2. To override the version (e.g., minor bump), squash-merge a commit with the body
   set to **only** the git trailer: `Release-As: 0.2.0` (no prose -- squash merge
   replaces the body, so prose destroys the trailer)
3. Maintainer merges the release-please PR
4. release-please creates a GitHub Release and tag via the API
5. The API-created tag does **not** trigger `on: push: tags` workflows. Run the
   publish workflow manually: `gh workflow run publish.yaml -f tag=v0.2.0`
6. Verify on PyPI that the new version is available

## Imports

Follow PEP 8 grouping (stdlib, third-party, local). All imports at the top of the file.

Local imports are allowed only for: circular import avoidance, lazy loading of heavy dependencies,
optional dependency isolation, or plugin/backend switching. Add a comment explaining which applies.

For type hints causing circular imports, use `TYPE_CHECKING` guard with string annotations.

## JAX-Specific Patterns

- **Base class**: All geometric types inherit from `equinox.Module`
- **Type annotations**: All public APIs use `jaxtyping` annotations (`Float[Array, "..."]`)
- **Static shapes**: JAX requires static shapes for jit. Use padding + masking for variable-size structures (see ADR-0005)
- **Transformations**: Every public function must be compatible with `jit`, `vmap`, and `grad`. Verified by integration tests in `tests/integration/`
- **Custom differentiation**: Use `jax.custom_vjp` for boundary corrections, `jax.custom_jvp` for numerical stability. Prefer implicit differentiation via `optimistix` for root-finding operations
- **Runtime type checking**: `jaxtyping` + `beartype` are enabled during tests via pytest hook

## Scope Boundaries

Implemented:

- 8 primitives: Disk (2D), Sphere, Cylinder, Plane, Cone, Torus, Box, FiniteCylinder
- Union, intersection, and subtract Boolean operations via Method (A) smoothing and Method (C) stratum-aware
- Stratum-dispatched gradients: analytical exact in 3/4 strata for bounded pairs
- Hybrid optimizer API skeleton in `experimental/optimizers/`
- Mold direction optimizer demonstrator in `experimental/applications/`
- 2D Disk analytical ground truth + 3D Sphere analytical ground truth
- 7 example notebooks, 13 ADRs

Not yet in scope (defer to future work):

- Method (B) TOI correction implementation (deferred to hybrid optimizer context, ADR-0009)
- STEP file read / analytical primitive conversion
- Topology data structures (half-edge mesh)
- Persistent homology integration
- B-Rep bridge layer (OCP-to-JAX conversion)
- C++ extensions

If scope expansion is needed, add an ADR documenting the decision. Do not expand scope without explicit approval.

## Things to Avoid

- **No C++ extensions** until benchmarks show a specific bottleneck (pure JAX for now)
- **No code copying** from existing OSS (JAX-FEM, Manifold, etc.) -- reference only
- **No dynamic shapes** without explicit justification -- use padding strategies instead
- **No backward compatibility code** -- this project is in alpha; prefer clean breaks
- **OCCT dependency is core identity** -- do not propose removing cadquery-ocp-novtk; changes to OCCT strategy require an ADR
- **No direct OCP imports** outside `brepax/_occt/` -- all OCCT access goes through `brepax._occt.backend` (see ADR-0008)
- **Never merge release-please PRs without explicit owner instruction** -- release-please auto-creates changelog PRs; these contain a strategic release decision and must only be merged when the owner explicitly requests it. Admin bypass for these PRs is forbidden.

## Development Philosophy

- **Understand before changing**: Read existing code and docs before modifying anything.
- **Research first**: Investigate modern best practices before implementing non-trivial decisions.
- **No over-abstraction**: Three similar lines are better than a premature abstraction.
- **No hardcoding**: Magic numbers and defaults must be defined in constants or configuration.

## Code Quality Checks (mandatory before EVERY commit and push)

**NEVER commit or push without running ALL of these checks first.**
A CI failure after push means the check was skipped locally. No exceptions.

```bash
uv run ruff format src/ tests/
uv run ruff check src/ tests/ --fix
uv run mypy src/
uv run pytest -m "not slow"
```

All four must pass with zero errors before `git commit`. If any fails, fix it before committing.
Do not rely on CI to catch errors -- CI is a safety net, not the primary gate.

## Reference Materials

- JAX documentation: https://docs.jax.dev
- Equinox documentation: https://docs.kidger.site/equinox
- jaxtyping documentation: https://docs.kidger.site/jaxtyping
- Diataxis framework: https://diataxis.fr
- ADR pattern: https://github.com/joelparkerhenderson/architecture-decision-record
