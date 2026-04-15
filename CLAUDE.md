<!-- markdownlint-disable MD013 -->
# Project Guidelines

## Language and Style

**English** -- code, comments, docstrings, commit messages, log messages.
**Japanese** -- PR titles/bodies, issue titles/bodies, markdown documentation, user-facing communication.
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

# Pre-commit hooks
pre-commit run --all-files

# Build documentation
uv run mkdocs serve
```

### PR Checklist

1. All tests pass (`uv run pytest`)
2. Type checking passes (`uv run mypy src/`)
3. Linting passes (`uv run ruff check src/ tests/`)
4. New public API has docstrings with Examples section
5. Commit messages follow Conventional Commits
6. ADR added if an architectural decision was made

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

Current scope (Phase 1):

- 3D primitives in order: Sphere, Cylinder, Plane, Cone, Torus, Box
- Union, intersection, and subtract Boolean operations via Method (A) smoothing and Method (C) stratum-aware
- Hybrid optimizer API skeleton in `experimental/optimizers/`
- Midpoint milestone: Cylinder + Plane drilling demonstration

Completed (concept proof):

- 2D Disk primitive with Method (A) and Method (C) gradient comparison
- Analytical ground truth for two-disk configurations
- Gradient accuracy benchmarking (Gate 1 PASS, Gate 2a PASS, Gate 3a PASS)

Not yet in scope (defer to future work):

- Method (B) TOI correction implementation (deferred to hybrid optimizer context, ADR-0009)
- Topology data structures (half-edge mesh)
- Persistent homology integration
- B-Rep bridge layer (OCP-to-JAX conversion)
- C++ extensions

If scope expansion is needed, add an ADR documenting the decision. Do not expand scope without explicit approval.

## Things to Avoid

- **No C++ extensions** until benchmarks show a specific bottleneck (pure JAX for now)
- **No code copying** from existing OSS (JAX-FEM, Manifold, etc.) -- reference only
- **No private API exposure** -- `_internal/` must never be imported from public modules
- **No dynamic shapes** without explicit justification -- use padding strategies instead
- **No backward compatibility code** -- this project is in alpha; prefer clean breaks
- **OCCT dependency is core identity** -- do not propose removing cadquery-ocp-novtk; changes to OCCT strategy require an ADR
- **No direct OCP imports** outside `brepax/_occt/` -- all OCCT access goes through `brepax._occt.backend` (see ADR-0008)

## Development Philosophy

- **Understand before changing**: Read existing code and docs before modifying anything.
- **Research first**: Investigate modern best practices before implementing non-trivial decisions.
- **No over-abstraction**: Three similar lines are better than a premature abstraction.
- **No hardcoding**: Magic numbers and defaults must be defined in constants or configuration.

## Code Quality Checks (mandatory before commit)

```bash
uv run ruff format src/ tests/
uv run ruff check src/ tests/ --fix
uv run mypy src/
uv run pytest
```

## Reference Materials

- JAX documentation: https://docs.jax.dev
- Equinox documentation: https://docs.kidger.site/equinox
- jaxtyping documentation: https://docs.kidger.site/jaxtyping
- Diataxis framework: https://diataxis.fr
- ADR pattern: https://github.com/joelparkerhenderson/architecture-decision-record
