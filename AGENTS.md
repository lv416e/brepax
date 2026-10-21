# BRepAX agent instructions

## Canonical instructions

Edit `AGENTS.md` and `.agents/skills/` as the source of truth.
`CLAUDE.md` links to `AGENTS.md`; `.claude/skills` links to
`../.agents/skills`. Keep these relative symlinks instead of maintaining copies.
Tool-specific permissions and local state remain outside these shared rules.

Read the relevant repository skill before its workflow:

| Task | Skill |
|---|---|
| Non-trivial implementation or refactoring | [brepax-implementation](.agents/skills/brepax-implementation/SKILL.md) |
| Selecting tests or validating a code change | [brepax-validation](.agents/skills/brepax-validation/SKILL.md) |
| Committing, pushing, preparing a PR, or an authorized release | [brepax-pull-request](.agents/skills/brepax-pull-request/SKILL.md) |

## Project and scope

BRepAX develops JAX-native differentiable CAD Boolean operations and
STEP/B-Rep evaluation. It is not a general CAD kernel, mesh library, or physics
simulator. Distinguish implemented behavior from research hypotheses and from
what a selected benchmark establishes.

- Library code lives in `src/brepax/`; numerical experiments and application
  prototypes live in `src/brepax/experimental/` or `benchmarks/` as appropriate.
- All OCCT access in library code goes through `brepax._occt.backend` and
  `brepax._occt.types`. No direct OCP imports outside `src/brepax/_occt/`.
- Retain `cadquery-ocp-novtk` and the OCCT abstraction boundary. OCCT strategy
  changes require explicit approval and an ADR.
- Confirm scope before architectural changes, public API changes, or a new
  major capability. Record approved significant decisions in
  `docs/architecture/adr/`. Routine private function extraction needs no ADR.
- Do not add production dependencies without briefly justifying benefits,
  maintenance costs, and alternatives. No C++ extensions without a measured
  bottleneck and an approved scope change.
- Use current code, tests, and accepted ADRs to establish implementation state;
  avoid hardcoded feature counts or treating historical proposals as contracts.

## Implementation style

- Inspect the relevant code, callers, tests, and ADRs before non-trivial edits.
  State the intended scope and key assumptions in 1-3 sentences.
- Separate file I/O, OCCT extraction, numerical kernels, orchestration, and
  presentation when they have different responsibilities or side effects.
  Keep CLIs thin and reusable computations independent of CLI arguments.
- Extract functions around named operations and modules around cohesive
  responsibilities. Use change reasons, data flow, and testability to decide
  boundaries; line counts alone are not a reason to split code.
- Prefer explicit parameters and return values to hidden mutable state.
  Use a small typed record when a cross-module result needs one; avoid growing
  positional tuples, unstructured dictionaries, and generic context objects.
- Avoid one-line forwarding layers, speculative factories, catch-all `utils`
  modules, and abstractions with no demonstrated second use. Remove redundant
  work when sharing a real computation; do not create a framework to deduplicate
  superficially similar code.
- Preserve behavior in refactors. Separate intentional numerical or API changes
  from structural edits. This project is alpha: prefer an approved clean break
  to a speculative compatibility layer, and identify affected callers.
- Catch errors where the caller can recover or report them. In experiments,
  preserve failures and unsupported cases explicitly instead of silently
  substituting a plausible result.

## Python and documentation

- All repository content is English; no emojis in code, comments, or docs.
- `ruff.toml`, `pyproject.toml`, and existing tooling define formatting, lint,
  typing, and test configuration. Do not duplicate or weaken those settings
  to satisfy a change.
- Keep imports at the top, grouped stdlib / third-party / local. Local imports
  are reserved for circular imports, heavy lazy loads, optional dependencies,
  or backend selection; explain the reason. Use `TYPE_CHECKING` for type-only
  circular imports.
- Annotate public APIs. Document behavior, shapes, units, failure modes, and
  numerical assumptions where relevant; include a useful example for a new
  non-trivial public API.
- Comments explain intent, numerical choices, or constraints. Avoid restating
  the code, commented-out implementations, debug prints, personal notes, and
  development-stage markers such as `WIP` or `TODO Phase 3`.
- Keep docs under the existing Diataxis structure. Link to canonical contracts
  instead of copying them. This repository is public: keep internal work plans,
  restart notes, audit notes, and progress logs outside the checkout unless
  the user explicitly requests publication. Preserve useful working evidence
  in local notes, and put public change-specific rationale in the PR.
- Update an existing relevant document before adding one. Raw measurements
  belong in run artifacts when needed. Routine changes need no documentation
  update unless they change a maintained public contract.
- Add a document only for a distinct, durable reader need that existing docs
  cannot serve. ADRs record approved significant decisions, not routine
  refactors or each experiment. Consolidate duplicated guidance at its source.

## Geometry and JAX

- Separate host-side OCCT/I/O preparation from JAX-traceable evaluation.
  Do not require file readers or OCCT wrappers themselves to support JAX
  transformations. Numerical APIs intended for tracing must support their
  documented `jit`, `vmap`, and differentiation use cases.
- Use `equinox.Module` for differentiable geometric model types and `jaxtyping`
  annotations for array shapes. Ordinary host metadata need not be an Equinox
  module.
- Keep traced shapes static; use padding and masks for variable-size data
  (ADR-0005). Avoid host conversion, side effects, and fresh JIT closures in
  numerical hot paths.
- Make units, dtype/precision, tolerance meaning, scale dependence, orientation,
  and degenerate-input behavior explicit. Preserve reproducibility and document
  any nondeterminism relevant to an experiment.
- State applicable invariance/equivariance laws for values and gradients.
  Transform external directions along with geometry when the contract requires
  it. Distinguish exact geometric laws from discretization error, and scale
  dimensional tolerances and smoothing parameters consistently with units.
- Distinguish derivatives of a frozen discretization from derivatives of
  rebuilt CAD geometry. Topology, trim, and shared-boundary changes require
  their own validity checks.
- Choose differentiation methods for the mathematical contract. Use custom
  VJP/JVP rules only with a stated reason and independent derivative evidence;
  do not introduce a solver dependency solely to follow a preferred pattern.

## Tests and verification

- Add tests for meaningful behavior, mathematical invariants, or reproduced
  failures. Do not require unit tests and property tests simultaneously for
  every feature. Use each where it supplies distinct evidence.
- Prefer the public contract and an independent oracle to private helper
  structure. Avoid tests that mirror the implementation, mock an internal
  call sequence, or merely assert finiteness when accuracy or convergence is
  the actual claim.
- For a bug fix, reproduce the failure before changing behavior when practical.
  For a refactor, retain a behavior baseline and use existing tests before
  adding new ones. Documentation-only or mechanical changes need no new unit
  tests unless they change an executable contract.
- Run the smallest relevant checks after editing. Report commands, results,
  skipped checks, and material limitations. Keep experimental observations
  separate from production guarantees.
- Before **every commit and push**, all four existing gates must pass:

  ```bash
  uv run ruff format src/ tests/
  uv run ruff check src/ tests/ --fix
  uv run mypy src/
  uv run pytest -m "not slow"
  ```

  Include changed benchmark or other Python paths in lint/format checks too.
  Inspect auto-fixes for unrelated edits. The validation skill explains
  focused checks and evidence selection; the full gates still apply before
  committing or pushing.

## Work modes and repository operations

Production-oriented work should make the smallest coherent change that solves
the problem. Exploratory work may replace abstractions, use temporary tooling,
or build disposable implementations within the agreed experiment. Keep that
work isolated where practical; state what was validated and what production
use would still require.

All changes go through PRs; do not push directly to `main`. Use Conventional
Commits and the repository PR template. Run the pull-request skill for its
checks and release procedure.

Never merge a release-please PR without the maintainer's explicit instruction.
Do not change branch protection, secrets, or deploy keys, force-push, rewrite
shared history, or delete another contributor's work without explicit
authorization. Never use an admin bypass of branch protection.
