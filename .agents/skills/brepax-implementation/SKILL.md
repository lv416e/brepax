---
name: brepax-implementation
description: Plan and implement non-trivial BRepAX code changes or refactors, including geometry kernels, OCCT extraction, metrics, and benchmark CLIs. Use before editing those paths.
---

# BRepAX implementation

Read the root `AGENTS.md` for policy. Use this workflow to choose maintainable
boundaries for the actual change, without reorganizing unrelated code.

## Establish the contract

Inspect the entry point, its callers, existing tests, and relevant ADRs. State
the observable behavior being changed or preserved. Identify host preparation,
traced computation, and output side effects. For an experiment, name the
hypothesis and the measurement that could reject it.

Follow the call path involved in the task; do not audit the whole repository
or write an implementation plan for a small edit. Existing large modules are
context to understand, not a reason for an unsolicited rewrite.

## Choose responsibility boundaries

| Responsibility | Existing home | Design criterion |
|---|---|---|
| OCCT bindings and API isolation | `src/brepax/_occt/` | Keep backend-specific calls behind the existing boundary. |
| File reading and host extraction | `io/` and B-Rep extraction in `brep/` | Prepare geometry and metadata outside the differentiated function. |
| Parametric evaluation, distances, Boolean rules | `primitives/`, `nurbs/`, `boolean/`, numerical parts of `brep/` | Accept explicit arrays/model data and return numerical results without file writes or logging. |
| Geometric objectives | `metrics/` | Reduce reusable numerical data; avoid unnecessary repeated extraction or triangulation within a task. |
| Experiment orchestration and reports | `benchmarks/` | Compose computations, retain per-case outcomes, and render/write results at the boundary. |
| Unvalidated optimizers and physics/application prototypes | `src/brepax/experimental/` | Keep experimental contracts explicit and promotion into supported paths deliberate. |

Resolve directory paths in this table relative to `src/brepax/` unless an
explicit repository path is shown. Use existing nearby modules before adding
a new package.

Extract a function when it names an operation, isolates a side effect, or
allows independent reasoning. Split a module when responsibilities change for
different reasons or expose a stable data boundary. Keep short, cohesive code
together; do not add a wrapper just to make its caller shorter.

For scripts, separate argument parsing, the callable operation, and rendering
when their complexity warrants it. `main()` should orchestrate them. Shared
benchmark stages belong in a cohesive internal module rather than imports
from another CLI's private helpers. Put shared code in the library only when
it belongs to the library's supported responsibility.

Prefer named typed results for substantial cross-module data. Keep their
fields tied to consumers; do not add a configurable pipeline or plugin system
without an actual requirement. Preserve existing numerical semantics during
extraction, including dtype, ordering, tolerances, and failure behavior.

## Preserve numerical meaning

- Identify which inputs are differentiable, which topology/UV data are frozen,
  and which values are static under JIT. Preserve parameter-to-face ownership
  and linked boundary motion when moving code.
- Keep JAX functions pure and compilation signatures stable. Do not move
  NumPy/OCCT operations into traced evaluation or accidentally introduce
  device-to-host synchronization.
- Treat tolerances as part of the numerical contract: units, scale, precision,
  and degeneracies matter. Explain justified constants without turning every
  coefficient into configuration.
- Separate structural refactors from numerical-method changes. Benchmark
  optimizations against a measured baseline, including compilation/setup
  costs where relevant.
- Keep unsupported results and failed cases visible. Broad exception handling
  can be appropriate at an experiment's case boundary if it records the
  failure; it should not conceal a kernel defect.

## Validate the changed contract

Use `brepax-validation` to select independent evidence and relevant checks.
Review the final diff for accidental API changes, dependency growth, shifted
responsibilities, and unrelated cleanup. For exploratory work, retain working
results outside the public checkout. Update public documentation only when a
result changes a maintained contract or an approved technical decision.
