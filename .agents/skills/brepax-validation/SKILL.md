---
name: brepax-validation
description: Select meaningful tests and verification for BRepAX code changes, refactors, geometry algorithms, and benchmark claims. Use when deciding what evidence a change needs and before handing it back.
---

# BRepAX validation

Start from the behavior or claim that could fail. Reuse existing coverage
when it already establishes the contract; select new tests by information
gained, not by the number of helpers, branches, or files changed.

## Select evidence

| Change or claim | Useful evidence |
|---|---|
| Behavioral bug fix | Small failing input before the fix, passing afterward; exercise the affected caller when helper-level coverage would miss the fault. |
| Structural refactor | Existing behavior tests plus a representative before/after result; add characterization only where relevant behavior is unprotected. |
| Geometric formula | Analytic result or independent OCCT reference with stated units, orientation, dtype, and tolerances. |
| General geometric invariant | Property tests over valid domains, with scale and degeneracy cases chosen for the actual mathematical contract. |
| Differentiation | Independent analytic derivatives or a finite-difference step sweep away from events; one-sided limits at nonsmooth events. |
| JAX transformation support | Exercise the actual numerical entry point under the documented `jit`, `vmap`, and `grad` uses. Host file readers are outside this contract. |
| Optimization improvement | Fixed initializations, true objective and constraint checks, convergence/parameter error, and comparable computation budgets. |
| Benchmark/reporting change | One representative fixture through computation and output; verify failures, unsupported cases, and the meaning of timing fields if affected. |
| Instructions, prose, or mechanical edits | Applicable schema, link, symlink, diff, format, or executable-example checks; no new unit test by default. |

Choose unit, property, and integration tests when each adds distinct evidence.
A feature need not add all three. Parameterize genuinely parallel cases;
avoid duplicate suites for equivalent execution paths.

## Reject weak tests

- A test should fail for a plausible incorrect behavior. Avoid assertions
  that reconstruct the implementation or inspect private call order.
- Finite values do not establish numerical accuracy, nonzero gradients,
  progress, or convergence. Assert the actual claim and allow valid zero
  derivatives when the mathematical objective is locally constant.
- A low smoothed loss does not establish a correct unsmoothed solution.
  Check an independent objective or known target.
- Do not mock the geometric computation being verified, add tests solely for
  coverage percentages, or preserve obsolete tests after their contract is
  deliberately removed. Existing coverage gates still apply.
- Do not create placeholder test modules or empty cases. A collected file
  with no tests supplies no evidence.

## Geometry-specific checks

Use the smallest set that distinguishes the intended behavior from plausible
failures. Include relevant degenerate or near-boundary cases; do not build an
exhaustive Cartesian product of unrelated options.

For finite differences, vary step size relative to model scale and avoid
cancellation. Compare the same function as AD first; a frozen-mesh derivative
and a remeshed OCCT derivative answer different questions. Record both when
the application needs the latter.

For shape changes, assess closure, shared edges, orientation, trim behavior,
and near-zero volume where relevant. State whether the checked tolerance is
absolute or relative. Fix seeds and inputs for reproducibility. Use x64 where
the numerical question requires it, and account for the test suite's x64
fixture when comparing CLI output.

For benchmarks, retain every attempted case and distinguish failures from
unsupported inputs. Synchronize JAX results before stopping timers; distinguish
extraction, compilation, forward, and backward costs. Do not interpret
availability counts or fixture-only accuracy as generalization evidence.

## Run checks proportional to the change

During iteration, run the focused test file, test node, or relevant lint/type
check. Expand when failures, changed shared behavior, or unresolved concerns
justify it. For reproducibility, use the existing lock; do not upgrade
dependencies just to run a test.

The root `AGENTS.md` defines the mandatory full gates before every commit and
push. If a failure appears pre-existing, report the evidence and resolve it
before committing rather than weakening the gate. Include changed Python
paths outside `src/` and `tests/` in lint and format checks.

For instruction/skill changes, validate frontmatter and local references,
resolve canonical symlinks, and check which files Git ignores. Check the
configuration rather than writing tests that assert exact instruction prose.

Report commands run, pass/fail results, what was not assessed, and any limits
on the conclusion. A passed smoke test is not a full-suite result.
