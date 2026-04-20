# ADR-0017: Triangulation cold / subsequent / warm performance regions

## Status

Accepted

## Context

The original performance target for BSpline-heavy triangulation was stated as
"CTC-02 (664 faces, 34 BSpline, 247K triangles) < 10 s cold". After the
triangulation refactor in commits `8931f24` (JIT sharing per surface signature),
`9d799f3` (face dispatch batching), and `725fbf9` (persistent compilation cache
public API), the measured CTC-02 timings are:

| Scenario | Time |
|---|---|
| Fresh process, cache off | ~12 s |
| Fresh process, cache on, first-ever run | ~14 s (cache write cost) |
| Fresh process, cache on, second run onward | ~6 s |
| Repeated call in the same Python process | ~0.8 s |
| `divergence_volume` + `jax.grad` on an already-triangulated mesh | < 1 s |

A single "cold" number hides two things.  First, the first-ever run always pays
the full XLA compile cost.  Second, the steady-state cold under a populated
disk cache is a factor of two faster.  Reporting one number either
over-promises (if we quote 6 s) or under-promises (if we quote 14 s) on every
user's actual first interaction.

Practical CAD optimization in BRepAX stays in the warm regime almost
exclusively.  A typical pipeline loads one shape, calls `triangulate_shape`
once (or uses `extract_mesh_topology` + `evaluate_mesh` for parametric sweeps),
and runs hundreds of gradient steps in the same Python process.  Cold latency
matters only for batch jobs, interactive exploration, and the first run after
installing the library.

## Decision

Measure and report three distinct performance regions rather than one gate:

1. **First-ever cold.** Fresh Python process, no disk cache on the machine.
   Lower bound for a user's very first call on BRepAX.
2. **Subsequent cold.** Fresh Python process with a populated disk cache
   from an earlier run (opt-in via `brepax.enable_compilation_cache`).
   Covers CI jobs, batch scripts, and interactive shells after the first run.
3. **In-process warm.** Repeated calls within the same Python process, hitting
   JAX's in-memory JIT cache.  Covers optimization loops and Jupyter sessions.

CTC-02 targets under the new framing:

| Region | Target | Measured |
|---|---|---|
| First-ever cold | as close to 10 s as is feasible | 12–14 s |
| Subsequent cold | < 10 s | ~6 s |
| In-process warm | < 2 s | ~0.8 s |

The original 10 s gate is not met in region 1 as strictly stated, and is met
with margin in regions 2 and 3.  The remaining first-ever cost breaks down
approximately as 3–5 s of OCCT Python face traversal, 5–8 s of per-signature
XLA compilation, and ≈1 s of small overhead (STEP parse, mesh concat).
Further tightening requires attacking the OCCT traversal path or collapsing
BSpline signatures into a padded single-compile dispatch; neither is currently
planned.

## Consequences

- Benchmarks (`tests/benchmarks/test_ctc02_baseline.py`,
  `tests/benchmarks/test_multi_model_baseline.py`) print cold and warm columns
  side by side so results can be interpreted under the new framing.  A future
  enhancement is N-run median to tighten the single-shot variance.
- Documentation for `enable_compilation_cache` explains that it affects
  subsequent cold only; the first-ever cold on a new machine is unchanged.
- The paper narrative for triangulation performance reports three numbers per
  model instead of one.  This is an honest framing that avoids having to
  explain why "cold" fluctuates between invocations.
- First-ever cold becomes a separate optimization target if ever needed.  The
  remaining levers (OCCT face traversal batching, BSpline padding to a single
  compile, ahead-of-time warmup on import) are catalogued for a future
  milestone but not scheduled.
- Users who need predictable first-call latency should call
  `brepax.enable_compilation_cache()` before their first BRepAX call.
