# ADR-0009: Method (B) TOI Correction Deferred from Concept Proof

## Status

Accepted

## Context

The original concept proof plan called for three methods: (A) smoothing, (B) TOI correction, (C) stratum-aware tracking, compared head-to-head on the disk-disk union area problem.

During the mathematical derivation of Method (B) (see `docs/explanation/toi_derivation.md`), a key finding emerged: **the gradient jump at external tangent is second-order**. At $d = r_1 + r_2$, the area gradient is continuous -- the intersecting formula reduces exactly to the disjoint formula. This means Method (B)'s TOI correction term vanishes to first order at external tangent.

At internal tangent ($d = |r_1 - r_2|$), the gradient jump is genuine first-order, but Method (C) also handles this case via its stratum-aware formulas.

Separately, the boundary proximity benchmark revealed that Method (A)'s primary failure mode is **smoothing bias** (systematic error from temperature parameters), not gradient discontinuity. Method (C) eliminates smoothing bias entirely by using exact formulas, making it the direct comparison partner for Method (A).

## Decision

Defer Method (B) implementation from the concept proof. Proceed with Method (A) vs Method (C) comparison only.

Rationale:

- Method (B) provides no benefit at external tangent (correction is first-order zero)
- At internal tangent, Method (C) is equally effective and has a cleaner implementation (no root-finding for TOI)
- Reducing from 3 methods to 2 shrinks the benchmark matrix from 9 cells to 6 cells
- The concept proof hypothesis ("stratum-aware beats smoothing") is testable with 2 methods

## What is preserved

- `docs/explanation/toi_derivation.md`: Complete mathematical derivation, including the second-order finding
- `src/brepax/boolean/toi.py`: Skeleton module, ready for future implementation
- Contact dynamics correspondence table in the derivation doc

## Re-evaluation conditions

Method (B) should be reconsidered if:

- 3D primitive Boolean operations produce higher-order boundary transitions where TOI correction becomes non-trivial at first order
- Optimization trajectories require explicit boundary-crossing awareness (e.g., path-dependent stratum tracking)
- A use case emerges where Method (C) is insufficient but Method (B)'s root-finding approach adds value

## Empirical Validation

3D sphere-sphere and cylinder-sphere gradient benchmarks confirmed
the deferral decision with an additional finding:

**Method B does not address the primary Stratum 1 gradient error.**
Forced-label experiments (bypassing stratum detection, directly
computing STE gradient with correct stratum=intersecting) showed
that STE accuracy at near-tangent configurations degrades to
25-542% error regardless of detection quality. The root cause is
the sigmoid kernel width exceeding the intersection feature size,
not stratum misclassification at the boundary.

Method B corrects the gradient *jump* at stratum transitions.
The observed error is a stratum-*internal* integration problem
that Method B cannot address.

Interior STE accuracy is < 1% (sphere-sphere) and < 10%
(cylinder-sphere) at resolution 128, and optimization converges
despite near-tangent error.

See `tests/benchmarks/test_gradient_accuracy_3d.py` for data.

## Consequences

- Concept proof benchmark compares Method (A) vs Method (C) only
- Gate criteria apply to this 2-method comparison
- `boolean/toi.py` remains as skeleton; `NotImplementedError` in `union_area()` dispatch
- The TOI derivation document serves as architectural knowledge for future extensions
