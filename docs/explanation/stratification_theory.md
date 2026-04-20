# Stratification Theory

How Whitney stratification applies to CAD Boolean operations,
and why gradient discontinuities at topological boundaries are a
geometric fact rather than a numerical artifact.

## Configuration Space of Two Primitives

Consider two primitives A and B parameterized by design variables
(radii, centers, axes). As parameters vary, the topological
relationship between A and B changes:

| Stratum | Label | Relationship | Example (two spheres) |
|---|---|---|---|
| 0 | Disjoint | No overlap | d > r1 + r2 |
| 1 | Intersecting | Partial overlap | \|r1 - r2\| < d < r1 + r2 |
| 2 | A contained in B | A entirely inside B | d + r1 <= r2 |
| 3 | B contained in A | B entirely inside A | d + r2 <= r1 |

These four regions partition the configuration space into a
**Whitney stratification**: a decomposition into smooth manifolds
(strata) satisfying Whitney's regularity conditions (a) and (b)
at their boundaries.

## Gradient Structure Per Stratum

Each stratum has a distinct analytical gradient formula for Boolean
volume. For union:

| Stratum | V_union formula | d(V_union)/d(p) |
|---|---|---|
| 0 (disjoint) | V_a + V_b | d(V_a)/d(p) + d(V_b)/d(p) |
| 1 (intersecting) | V_a + V_b - V_intersection | Requires integration over intersection boundary |
| 2 (A in B) | V_b | d(V_b)/d(p) |
| 3 (B in A) | V_a | d(V_a)/d(p) |

In strata 0, 2, and 3, the gradient is exact and analytical because
the volume is a simple function of individual primitive volumes.
In stratum 1, the intersection boundary curve introduces a
non-trivial integration that BRepAX approximates via the
straight-through estimator (STE).

## Gradient Discontinuity at Boundaries

At stratum boundaries (tangent configurations), the gradient is
generically discontinuous. This is not a bug but a consequence
of the stratification structure:

**External tangency** (d = r1 + r2): the gradient of union area/volume
is actually continuous here because the intersection formula
reduces to the disjoint formula as overlap vanishes. The gradient
jump is second-order zero, as derived in the
[TOI derivation](toi_derivation.md#correction-term-at-external-tangent).

**Internal tangency** (d = |r1 - r2|): the gradient is genuinely
discontinuous. The contained-stratum gradient depends only on the
outer primitive, while the intersecting-stratum gradient depends on
both. This is a first-order jump that cannot be smoothed away.

## BRepAX Implementation

BRepAX implements stratum-dispatched gradients via `jax.custom_vjp`:

1. **Forward pass**: exact SDF Boolean (min/max) + Heaviside indicator
   on a grid. The stratum label is detected via grid-based SDF
   evaluation and saved as a residual.

2. **Backward pass**: `jnp.where` dispatch selects the gradient
   formula based on the saved label. This is branchless and
   compatible with `jax.vmap`.

The boundary convention (ADR-0004) uses the intersecting-stratum
gradient at exact tangency, following the principle that the
generic (highest-dimensional) stratum provides the most informative
gradient signal.

## Convergence Properties

Empirical measurement on sphere-sphere and cylinder-sphere pairs
shows:

- **Strata 0, 2, 3**: exact analytical gradient (zero error)
- **Stratum 1 interior**: STE gradient error < 1% (sphere) and
  < 10% (cylinder) at resolution 128
- **Stratum 1 near-tangent**: error increases to 25-542% as the
  sigmoid kernel width exceeds the intersection feature size

The convergence rate is between O(h) and O(h^2) in the interior,
consistent with grid-based integration of a sigmoid-smoothed
discontinuous integrand. Near tangency, the rate degrades because
the effective feature size shrinks faster than the grid refines.

Despite near-tangent error, optimization converges to correct
optima because the gradient sign is always correct and the bias
direction (volume overestimation) is consistent.

## Comparison

No existing differentiable CSG system provides exact gradients in
any stratum. BRepAX's 3/4 exact + 1/4 characterized is a unique
positioning:

| System | Exact strata | Gradient method |
|---|---|---|
| BRepAX | 3/4 | Analytical dispatch + STE |
| DiffCSG | 0/4 | Goldfeather rasterization + edge AA |
| Fuzzy Boolean | 0/4 | t-norm/t-conorm smoothing |
| TreeTOp | 0/4 | Continuous relaxation |

## References

- Whitney, H. (1965). Tangents to an analytic variety.
- Goresky, M. & MacPherson, R. (1988). Stratified Morse Theory.
- Guest, J.K. et al. (2004). Achieving minimum length scale in
  topology optimization using nodal design variables and projection
  functions.
- Bengio, Y. et al. (2013). Estimating or propagating gradients
  through stochastic neurons for conditional computation.
