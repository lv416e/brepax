# ADR-0012: Primitive-Independent Stratum Detection

## Status

Accepted

## Context

The concept proof `_detect_stratum` uses center distance vs radius sum/difference, which is specific to Sphere x Sphere (and Disk x Disk). This function crashes for Cylinder, Plane, Cone, or any heterogeneous primitive pair because those primitives lack a "center" parameter.

As a result, subtract and intersect operations bypass stratum dispatch entirely, running straight-through estimator uniformly. This means these operations achieve Method (A)-equivalent precision rather than Method (C)'s stratum-aware precision.

Additionally, union operations with heterogeneous primitive pairs (e.g., Sphere x Cylinder) trigger a KeyError crash in `_detect_stratum`.

## Decision

Replace parameter-dependent stratum detection with SDF-based geometric relationship detection using grid sampling.

### Stratum definitions (primitive-independent)

Given two primitives A and B evaluated on a grid:

- **Disjoint**: `min(sdf_a) > 0` OR `min(sdf_b) > 0` within the other's interior does not exist. Operationally: no grid point satisfies both `sdf_a < 0` AND `sdf_b < 0`.
- **Contained (A in B)**: all grid points where `sdf_a < 0` also have `sdf_b < 0`.
- **Contained (B in A)**: all grid points where `sdf_b < 0` also have `sdf_a < 0`.
- **Intersecting**: some but not all interior points of A are interior to B (and vice versa).

### Implementation

Grid-based detection reuses the grid already computed for volume measurement. The detection adds negligible cost (min/max reductions over existing SDF evaluations).

### Trade-offs

- Detection precision is resolution-dependent (a very thin overlap at low resolution may be missed)
- Grid computation is required before stratum detection (cannot be done parameter-only)
- Analytical stratum detection for Sphere x Sphere is preserved in `analytical/` as ground truth

## Consequences

- All Boolean operations (union, subtract, intersect) can use stratum dispatch
- No per-pair parameter-dependent code needed for stratum detection
- New primitives (Cone, Torus, Box) get stratum dispatch automatically
- Method (C) precision advantage extends to heterogeneous primitive pairs
