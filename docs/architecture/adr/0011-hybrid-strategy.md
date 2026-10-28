# ADR-0011: Stratum-Aware Locality and Hybrid Optimization Strategy

## Status

Accepted as an optimizer-level design direction; implementation pending.

Scope and empirical claims corrected on 2026-10-28. The experimental
`hybrid_optimize` function remains a stub.

## Context

For an objective locally independent of a design variable, that derivative
is zero. Disjoint disks have union area $\pi r_1^2+\pi r_2^2$, independent
of centers; gradient descent on this objective cannot initiate center motion.

Smoothing can add exploration information by changing the objective. It is
one possible strategy, not the only way to explore, and guarantees neither
nonzero nor useful gradients everywhere.

## Decision

Keep exploration/refinement scheduling at the optimizer level. Callers
select Boolean methods explicitly; the Boolean API does not silently change
methods based on optimization history.

The historical proposal combines smoothing for exploration with Method C
for refinement. Its usefulness remains an empirical question. The
[hybrid strategy](../../explanation/hybrid_optimization_strategy.md) defines
the needed comparison; it does not promise exact convergence.

## Corrections to the Historical Rationale

- The original $7\times10^{-6}$ radius error and 3000-times precision claim
  are superseded by the maintained grid-based demo, not current evidence
  of exact intersecting gradients.
- Intersection volume grows continuously from zero at ordinary first
  contact. Ordinary internal and external disk/sphere area/volume contacts
  are $C^1$ under the assumptions in
  [Stratification Theory](../../explanation/stratification_theory.md).
- Selected successful trajectories do not prove that gradient sign is
  always correct or frozen-mesh derivatives suffice for moving trims and
  changing topology.
- Rendering offers an analogy, not an isomorphism or proof of effectiveness.
- Event-time sensitivity cannot transfer a future gradient into an exact
  current static derivative. Method B remains deferred under
  [ADR-0009](0009-method-b-deferred.md).

## Consequences

The schedule/result records describe an experimental interface, not a
working framework. Validate a concrete schedule against individual methods
and an independent unsmoothed objective before claiming improvement.
