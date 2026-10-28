# Hybrid Optimization Strategy

Combining smoothing and relation-dispatched gradients is a research
proposal, not an implemented optimizer. The experimental `hybrid_optimize`
entry point always raises `NotImplementedError`.
[ADR-0011](../architecture/adr/0011-hybrid-strategy.md) records the decision
to keep scheduling at the optimizer level.

## Why Combine Methods?

An exact static objective can be locally constant in a design direction.
Moving disjoint disks without changing radii leaves union area unchanged.
No exact local derivative supplies a nonzero translation signal there.

Smoothing changes the objective and may supply exploration information.
Method C avoids some smoothing in simple relation regions but uses a grid
STE in partial overlap. Switching to it does not imply exact refinement.
See [Stratification Theory](stratification_theory.md#current-brepax-implementation)
for the implemented rules.

The proposed schedule uses smoothing for exploration and changes methods or
resolution for refinement. Its usefulness depends on objective, geometry,
initialization, and budget. Gradients can remain too small, cancel by
symmetry, or lead to a different optimum.

## What Would Validate a Schedule?

Compare with each individual method under matched initializations and
budgets, including temperature and resolution choices. Independently
evaluate the final unsmoothed objective and constraints. Report failed
starts, contact events, parameter error, and cost, not only smoothed loss.

The former 3000-times-precision claim does not describe the current
implementation; see the [historical evaluation](concept_proof_evaluation.md).
An analogy with rendering does not establish this schedule's effectiveness.

TOI cannot supply nonzero exact static derivatives on a true plateau.
Using future contact requires a specified look-ahead objective, trajectory
derivative, or surrogate. The [TOI discussion](toi_derivation.md) separates
the valid event-time calculation from the invalid former correction.
