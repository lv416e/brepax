# ADR-0022: Boundary derivatives for a circular cut crossing a plate edge

## Status

Accepted for an experimental scalar evaluator and reproducible benchmark.

## Context

Method C samples a hard indicator for its value and uses a smoothed
intersecting backward rule. Small overlaps can escape sampled detection.
The bounded editing API in ADR-0021 deliberately stays away from contact;
its fixed mesh cannot establish derivatives of topology-changing geometry.

A perpendicular circular through-cut in a rectangular plate is a bounded
case with curved boundaries, an actual hole-to-open-notch transition, and
independent volume references. Its volume is C1 at ordinary edge and corner
contacts. Positive-area planar contact, arbitrary STEP reconstruction, and
surface-area derivatives are different problems.

## Decision

Add `brepax.experimental.contact` for disk/rectangle intersection area and
the remaining extruded plate volume. Integrate the analytic circular arcs
and straight boundary segments using Green's theorem. Use a local gauge
and a small-segment series to resolve small overlap areas without subtracting
large boundary integrals. Factor axis-aligned chord expressions to avoid
the cancellation of a generic quadratic discriminant.

Supply a custom JVP derived from boundary velocity for the same continuous
measure. For the exposed circular arcs Gamma, center and radius variations
give `dA = integral_Gamma (dc dot n + dr) ds`. Rectangle-size variations
also include the clipped straight-edge lengths. Plate height differentiates
the extrusion directly. All parameters remain differentiable, including
rectangle dimensions; no future-event correction is added.

This rule avoids singular endpoint intermediates and arbitrary AD choices
when sorted endpoints coincide. Ordinary AD of the same safe primal is a
required comparator: it agrees in regular configurations but can select
incorrect derivatives at representation ties even when the area is smooth.
That observation is specific to this representation, not an impossibility
result for ordinary AD. Only first derivatives are supported by the contract.

Keep host OCCT rebuilding and measurement in the benchmark. Compare with
independent cross-section integration, analytic contact formulas, fresh CAD
volumes and finite differences, and fresh meshes after actual CAD transforms.
Use a two-dimensional grid times extrusion height for sampled plate baselines;
retain temperature choices, failures, setup costs, and synchronized timings.

## Consequences

The experiment can evaluate volume through a contact without freezing a
mesh topology. It is neither a general Boolean engine nor an extension of
the imported-hole editing contract. Interior hole-center motion still has
zero volume derivative. Radius-driven inverse design can cross the first
edge while using a volume objective with a valid nonzero derivative.

At exact tangency the boundary can pinch despite the scalar volume being
C1. A failed mesh closure check there does not refute the volume derivative;
it prevents claiming a regular solid mesh at that configuration. OCCT's
absolute tolerance floor can also spoil scale covariance of remeshing.

Use local coordinates and x64 for sensitive comparisons. Extreme radius to
plate ratios and nearly complete stock removal lose relative accuracy in
floating point. Supported examples do not certify arbitrary aspect ratios,
second derivatives, or topology-changing edits of general input CAD.

Boundary shape calculus and exact derivatives of cut discretizations are
established techniques; see [Wegert et al. (2025)](https://arxiv.org/abs/2504.09748v3)
and [Wegert et al. (2026)](https://arxiv.org/abs/2609.15084v2).
This implementation makes no novelty claim for those techniques. Its evidence
concerns this analytic curved family, first-derivative consistency, and
measured accuracy/cost. Broader generalization requires a separate decision.
