# ADR-0022: Boundary derivatives for bounded contact-volume experiments

## Status

Accepted for experimental scalar and one-sided directional evaluators with
reproducible benchmarks.

## Context

Method C samples a hard indicator for its value and uses a smoothed
intersecting backward rule. Small overlaps can escape sampled detection.
The bounded editing API in ADR-0021 deliberately stays away from contact;
its fixed mesh cannot establish derivatives of topology-changing geometry.

A perpendicular circular through-cut in a rectangular plate is a bounded
case with curved boundaries, an actual hole-to-open-notch transition, and
independent volume references. Its volume is C1 at ordinary edge and corner
contacts. Positive-area planar contact has a different derivative contract,
addressed by the bounded extension below. Arbitrary STEP reconstruction and
surface-area derivatives remain outside this decision.

## Decision

### Circular cut with a continuous first derivative

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

### Isolated opposing planar contact

Add `brepax.experimental.planar_contact` for a known convex planar contact
polygon and a supplied rigid-motion direction. Return the positive one-sided
intersection-volume response, integrating the positive normal approach speed
over the polygon. A triangle fan reduces the affine integrand to closed-form
triangle integrals. Normalize speeds and use positive factored expressions;
subtracting a nearly equal negative cap loses small positive regions.

This response is nonlinear in direction. Opposite tilts of a contacting
square can both create overlap, so no single gradient or ordinary JVP can
represent all directions. Do not install a custom JVP or revive the TOI stub
for this contract. The [stratification explanation](../../explanation/stratification_theory.md#directional-response-at-opposing-planar-contact)
states the isolated-contact assumptions and local overlap-layer derivation.

Keep B-Rep preparation in `experimental.planar_contact_cad`. Require explicit
face IDs, convex polyhedral solids, unique opposing support faces, and a
positive-area polygon intersection. Return a nominal support-polygon model
with source residuals. Require callers to choose computed-zero residuals or
explicit roundoff-scale projection. Neither mode certifies exact contact of
the source CAD or modifies its solids. OCCT tolerances are not a physical
contact criterion; a positive gap has a different derivative.

Compare with independent cross-section integration, existing sampled Method C,
sigmoid and smoothed-composition estimators, same-forward finite differences,
and fresh finite CAD edits. Keep calibration and heldout geometry groups
separate across five fixed splits. Select directions using the same candidate
sets and compare actual CAD volume at the same finite step, including a panel
where ordinary normal approach can remove any advantage. Report host setup,
compilation, synchronized execution, finite-step errors, and tolerance failures.

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

The planar extension handles one isolated opposing contact patch with positive
thickness, not existing overlap, general multiple contacts, or curved contact
patches. A directional response is only first-order information: verify the
chosen finite edit independently. Mixed tilt can create wedges below OCCT's
resolution even when the continuous response is nonzero.

Boundary shape calculus and exact derivatives of cut discretizations are
established techniques; see [Wegert et al. (2025)](https://arxiv.org/abs/2504.09748v3)
and [Wegert et al. (2026)](https://arxiv.org/abs/2609.15084v2).
This implementation makes no novelty claim for those techniques. Its evidence
concerns the stated bounded families, derivative consistency, and measured
accuracy/cost. The cited cut-AD constructions assume stable cut topology;
they do not directly prove the birth of overlap at opposing-face contact.
Broader generalization requires a separate decision.
