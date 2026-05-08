# ADR-0020: Marschner trim-aware composition is also out of scope for BSpline CSG-Stump primitives

## Status

Accepted

## Context

ADR-0019 corrected the scope of the Marschner trim-aware composition
defined in ADR-0018: every analytical primitive (plane, cylinder,
sphere, cone, torus) inside a CSG-Stump must contribute its raw
untrimmed signed distance to the DNF.  Substituting the Marschner
blend (`d_T = chi_T * d_S + (1 - chi_T) * d_partial`) breaks the DNF
because outside the trim parameter range `chi_T -> 0` and
`d_T -> d_partial >= 0`, classifying the query as outside the trimmed
primitive even when the analytical half-space would correctly classify
it as inside.

ADR-0019 left BSpline primitives outside its scope, on the assumption
that "BSpline patches are finite in parameter space, so the untrimmed
extension is the phantom source (ADR-0016, Linkrods +219%) and the
Marschner blend is the right replacement".  The trim-aware path
implemented for BSpline slots in the follow-up wired the Marschner
blend into the per-slot SDF inside `TrimmedCSGStump`.

A measurement on the Linkrods fixture (37 primitives reconstructed
from the OCCT B-Rep, of which 18 are BSpline) under that wiring
collapsed the volume catastrophically:

| Resolution | V_OCCT | V_trim_aware (BSpline Marschner) | Error |
|---|---|---|---|
| 8 | 3.847 | 0.0214 | -99.45% |
| 16 | 3.847 | 0.0001 | -100.00% |

The CSG-Stump direct path on Linkrods reports a +31.6% phantom
(memory `project_bspline_halfspace.md`); the +219% number that
ADR-0018 cited is a different metric (GWN-signed-min-dist path,
since abandoned in favour of GWN-only) and does not reflect the
CSG-Stump direct path's behaviour.  The motivation for applying
Marschner inside the CSG-Stump DNF therefore confused two failure
modes from different paths.

The Linkrods collapse is treated here as a **counterexample** to the
hypothesis that the Marschner blend is a valid per-primitive ingredient
inside the CSG-Stump DNF.  A counterexample of this magnitude (-99% on
volume) is by itself sufficient grounds to reject the hypothesis and
roll the BSpline path back to the raw signed distance, even without a
fully isolated causal mechanism.

The following mechanism is **consistent** with the observation and
with the analytical analysis recorded in ADR-0019 — but it has not
been confirmed by a slot-level isolation experiment (per-query
chi_T / d_partial / d_T traces with controlled inputs), so it is
stated as a working hypothesis rather than a proven cause:

- The CSG-Stump's DNF treats every primitive's signed distance as a
  half-space ingredient.  A multi-face closed solid is the
  intersection of half-spaces — for every face, the signed distance
  carries information about which side of the surface the query
  lies on, even at distances much larger than the face's spatial
  extent.
- The Marschner blend's `d_T(p)` represents the signed distance to a
  single trimmed face's effective surface, not to its half-space.
  Outside the trim parameter range it collapses to the unsigned
  distance to the trim-boundary polyline, which is by construction
  non-negative.
- For a query at the centre of a multi-face solid, the foot of
  perpendicular onto BSpline face k may legitimately land outside
  face k's trim polygon, especially when the solid contains slender
  features or many faces sharing the same parametric surface.  Under
  this hypothesis the Marschner blend would return `d_partial >= 0`
  for face k, which the DNF would interpret as "outside the trimmed
  primitive k", flipping the intersection-cell row to "outside the
  solid" for queries the analytical half-space would have placed
  inside.
- BSpline patches do not obviously change this picture.  Their
  untrimmed surface is finite in parameter, but the underlying
  mathematical surface still extends across the patch (and beyond,
  on its parametric extension); the CSG-Stump primitive's role is to
  expose the half-space sign of that surface.

Confirming this mechanism with a per-slot trace is left as future
work for any caller who wishes to revisit the Marschner-as-DNF-
ingredient direction; the present ADR's decision does not depend on
that confirmation.

Independent of the mechanism, the plane non-asymmetry recorded in the
original `TrimmedCSGStump` module docstring ("the untrimmed half-space
is *already* the correct CSG ingredient") and extended in ADR-0019 to
all analytical primitives is treated here as the working rule for
BSpline as well: every primitive type, including BSpline, exposes the
untrimmed signed distance inside the CSG-Stump DNF.

## Decision

The Marschner trim-aware composition is reserved for **two use
cases**, unchanged from ADR-0019:

1. Standalone trimmed-face distance queries (mesh-SDF replacement,
   OCCT distance comparison) handled by `brep/trim_frame.py`'s
   `*_face_sdf_from_frame` wrappers.  This includes the BSpline
   wrapper `bspline_face_sdf_from_frame`, which is unaffected by
   this ADR and continues to use the Marschner formula.
2. Future composition strategies that don't require half-space
   semantics (e.g. GWN, where each face contributes a winding-number
   sample rather than a half-space ingredient).

The Marschner trim-aware composition is **not** used as the
per-primitive SDF inside `TrimmedCSGStump` for any primitive type,
analytical or BSpline.  Every slot returns its raw
`primitive.sdf(query)`, the same SDF that
`DifferentiableCSGStump` uses.  This makes `TrimmedCSGStump`
analytically equivalent to `DifferentiableCSGStump` on every fixture;
the class continues to exist as the entry point for trim-aware
composition because it carries the per-face trim frame data and the
build pipeline (`enrich_with_trim_frames`) already extracts it from
OCCT.

The trim-frame data — `PlaneTrimFrame`, `CylinderTrimFrame`,
`SphereTrimFrame`, `ConeTrimFrame`, `TorusTrimFrame`, and
`BSplineTrimFrame` — continues to be extracted at face-reconstruction
time and stored on the `TrimmedCSGStump`.  The standalone-face use
case (1) consumes them, and the future composition use case (2) will
consume them without re-extracting.

## Consequences

- `TrimmedCSGStump.sdf` no longer dispatches on primitive type.  All
  primitives return `primitive.sdf(query)`.
- `TrimmedCSGStump` is bit-equivalent to `DifferentiableCSGStump` on
  every fixture, including BSpline-bearing ones.  The unit tests
  pin this invariant on `sample_box`, `box_with_holes`, and
  `nurbs_box`; `test_trim_baseline` extends it to the full benchmark
  set.
- The Linkrods CSG-Stump direct-path phantom remains unaddressed
  under this approach.  Memory recorded a +31.6% figure from the
  v0.4.0 era (`project_bspline_halfspace.md`); a session-time
  re-measurement on the current codebase reproduced ~+85% at res=8
  (the only resolution that fits in memory for this 37-primitive
  fixture).  Different resolutions and bbox-padding choices give
  materially different numbers, so any future ADR proposing to
  attack this phantom should fix the measurement protocol before
  proposing a strategy.  GWN-based composition and per-face
  boundary integration on the trim polylines are two of the
  candidates; both are out of scope for this ADR.
- The "trim-aware metrics" milestone direction this PR was
  originally framed under is preserved as an open question.  This
  ADR closes only the Marschner-as-DNF-ingredient hypothesis.  The
  cheapest first step recorded in the milestone task list — using
  the existing `BSplineSurface.trim_polygon` field (populated by
  `_convert_bspline_face` since v0.4.0) inside the per-face metric
  computations — has not been attempted yet and remains the
  recommended next direction.
- ADR-0018's standalone-face Marschner formula and its
  per-primitive verifications (PR #67 for plane against
  `BRepExtrema_DistShapeShape`) remain in effect.  ADR-0018 is not
  superseded; this ADR narrows what was implicitly thought to be a
  CSG-Stump composition use case.
- Gradients through `TrimmedCSGStump.sdf` flow through every
  primitive's differentiable parameters (analytical: `radius`,
  `axis`, plane `normal` / `offset`; BSpline: `control_points`,
  `knots_u`, `knots_v`, `weights`) as on `DifferentiableCSGStump`.
  Frame fields (trim polygon vertices, 3D polyline samples) are
  stored but not on the gradient path.
- `TrimmedCSGStump` keeps the BSpline trim frame because the
  standalone-face wrapper consumes it and because future
  composition strategies (GWN, etc.) will reuse the same
  metadata; re-extracting from OCCT for those callers would
  duplicate work.
