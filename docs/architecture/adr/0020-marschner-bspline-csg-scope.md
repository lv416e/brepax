# ADR-0020: Marschner trim-aware composition is also out of scope for BSpline CSG-Stump primitives

## Status

Proposed

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

The mechanism behind the Linkrods volume collapse is the same one
ADR-0019 documented for analytical primitives, applied to BSpline:

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
  features or many faces sharing the same parametric surface.  The
  Marschner blend then returns `d_partial >= 0` for face k, which
  the DNF interprets as "outside the trimmed primitive k", and the
  intersection-cell row evaluates the query as outside the solid
  even though the analytical half-space would have placed it inside.
- BSpline patches do not change this picture.  Their untrimmed
  surface is finite in parameter, but the underlying mathematical
  surface still extends across the patch (and beyond, on its
  parametric extension); the CSG-Stump primitive's role is to expose
  the half-space sign of that surface, and the patch's parametric
  finiteness is irrelevant to that role.

The plane non-asymmetry recorded in the original `TrimmedCSGStump`
module docstring ("the untrimmed half-space is *already* the correct
CSG ingredient") and extended in ADR-0019 to all analytical
primitives applies uniformly: every primitive type, including BSpline,
should expose the untrimmed signed distance inside the CSG-Stump DNF.

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
- The Linkrods CSG-Stump direct-path phantom (+31.6%) remains
  unaddressed under this approach.  Reducing that phantom is
  scope for a future ADR that explores GWN-based composition or
  per-face boundary integration on the trim polylines, and is
  tracked separately from the trim-aware composition story.
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
