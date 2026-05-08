# ADR-0019: Marschner trim-aware composition scope

## Status

Proposed

## Context

ADR-0018 fixes the composition formula

```
d_T(p) = chi_T(pi_S(p)) * d_S(p)
       + (1 - chi_T(pi_S(p))) * d_partial(p)
```

with `d_S` signed and `d_partial` non-negative.  The formula and its
four sub-decisions (hard min, untrimmed knot domain, sign by dropping
`abs`, frame as composition argument) are mathematically sound and
verified end-to-end on a single trimmed plane face against
`BRepExtrema_DistShapeShape`.

The implementation that wired `d_T` into `TrimmedCSGStump` substituted
the Marschner blend for every curved analytical primitive (cylinder,
sphere, cone, torus) inside the CSG-Stump DNF composition, while
keeping plane primitives on their raw half-space SDF.  A baseline
benchmark on four analytical fixtures (`sample_box`, `sample_cylinder`,
`box_with_holes`, `l_bracket`) measured the trim-aware composite
against OCCT analytic ground truth at 64-cube grid resolution and
showed:

| Model | V_direct error | V_trim_aware error |
|---|---|---|
| `sample_box` (planes only) | -0.25% | -0.25% |
| `sample_cylinder` (1 cyl + 2 caps) | +0.89% | -0.81% |
| `box_with_holes` (6 planes + 2 holes) | +0.10% | -3.30% |
| `l_bracket` (planes only) | -0.44% | -0.44% |

Plane-only fixtures are bit-identical between the two paths, confirming
the dispatch wiring is sound for that surface type.  The two fixtures
that exercise curved primitives degrade under the trim-aware path:
`box_with_holes` collapses from a near-zero direct error to -3.30%.

Isolation experiments narrowed the cause to a category error in the
substitution itself, not a numerical or sign bug:

- The CSG-Stump DNF treats every primitive's signed distance as a
  *half-space ingredient*.  A box is the intersection of six plane
  half-spaces; a cylindrical hole is the intersection of an outer
  half-space (`d_cylinder > 0`) with the surrounding plane
  half-spaces.  The untrimmed primitive SDF *is* that ingredient and
  is correct for any analytical primitive, regardless of whether the
  face is trimmed.
- `d_T(p)` is not a half-space SDF.  It is the signed distance to the
  trimmed face's effective surface, which is finite.  Outside the
  trim parameter range, `chi_T -> 0` and `d_T -> d_partial >= 0`,
  classifying the query as outside the trimmed primitive even when
  the analytical half-space would correctly classify it as inside.
  Substituted into the DNF, this collapses entire half-spaces to
  thin shells and breaks the composition.
- The plane non-asymmetry recorded in the original trimmed-stump
  module docstring ("the untrimmed half-space is *already* the
  correct CSG ingredient") generalises to every analytical primitive.
  Plane was not the special case; it was the only case implemented
  correctly by accident, because the implementation kept the raw
  half-space distance for that one surface type.

The asymmetry that *does* exist in the CSG-Stump's needs:

- Analytical primitives (plane, cylinder, sphere, cone, torus) all
  have well-defined infinite half-spaces.  The untrimmed analytic
  signed distance is the correct CSG ingredient regardless of trim.
- BSpline patches are *finite* in their parameter domain.  Their
  untrimmed signed distance loses geometric meaning past the patch
  boundary, and the half-space concept does not extend.  This is the
  case ADR-0016 documented as "BSpline half-space limitation" and
  the case Linkrods (+219% on volume via raw `abs(BSpline SDF)`)
  empirically demonstrated.  The Marschner blend bypasses the
  half-space concept by composing with `d_partial`, recovering a
  signed quantity that classifies queries outside the patch as
  outside the primitive.

ADR-0018's motivation paragraph cites Linkrods explicitly, so the
intent was always BSpline-focused; what was missing was a written
boundary stating that analytical primitives keep their raw SDF.

## Decision

The Marschner trim-aware composition (`brep/trim_sdf.py:trim_aware_sdf`)
and its per-primitive wrappers (`brep/trim_frame.py`'s
`*_face_sdf_from_frame`) are reserved for **two use cases**:

1. **Standalone trimmed-face distance queries.**  Computing the
   signed distance from a query point to a single trimmed face, for
   uses such as mesh-SDF replacement, OCCT distance comparison, or
   point-to-shell queries that operate on one face at a time.
   Verified against `BRepExtrema_DistShapeShape` on plane in PR #67;
   sphere / cylinder / cone / torus wrappers extend the same
   semantics.
2. **BSpline-patch composition inside a CSG-Stump.**  When a future
   PR wires BSpline primitives into the trim-aware composition,
   their dispatch path uses `trim_aware_sdf` because BSpline patches
   are finite in parameter and their untrimmed extension is the
   phantom source.

The Marschner trim-aware composition is **not** used as the
per-primitive SDF inside `TrimmedCSGStump` for analytical primitives.
Every analytical slot returns its raw `primitive.sdf(query)`, the
same SDF that `DifferentiableCSGStump` uses.  This makes
`TrimmedCSGStump` analytically equivalent to `DifferentiableCSGStump`
on analytical-only models; the class continues to exist as the entry
point for the BSpline path that ADR-0018 motivates.

The trim-frame data (`PlaneTrimFrame`, `CylinderTrimFrame`,
`SphereTrimFrame`, `ConeTrimFrame`, `TorusTrimFrame`) is still
extracted at face-reconstruction time and stored on the
`TrimmedCSGStump`, both because the standalone-face use case (1)
needs it and because the BSpline integration (2) is the next
implementation step and will need a uniform per-slot frame slot.

The intersection-matrix column flip that compensated for the
dispatch's sign-flipping of REVERSED faces is removed: with raw
analytical primitives in every slot, the original
`stump.intersection_matrix` is the correct DNF coefficient matrix
without modification.

## Consequences

- `TrimmedCSGStump` on analytical-only models is bit-equivalent to
  `DifferentiableCSGStump` on the same grid.  The per-model gates in
  the trim-baseline benchmark collapse to "trim-aware path matches
  untrimmed path on analytical models" rather than "trim-aware path
  is closer to OCCT GT".  The actual phantom-reduction gate moves to
  the BSpline integration that follows.
- The plane non-asymmetry comment in the trimmed-stump module
  docstring is rewritten as the general analytical-vs-BSpline
  asymmetry stated above.
- ADR-0018 stands as the formula's mathematical specification and is
  not superseded.  Its sub-decisions (hard min, untrimmed knot
  domain, signed blend, frame as composition argument) remain in
  effect for the standalone-face and BSpline-patch use cases.
- Gradients through `TrimmedCSGStump.sdf` flow through primitive
  parameters (`radius`, `axis`, plane `normal`, etc.) as on
  `DifferentiableCSGStump`.  Frame fields (trim polygon vertices,
  3D polyline samples) are stored but not on the analytical
  gradient path; they will become differentiable inputs only when
  the BSpline path lands and a future trim-as-design-variable
  feature is added.
- `enrich_with_trim_frames` no longer flips intersection-matrix
  columns based on face orientation; the original matrix is used
  unchanged.
- The standalone-face wrappers (`plane_face_sdf`,
  `cylinder_face_sdf`, etc.) and their unit tests are unaffected;
  they continue to verify the Marschner formula against OCCT for
  use case (1).
- A regression benchmark (`tests/benchmarks/test_trim_baseline.py`)
  records the both-paths-match invariant on the four analytical
  fixtures so any future regression of the analytical path against
  `DifferentiableCSGStump` surfaces immediately.
