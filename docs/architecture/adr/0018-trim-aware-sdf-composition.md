# ADR-0018: Trim-aware surface SDF composition

## Status

Proposed

## Context

BRepAX reconstructs each face as a CSG-Stump primitive and reduces the
composite SDF to a volume via sigmoid integration on a regular grid.  On
trimmed faces — especially BSpline patches whose parametric extension runs
past the face boundary — the untrimmed `abs(SDF)` reports a distance to the
*mathematical* surface, not to the *trimmed* surface, and the sigmoid layer
captures phantom material outside the solid.  Linkrods is the worst measured
case (`abs(BSpline SDF)` drifts +219 % on volume; see ADR-0016).

The remediation path adopted here is the Marschner-style composition
(Marschner et al. 2023, "CSG on Neural SDFs").  The literal paper form is
unsigned and relies on a shell-level sign source:

```
  |d_T(p)|  =  chi_T(pi_S(p)) * abs(d_S(p))
            +  (1 - chi_T(pi_S(p))) * d_partial(p)
```

The form BRepAX uses drops the `abs` and blends the already-signed
`d_S` with the unsigned boundary distance so the result is itself a
signed SDF without any shell-level sign reconstruction:

```
  d_T(p)  =  chi_T(pi_S(p)) * d_S(p)
          +  (1 - chi_T(pi_S(p))) * d_partial(p)
```

- `d_S(p)` — signed distance to the untrimmed analytic surface
  (existing primitive `.sdf()`).
- `pi_S(p)` — foot-of-perpendicular on the untrimmed surface, returning
  the 3D foot and the UV parameter (PR #62).
- `chi_T(u, v)` — smooth trim indicator in UV, 1 inside the trim polygon
  and 0 outside (existing `nurbs/trim.trim_indicator`).
- `d_partial(p)` — unsigned 3D distance to the closed trim boundary
  polyline (PR #63), always `>= 0`.

Four sub-decisions remain before `d_T(p)` itself can be wired into the
CSG-Stump path.  They are all small, tightly related, and cheap to reverse
individually.  This ADR fixes them in one place so the composition code and
its tests can be written against a pinned contract rather than re-deriving
the contract from implementation.

The four sub-decisions:

1. The `min` over the trim-boundary segments is non-smooth.  Does the first
   implementation use hard `jnp.min`, or a smoothed approximation
   (`softmin_beta`)?
2. The existing BSpline projection clamps `(u, v)` to a trimmed parameter
   range when one is provided.  Marschner's `pi_S(p)` is defined on the
   *untrimmed* surface.  Which domain does the composition use?
3. `d_partial(p)` is unsigned by construction.  How does `d_T(p)` acquire a
   sign that is correct both inside the trim and outside it, without a
   shell-level winding pass?
4. Analytic primitives in BRepAX do not store a local 2D frame, but
   `chi_T(u, v)` and `pi_S(p)` both require one.  Does the frame live on
   the primitive class, or is it passed through the composition function?

## Decision

**1. Hard `jnp.min` first.**  The polyline distance uses the hard minimum
already implemented in `brep/polyline.py`.  A smoothed variant is not added
until a concrete measurement on representative fixtures (planned in the
baseline follow-up to PR #61) shows that optimization through the kink
breaks convergence.  The softmin variant has one hyperparameter
(`beta`) that couples to scene scale and would require its own tuning
study; hard-min has none.  Keeping hard-min until a number justifies
switching is the cheapest gate-first action.

**2. Use the untrimmed knot domain for `pi_S(p)`.**  When the composition
calls `closest_point_and_foot`, it passes `param_u_range=None` and
`param_v_range=None` so the Newton iterate explores the full knot domain.
Clamping to the trim sub-range collapses `pi_S(p)` onto the trim boundary
for every outside query, which defeats `chi_T` and violates the Marschner
identity.  No library change is required: the existing `None` default on
those two arguments is already correct.  What changes is that
*composition-layer callers do not forward any trimmed range*.

**3. Sign is derived at the primitive level by dropping `abs(d_S)` from
the composition formula.**  The literal Marschner formula multiplies
`chi_T` by the *unsigned* `|d_S|` and then relies on a shell-level sign
source such as a winding-number pass on the closed trimmed shell.  A naive
attempt to use `sign(d_T) = sign(d_S)` on top of that unsigned form does
not eliminate phantom material, because a query that is inside the
infinite half-space but outside the trim still gets `sign(d_S) < 0` and
the CSG-Stump PMC classifies it as inside the primitive.

BRepAX instead uses the signed blend

```
d_T(p) = chi_T(pi_S(p)) * d_S(p)
      + (1 - chi_T(pi_S(p))) * d_partial(p)
```

with `d_S` signed (from the primitive's `.sdf()`) and `d_partial` always
non-negative.  The four regimes behave correctly without any shell-level
reconstruction:

- Inside trim, inside half-space (`chi_T ~ 1`, `d_S < 0`): `d_T ~ d_S < 0`.
  Classified as inside the trimmed primitive.
- Inside trim, outside half-space (`chi_T ~ 1`, `d_S > 0`): `d_T ~ d_S > 0`.
  Outside.
- Outside trim, either side of the half-space (`chi_T ~ 0`): `d_T ~ d_partial >= 0`.
  Outside — phantom material eliminated regardless of the half-space
  sign.

The existing CSG-Stump PMC then composes these already-correct signed
primitive SDFs with no change to its own logic.  A standalone
winding-number pass on the trim-boundary polyline is rejected because
this signed blend carries the same information at no extra compute.  If
a future standalone-face use case needs the literal `|d_S|` formulation
with an external sign source, a follow-up ADR will revisit.

**4. UV frame flows through composition arguments, not primitive state.**
The per-face 2D frame required by `chi_T(u, v)` and by the analytic
`pi_S` (for plane, cylinder, cone, torus) is extracted once from OCCT at
face-reconstruction time and passed to the composition function as plain
tensors.  Primitive classes (`Plane`, `Sphere`, `Cylinder`, `Cone`,
`Torus`) are **not** extended to carry a frame.  Motivation: the frame is
face-level data (several faces can share the same analytic surface with
different trim frames), and pushing frame onto the primitive class
enlarges the public API of five classes that are otherwise pure math
objects.  The composition function takes `frame_u`, `frame_v`, and a
reference `origin` as additional arguments; the trim polygon in UV is
already face-level in the existing `nurbs/trim.py` pattern, so nothing
there changes.

## Consequences

- The composition module (name tentatively `src/brepax/brep/trim_sdf.py`,
  to be finalized with the implementation PR) is a thin wrapper over
  existing parts: `primitives/foot.py`, `nurbs/projection.py`,
  `nurbs/trim.py`, and `brep/polyline.py`.  No new primitive math is
  required.
- The BSpline projection path that the composition uses is called with
  `param_u_range=None`.  Other callers of `closest_point_and_foot` that
  need clamping to a trimmed sub-range keep their existing behaviour;
  there is no signature change.
- Sign correctness at the primitive level is now the composition
  module's responsibility via the signed blend.  The CSG-Stump PMC
  composes already-correct trimmed primitive SDFs and needs no change.
  If a future standalone-face use case needs the literal `|d_S|`
  formulation with an external sign source, a follow-up ADR will revisit
  decision 3.
- Softmin and a standalone winding-number sign pass are deferred, not
  rejected.  A measurement in the trim-aware follow-up to PR #61 decides
  whether either needs to be revisited; that follow-up reports phantom
  error before and after the composition on a shared fixture set so the
  decision is grounded in numbers.
- The five analytic primitive classes stay pure math.  Face-level frame
  data is the caller's responsibility (OCCT extraction at
  face-reconstruction time is already how `nurbs/trim.py`'s polygon
  vertices are prepared).
