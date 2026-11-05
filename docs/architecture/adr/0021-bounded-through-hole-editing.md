# ADR-0021: Bounded through-hole editing from B-Rep input

## Status

Accepted

## Context

The roof/through-hole benchmark constructs one prescribed CAD family. Its
parameter derivatives do not supply an editing API for an imported part.
General CSG reconstruction and topology-changing edits have separate unresolved
limits (ADR-0020). A useful capability is independent radius and center editing of
recognized through-holes without assuming the stock's construction history.

## Decision

Provide an experimental application API for full cylindrical through-holes in
one valid solid, with perpendicular planar ends and sufficient clearance.
Keep recognition, normalization and OCCT reconstruction on the host, behind
the existing OCCT binding boundary. Keep the JAX model purely array-based.

The model holds the exact laws of the represented holes: with `t = r - r0`, a
hole of height `h` and center `c`, moved transversely by `s`, changes the
volume by `-pi h t (2 r0 + t)`, the surface area by
`2 pi h t - 2 pi t (2 r0 + t)` and the first moment by
`-pi h (t (2 r0 + t) c + r^2 s)`. Reference measures come from adaptive OCCT
integration, so an unedited model reproduces the CAD measures. Parameters are
external-unit radii and optional world-coordinate center shifts, with the hole
axis fixed. Omitting shifts preserves radius-only calls. Transverse projection
removes numerical axial residuals; materially axial input is rejected. Shift
magnitude and positive radius growth share the same conservative clearance
budget. Radius bounds, this budget and transversality alone decide whether an
edit is supported.

An optional frozen mesh serves responses that have no law. It is the watertight
triangulation of ADR-0024, whose shared vertices come from OCCT's edge
discretizations rather than from coordinate proximity. Radial and transverse
displacement move the cylinder and its planar boundary together, tapering to
zero before reaching another feature. Its derivatives remain derivatives of
this frozen discretization, and its local orientation check limits only that
mesh: a mesh problem never restricts a hole edit or its rebuild.

Measured on the local NIST, FreeCAD and miscellaneous fixtures with every hole
moved together, the exact volume change was closer to the rebuilt CAD than the
mesh on all 18 parts compared (median relative error about 6e-9 against 1e-3).
The mesh orientation check rejected edits that rebuilt correctly, including
more than half of the sampled radii on FTC-08 and STC-08, and a crack away from
the holes blocked FTC-07 entirely, while each of those edits rebuilt and
validated. With the model's bounds as the only gate, 1267 trials on 19 parts
(radii at and inside the bounds, shifts up to the budget, subsets of holes and
closest pairs moving toward each other) rejected only the 41 deliberately
out-of-bounds edits. All 1226 others rebuilt and passed an independent check:
one valid solid, whole-solid volume within 1.8e-7 and area within 2.3e-8
relative of the laws, and classifier probes at each edited bore.

Independent OCCT rebuilding fills the selected original cylinders and cuts
their replacements, with checks
against isolated-hole volume, area and first-moment changes. The moment check
detects misplaced holes that volume and area cannot distinguish. Whole-solid
OCCT properties can shift when a Boolean rebuilds faces far from the edited
holes: by about 1e-7 relative under OCCT's default integration rule, and by
up to about 1e-8 relative in geometry on the NIST parts measured. If those
checks fail, a coaxial envelope around each old and new bore decides
instead. Each solid is intersected with the envelope separately, and volumes
and moments are integrated adaptively to 1e-9 relative. Inside the envelope,
the change in volume and first moment must match the edit. Because
recognition accepts end planes within the CAD tolerance of the cylinder
extent, this comparison allows one tolerance-thick slice of the changed
cross-section at each end; it resolves misplaced bores at a few tolerances,
not below. Outside the envelope, volume and first moment may change by
1e-7 relative. A larger outside change passes only if every region of the
difference between the two solids lies beside the edited bores: on a part
with a large CAD tolerance, a Boolean can move faces within that tolerance
and change measured properties without leaving a difference region. The
inside comparison never subtracts the two solids from each other: on NIST
CTC-05, OCCT returned a "removed" region larger than the whole solid for two
nearly coincident solids. Whole-solid area must also follow its law to 1e-7
relative, adaptively integrated: material inside a bore thin enough to pass
recognition's volume-based emptiness check is erased by the fill, which shows
in the area and not in the volume or first moment.
Unify connected same-domain faces and edges after the Boolean operations so
fill seams do not divide the moved hole's end planes and prevent re-editing.
This preserves unselected geometry, not the original partition into faces.

This capability does not infer general modeling history, edit Boolean topology,
or support blind holes, oblique ends or interacting features. A cylindrical
wall that a STEP exporter split into several faces, such as two half-cylinder
faces or an axial seam, is rejected by default for the same reason a genuinely
incomplete cylinder is. An opt-in preparation step unifies same-domain
faces/edges of the working solid first, with a volume/area check against the
shape's own CAD tolerance, so such splits can be recognized as one complete
hole; it does not relax any other rejection. Face identifiers belong only to
the prepared input, which is the unified working solid when that step runs.
Feature identity tracking after reconstruction is not part of this decision.

## Consequences

Callers can optimize selected hole radii and centers on imported parts, then rebuild CAD for
independent validation or export. A fixed working frame avoids evaluating mass
properties at large world origins and restores dimensional scaling explicitly.

CAD tolerances remain part of the error budget of the laws. Discretization
error belongs only to the optional mesh, whose preparation and local orientation
checks are not a general mesh intersection certificate. Unsupported faces,
failed welds, and invalid edits remain explicit. Promotion
beyond experimental status requires broader CAD coverage and accuracy evidence;
no new production dependency or general CAD-kernel claim is introduced.
