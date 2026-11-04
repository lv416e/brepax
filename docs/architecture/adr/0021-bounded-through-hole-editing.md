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

A fixed reference mesh shares welded vertices; it is the watertight
triangulation of ADR-0024, whose shared vertices come from OCCT's edge
discretizations rather than from coordinate proximity. Radial and transverse displacement
move the cylinder and its planar boundary together, tapering to zero before
reaching another feature. Parameters are external-unit radii and optional
world-coordinate center shifts, with the hole axis fixed. Omitting shifts
preserves radius-only calls. Transverse projection removes numerical axial
residuals; materially axial input is rejected. Shift magnitude and positive
radius growth share the same conservative clearance budget. Radius bounds
and a traced local orientation check reject unsupported edits. Mesh derivatives
remain derivatives of this frozen discretization. Independent OCCT rebuilding
fills the selected original cylinders and cuts their replacements, with checks
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
nearly coincident solids.
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

Discretization and CAD tolerances remain part of the error budget. Preparation
and local orientation checks are not a general mesh intersection certificate.
Unsupported faces, failed welds, and invalid edits remain explicit. Promotion
beyond experimental status requires broader CAD coverage and accuracy evidence;
no new production dependency or general CAD-kernel claim is introduced.
