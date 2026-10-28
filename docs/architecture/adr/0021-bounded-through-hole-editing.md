# ADR-0021: Bounded through-hole editing from B-Rep input

## Status

Accepted

## Context

The roof/through-hole benchmark constructs one prescribed CAD family. Its
parameter derivatives do not supply an editing API for an imported part.
General CSG reconstruction and topology-changing edits have separate unresolved
limits (ADR-0020). A useful next capability is independent radius editing of
recognized through-holes without assuming the stock's construction history.

## Decision

Provide an experimental application API for full cylindrical through-holes in
one valid solid, with perpendicular planar ends and sufficient clearance.
Keep recognition, normalization and OCCT reconstruction on the host, behind
the existing OCCT binding boundary. Keep the JAX model purely array-based.

A fixed reference mesh shares welded vertices. Radial displacement moves the
cylinder and its planar boundary together, tapering to zero before reaching
another feature. Parameters are external-unit radii. Conservative radius bounds
and a traced local orientation check reject unsupported edits. Mesh derivatives
remain derivatives of this frozen discretization. Independent OCCT rebuilding
fills the selected original cylinders and cuts their replacements, with checks
against the isolated-hole volume and area changes.

This capability does not infer general modeling history, edit Boolean topology,
or support blind holes, split cylindrical walls, oblique ends or interacting
features. Face identifiers belong only to the prepared input. Feature identity
tracking after reconstruction is not part of this decision.

## Consequences

Callers can optimize selected hole radii on imported parts, then rebuild CAD for
independent validation or export. A fixed working frame avoids evaluating mass
properties at large world origins and restores dimensional scaling explicitly.

Discretization and CAD tolerances remain part of the error budget. Preparation
and local orientation checks are not a general mesh intersection certificate.
Unsupported faces, failed welds, and invalid edits remain explicit. Promotion
beyond experimental status requires broader CAD coverage and accuracy evidence;
no new production dependency or general CAD-kernel claim is introduced.
