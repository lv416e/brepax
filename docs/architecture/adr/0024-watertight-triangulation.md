# ADR-0024: Watertight triangulation

## Status

Accepted

## Context

`triangulate_shape` tessellates each face with OCCT `BRepMesh`, then
re-evaluates every triangle corner from that face's own JAX-native surface
parameters, independently of its neighbors. A vertex on an edge shared by two
faces therefore gets two numerically close but generally distinct copies: the
triangle mesh is watertight as a surface covering (no missing faces, see
ADR-0016) but not watertight at the index level (no single, shared vertex
identity for a physically coincident point).

This absence of true index-level sharing blocks `prepare_hole_edit`'s
`closed_mesh` weld (ADR-0021), which instead reconstructs identity after the
fact from a position-distance tolerance bucket. On the local NIST, FreeCAD and
misc fixtures, that tolerance weld fails to close 8 of 39 solids, and succeeds
only by discarding genuine topological information in favor of a tunable
distance threshold that cannot distinguish a real gap from independent
re-evaluation noise.

OCCT's own `BRepMesh` triangulation already shares bit-identical boundary
nodes and curve parameters between a shared edge's two owning faces
(`Poly_PolygonOnTriangulation`), including a periodic face's own seam (which
OCCT represents as two distinct, oriented boundary occurrences of that one
face) and a degenerate edge's collapsed pole or apex. A prototype that
identifies shared vertices from this OCCT edge topology (union-find over
`Poly_PolygonOnTriangulation` node correspondence, matched by the edge's own
curve parameter) and evaluates each global vertex once closed most of the
previously failing solids.

That prototype picked one of the two incident faces' evaluations as each
shared vertex's representative, which silently gives `d(vertex)/d(params)`
one-sided: the other incident face's shape-derivative contribution at that
row is zero, biasing any future optimization that moves both adjacent
surfaces, and on NIST FTC-07/STC-07 (and STC-10) picking the worse of the two
sides inflated a since-fixed evaluator defect (see below) up to 46%
relative area error with no signal that anything was wrong.

## Decision

Add `extract_watertight_topology` (host-side OCCT meshing, edge-topology
welding, and manifold validation) and `evaluate_watertight_mesh` (a pure JAX
function of explicit per-face parameters), plus `triangulate_shape_watertight`
as a convenience composition of the two. All are explicit opt-in additions;
`triangulate_shape`'s output, including its exact floating-point values, is
unchanged.

- Shared vertices are identified from OCCT edge topology, not a distance
  tolerance: a union-find over `(face, local node)` pairs, matched through
  each edge's own `Poly_PolygonOnTriangulation` node correspondence and its
  intrinsic curve parameter (not vertex position). Walking each face's own
  `TopExp_Explorer` (rather than the shape-level edge-to-face ancestor map)
  yields two oriented occurrences of a periodic face's seam edge, handled by
  the same matching code as an edge shared by two different faces. A
  degenerate edge (pole, apex) collapses every node any of its occurrences
  references into one vertex, further unioned across degenerate edges that
  share a topological vertex.
- `evaluate_watertight_mesh(topology, params_list)` evaluates every incident
  face's own surface parameters at its own nodes, then averages in two
  levels: first each face's own nodes at a given vertex, then those per-face
  means across the vertex's distinct incident faces. The second level
  matters whenever a face touches a vertex through more than one of its own
  nodes, e.g. at the point where a periodic face's own seam meets a
  neighboring face: a flat mean over nodes would weight that vertex 2/3
  toward the seamed face and 1/3 toward its neighbor; this keeps it an equal
  half each, so `jax.grad` distributes symmetrically per incident *face*
  rather than per node. Separating topology (host, static) from this
  evaluation (pure JAX) lets a caller differentiate, `jit`, or `vmap` a
  mesh metric with respect to specific per-face surface parameters (e.g.
  one face's B-spline control points) without repeating the OCCT work,
  analogous to the existing `extract_mesh_topology`/`evaluate_mesh` split
  for the non-welded mesh. B-spline degrees are shape-determining dispatch
  and cache keys, not differentiable data, so they stay concrete Python
  ints on `WatertightTopology.face_records` rather than inside
  `params_list`: a degree sitting in the traced pytree would become an
  unhashable tracer the first time it was used as a dict key.
- `WatertightMesh` is a registered JAX pytree (an `equinox.Module` of arrays
  only -- `vertices`, `faces`, `max_gap`), so `jit` and `vmap` can return it,
  not just take it as an argument. Host metadata that `triangulate_shape`
  carries alongside its triangle array (`surface_type`, per-face triangle
  counts) is not array data, so it lives on `WatertightTopology` instead
  (`face_records`, `n_triangles_per_face`), not on `WatertightMesh`.
- `WatertightMesh.max_gap` reports an upper bound on the largest disagreement
  between any two of a shared vertex's incident *nodes* (twice the largest
  distance from any individual node's raw evaluation to the vertex's final
  welded position; exact by the triangle inequality), computed directly from
  those raw evaluations rather than from the per-face means above. The two
  are deliberately different: the per-face mean used for `vertices` would
  otherwise silently absorb a disagreement between two nodes of the *same*
  face (e.g. both sides of a periodic face's own seam) into that face's own
  mean, reporting an exact zero gap even when editing one control point
  measurably moves one of those two nodes relative to the other.
  `triangulate_shape_watertight`'s optional `tolerance` raises when this
  bound is exceeded, so a genuine evaluator disagreement is surfaced and can
  be rejected against the shape's own CAD tolerance instead of being
  silently absorbed into an averaged position.
- Validation checks both that every edge is shared by exactly two
  consistently oriented triangles, and that every vertex's incident triangles
  form one closed fan (a vertex-link check): the edge check alone accepts two
  otherwise-disjoint fans -- e.g. two tetrahedra -- pinched together at one
  shared vertex with no shared edge between them, since every edge inside
  each fan is still paired exactly twice.
- Returns an indexed `(vertices, faces)` pair (a `WatertightMesh`), not a
  `(n, 3, 3)` triangle-corner array: the index level is exactly what welding
  establishes, and `vertices[faces]` reproduces the existing triangle-corner
  layout for `divergence_volume` and the other mesh metrics.

`triangulate_shape` and the watertight path share their OCCT host-side
extraction and batched JAX evaluation (both already existed as one combined
function; the shared parts are now two private helpers). Eleven of the
twelve `.step` fixtures tracked in this repository produce bit-identical
`triangulate_shape` output before and after this refactor (checked once,
manually); the twelfth, `nurbs_revol.step`, raises `NotImplementedError` in
both versions (an unrelated, pre-existing unsupported-surface limitation).
A standing test checks that `triangulate_shape` reproduces OCCT's own
mesh triangle for triangle (located node coordinates, face orientation
applied) on four representative fixtures. A recorded hash is not a
portable reference, because OCCT's mesh itself differs between platforms:
`sample_sphere` has 2996 triangles on macOS and 3000 on Linux.

An earlier spike reported several percent volume error on FTC-07, STC-07
and STC-10. Welding was not the cause. OCCT placed some B-spline mesh nodes
slightly outside the knot domain, where `bspline_basis` returned a zero
basis and moved them to the origin, in both triangulation modes; welding
only spread each bad node to the neighboring face. `bspline_basis` now
continues the end polynomial pieces outside the domain, as
`Geom_BSplineSurface::Value` does. Re-measured on each of the 39 local
fixture solids alone, against OCCT `GProp` of that solid at the default
deflection: every mesh closes; the relative volume error is at most
1.4e-4 on the NIST parts (FTC-07 5.3e-6, STC-07 2.9e-6, STC-10 1.4e-4)
and at most 6.3e-3 on the coarse analytic samples, where it is
discretization error. `max_gap` bounds the raw disagreement between
incident faces before averaging, not the error of the averaged mesh. It
reaches 3.1e-2, 7.9e-3 and 6.2e-3 model units on those three parts and
0.11 on CTC-02, whose volume error is 2.1e-5; the source of these gaps has
not been investigated.

`triangulate_shape_watertight` can replace the `triangulate_shape` +
`closed_mesh` weld in `prepare_hole_edit`, but this decision does not wire
it in. `closed_mesh`'s position-based weld keeps whichever duplicate point
it meets first, while watertight welding averages, so the swap changes
`prepare_hole_edit`'s vertex positions, volumes and gradients on every
fixture, not only on the solids that previously failed to close. That
numerical change to an experimental public contract belongs in a separate
change with its own measurements.

## Consequences

Easier:

- All 39 solids in the local NIST, FreeCAD and misc fixtures close under
  `triangulate_shape_watertight`. `nurbs_saddle.step`, a single open
  B-spline patch with a genuine free boundary, contains no solid.
- A caller can differentiate, `jit`, or `vmap` a mesh metric with respect to
  specific per-face surface parameters (e.g. one face's B-spline control
  points) through `evaluate_watertight_mesh`, with a shape derivative that is
  symmetric across a shared boundary, matching the planned use this ADR's
  context describes for future per-face parameter edits.
- An evaluator disagreement at a shared vertex is a reportable, rejectable
  number instead of a silent, one-sided pick.
- A pinched (non-manifold) vertex is rejected, not just a mismatched edge.

Harder:

- Three function entry points now extract and batch-evaluate per-face data
  (via shared private helpers); a future change to that extraction must keep
  all three call sites' contracts in mind.
- `triangulate_shape_watertight` raises instead of returning a best-effort
  mesh when OCCT edge topology cannot resolve a shared vertex (a free
  boundary, a non-manifold edge, or an edge discretization with no stored
  curve parameters); callers that need a result for an open shell must use
  `triangulate_shape` instead.
- `prepare_hole_edit` still relies on `closed_mesh`'s tolerance weld and
  therefore keeps failing to close those same 8 fixtures until a dedicated,
  separately scoped change adopts `triangulate_shape_watertight` there.
