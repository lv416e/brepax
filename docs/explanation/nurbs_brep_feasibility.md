# NURBS Differentiation Scope

Differentiating a parametric surface, a trimmed face, and a closed CAD solid
are distinct problems. BRepAX implements parts of each, but their derivative
contracts cannot be interchanged. The [API overview](../reference/index.md)
records supported inputs and numerical assumptions.

## Surface evaluation and distance

A rational B-spline surface maps parameters `(u, v)` to a point using control
points, knot vectors, degrees, and weights. JAX evaluation exposes derivatives
of this map with the array shapes and degrees fixed.

`BSplineSurface.sdf` adds a closest-point problem. It chooses an initial
parameter pair from coarse samples and runs Newton projection on squared
distance. The sign comes from surface normals, using precomputed coarse
normals when available. An open patch does not by itself define the inside
of a closed solid.

The implemented derivative follows the unrolled Newton iterations in
`brepax.nurbs.projection`, with the discrete initial-sample choice held fixed.
It is not an implicit derivative of an exactly converged global minimum.
Multiple closest points, clipping at parameter bounds, or a change in the
selected initial sample can affect differentiability. Projection convergence
and independent distance/gradient checks therefore matter in addition to
surface-evaluation accuracy.

The earlier prototype proposal on this page recommended implicit
differentiation before the current implementation existed. That remains a
possible alternative, not the implemented contract or an approved solver
migration.

## Trimming and solid composition

A trim polygon restricts a face in parameter space. A nearest point on the
underlying surface can lie outside that polygon; the distance to a trimmed
face then also depends on its boundary. BRepAX has trim metadata and
standalone face-distance wrappers for this purpose.

Combining those face distances into a solid is a separate operation. A
finite-face distance does not necessarily carry the half-space sign needed
by a CSG intersection. The attempt to substitute a Marschner trim-aware blend
inside the CSG-Stump DNF was rejected for analytical surfaces and B-splines.
[ADR-0020](../architecture/adr/0020-marschner-bspline-csg-scope.md) records the
counterexample and current use of raw primitive SDFs. Retaining trim metadata
alone does not make that composition trim-aware or resolve its phantom-volume
failure cases.

## Solid metrics and geometry edits

The mesh path uses OCCT tessellation followed by JAX surface evaluation and
mesh integrals. It avoids using an approximate solid SDF for volume, but its
derivatives depend on which data are allowed to move:

- With fixed connectivity and UV samples, AD differentiates the chosen mesh
  evaluation. CAD reconstruction and tessellation are outside that graph.
- Moving a surface without moving its neighboring trim boundaries can open
  the mesh. Shared boundary motion must be part of the parameterization.
- Rebuilt CAD and fresh meshing supply a separate reference for an edit.
  Agreement on one fixed discretization does not establish agreement after
  reconstruction or through a change in topology.

The [CAD editing benchmark](https://github.com/lv416e/brepax/tree/main/benchmarks/cad_editing)
provides a bounded roof/through-hole family with coupled boundary motion.
Its derivative comparisons use a shared working mesh. The inverse-edit
benchmark separately checks rebuilt CAD, fresh meshing, and STEP roundtrips
after rigid transformations and unit changes. These checks apply to the
supplied family; they do not establish arbitrary STEP editing, derivatives
through remeshing, or topology-changing edits.

## Evidence needed for broader claims

For a new surface or edit family, check values against an independent
geometric reference and derivatives against analytic results or a
finite-difference step sweep. Include non-convex projection cases, trim
boundaries, closure, orientation, units, and near-degenerate inputs when they
are relevant to the claimed capability.

Rigid transformations and changes of units should be assessed for both
values and gradients. Transform external directions with the geometry and
scale dimensional tolerances consistently. Separate discretization error
from violations of the geometric law, and distinguish a transformed fixed
mesh from transformed CAD followed by fresh meshing.

Neither primitive-level stratum tests nor fixed-topology mesh derivatives
establish a general differentiable NURBS Boolean kernel. That broader claim
needs independent evidence across changing intersections and trim topology.
