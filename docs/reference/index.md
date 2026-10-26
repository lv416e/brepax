# API Reference

## Capabilities and limits

### STEP preparation and JAX evaluation

STEP reading and OCCT extraction run on the host, outside `jit`, `vmap`,
and differentiation. The numerical kernels operate on the resulting arrays
and geometric models; JAX transformation support follows each API's contract.

Face conversion and mesh extraction support planes, cylinders, spheres,
cones, tori, and B-spline surfaces (including rational weights). Other OCCT
surface types are unsupported. `face_to_primitive` warns and returns `None`
for them; mesh extraction raises `NotImplementedError` rather than returning
an incomplete mesh. CSG reconstruction also rejects unsupported faces.
Successful conversion alone does not establish an accurate solid SDF.

`extract_mesh_topology` and `evaluate_mesh` separate OCCT preparation from
JAX vertex evaluation. Connectivity and the extracted UV samples form a fixed
discretization; `uv_scale_param` can scale plane UV coordinates so that caps
follow a linked parameter. Parameter changes must preserve closure and shared
boundaries; a derivative of this discretization is not a derivative through OCCT
reconstruction or remeshing. Topology-changing edits require separate
validation.

### Mesh mass properties

For a watertight, consistently oriented mesh, the divergence theorem gives
the mesh's signed volume. Accuracy against the original curved CAD depends
on tessellation. The [integration report](benchmarks.md) records fixture
measurements, not an error bound for every supported surface or input.

| Function | Numerical contract |
|----------|--------------------|
| `divergence_volume` | Cubic polynomial in triangle vertices; signed volume in length cubed. |
| `mesh_surface_area` | Sum of triangle areas, in length squared; uses a norm, not a polynomial. Collapsed triangles use a chosen zero subgradient. |
| `mesh_center_of_mass` | First moments divided by volume, in length units; undefined for nonempty zero-volume inputs. |
| `mesh_inertia_tensor` | Tensor about the center of mass at unit density, in length to the fifth power; requires nonzero volume and outward orientation. |

Nearly degenerate solids can make the normalized moments ill-conditioned.
See the function contracts below for empty-input behavior and other assumptions.

### Boolean measures and trimmed surfaces

Union, intersection, and subtraction have stratum-dispatched measure
implementations. Where finite primitive volume formulas apply, disjoint and
contained configurations use analytical gradients. Intersecting configurations
and other fallbacks differentiate a smoothed indicator on a frozen grid as a
surrogate for the hard sampled forward measure. These estimators have
discretization error and do not establish exact derivatives through arbitrary
CAD topology changes.
See [Boolean operations](boolean.md) for the APIs.

Standalone trimmed-face distances and CSG half-space composition have
different contracts. `TrimmedCSGStump` uses raw primitive SDFs in its DNF
composition, including B-splines; trim frames are retained as metadata.
[ADR-0020](../architecture/adr/0020-marschner-bspline-csg-scope.md) explains
why the trim-aware blend is excluded from that composition. See
[NURBS differentiation scope](../explanation/nurbs_brep_feasibility.md) for
the remaining distinction between surface, face, and solid derivatives.

## Primitives

- [Primitives](primitives.md) -- Geometric primitives with SDF interface

## Boolean Operations

- [Boolean](boolean.md) -- Differentiable Boolean operations with stratum-dispatched gradients

## Analytical

- [Analytical](analytical.md) -- Closed-form ground-truth solutions

## Experimental

- [Applications](applications.md) -- Application demonstrators (mold direction optimizer)

## STEP and mesh APIs

::: brepax.io.step.read_step

::: brepax.brep.convert.face_to_primitive

::: brepax.brep.triangulate
    options:
      members:
        - triangulate_shape
        - extract_mesh_topology
        - evaluate_mesh
        - divergence_volume
        - mesh_surface_area
        - mesh_center_of_mass
        - mesh_inertia_tensor
      show_source: false
