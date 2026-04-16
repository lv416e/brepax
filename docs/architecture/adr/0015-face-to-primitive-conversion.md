# ADR-0015: Face-to-Primitive Conversion

## Status

Accepted

## Context

STEP faces are bounded (trimmed) regions of infinite analytic surfaces.
BRepAX primitives -- Plane, Cylinder, Sphere, Cone, Torus -- represent
the corresponding infinite surfaces with a signed distance function.

To build differentiable shape representations from imported STEP files,
we need a conversion path from OCCT face objects to BRepAX Primitive
instances.  The key design question is how to handle the trim boundaries
that make a face finite.

## Decision

Convert each face's underlying surface parameters to a BRepAX Primitive,
ignoring trim boundaries.

### Scope

**Included:**

- `face_to_primitive()` converts a single `TopoDS_Face` to a Primitive
  by inspecting the surface type via `BRepAdaptor_Surface` and extracting
  geometric parameters (normals, radii, centers, etc.) as JAX arrays.
- `faces_to_primitives()` iterates over all faces of a `TopoDS_Shape`
  and returns a list of Primitive objects.
- Supported surface types: Plane, Cylinder, Sphere, Cone, Torus.
- Unsupported types (B-spline, Bezier, etc.) return `None` with a
  runtime warning.

**Deferred:**

- Trim boundary extraction and representation.
- CSG tree reconstruction from face adjacency.
- B-spline / NURBS surface approximation or fitting.
- Gradient computation through the conversion path.

### Design choices

**Infinite surfaces, not finite faces.**  A converted Plane extends
infinitely rather than being bounded to the rectangular region of the
original face.  This is the correct abstraction for SDF-based Boolean
operations, where CSG tree reconstruction (future work) will reintroduce
the bounding via intersections.

**OCP access through the abstraction layer** per ADR-0008.  All OCCT
imports remain in `brepax._occt.backend`.

**JAX arrays for all parameters.**  Points, directions, and scalar radii
are converted to `jnp.array` at the boundary, making the resulting
Primitives immediately compatible with `jit`, `vmap`, and `grad`.

## Consequences

- Converted primitives represent the underlying infinite surface, not
  the finite face.  SDF evaluation at points far from the original face
  boundary will still return a meaningful signed distance, but it may
  not correspond to the designer's intent without CSG reconstruction.
- The five analytic surface types cover the vast majority of machined
  mechanical parts.  Freeform surfaces (NURBS) require a separate
  approximation strategy in future work.
- CSG tree reconstruction, which combines infinite primitives with
  Boolean operations to recover bounded geometry, can build directly on
  the Primitive list produced by `faces_to_primitives()`.
