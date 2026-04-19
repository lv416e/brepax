# ADR-0016: Divergence Theorem Volume

## Status

Accepted

## Context

CSG-Stump volume computation requires each face to define an infinite half-space.
BSpline patches are finite open surfaces whose SDF has no geometric meaning far
from the face boundary, causing phantom volume errors up to +31.6% (Linkrods) and
+570,000% (Linkrods with Newton divergence, prior to NaN guard).

Grid-based sigmoid integration (GWN-signed minimum distance SDF) was attempted
but failed: `abs(BSpline SDF)` returns distance to the untrimmed mathematical
surface extension, producing +219% error on Linkrods.

OCCT GProp internally uses the same divergence theorem on parametric surfaces.

## Decision

Compute volume via the divergence theorem on a triangle mesh:

```
V = (1/6) sum(v0 . (v1 x v2))
```

Triangulation uses OCCT BRepMesh for watertight mesh topology, with vertex
positions re-evaluated through JAX-native parametric surface functions so that
`jax.grad` flows from volume to surface parameters (control points, radii, etc.).

Face iteration is per-Solid (not per-Shape) to exclude orphan faces that are not
part of any closed solid.

The `gwn_signed_sdf` and `gwn_signed_volume` functions (grid-based GWN approach)
are removed as they are superseded and carry OOM risk from nested vmap.

CSG-Stump volume is retained for models with only analytical faces, but the
divergence theorem path works for all surface types.

## Consequences

Easier:

- Volume computation works for all surface types including BSpline (32 models,
  all < 0.5% error).
- Gradient `dV/dv = (1/6)(v_j x v_k)` is polynomial with no singularities
  (no arctan2, sigmoid, or grid artifacts).
- No grid resolution tuning needed for volume.

Harder:

- Mesh topology is frozen at tessellation time; large parameter perturbations
  require re-tessellation.
- BSpline face re-evaluation via vmap triggers per-face JIT compilation
  (~95 seconds for CTC-02 with 34 BSpline faces, 247K triangles).
- OCCT dependency for tessellation; pure JAX end-to-end is not possible.
