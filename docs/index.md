# BRepAX

**Differentiable rasterizer for CAD Boolean operations.**

BRepAX is a JAX-native library that brings differentiable computation to B-Rep
CAD operations. It translates contact dynamics formulations from differentiable
physics into the CAD domain, enabling gradient-based optimization through
topology changes where edges and faces appear or disappear.

## Key Features

- **9 geometric primitives** -- Disk, Sphere, Box, Cylinder, Cone, Torus, FiniteCylinder, Plane, BSplineSurface, all with SDF interface
- **Divergence theorem volume** -- exact mesh-based volume computation for all surface types including freeform B-spline (< 0.5% on 32 validated models)
- **Mesh-based mass properties** -- surface area, center of mass, and inertia tensor via surface integrals, polynomial and singularity-free
- **Parametric optimization** -- design parameters (radius, control points) flow through mesh vertices to volume via `jax.grad`
- **Differentiable Boolean operations** -- union, intersection, subtraction with gradients that flow through topology changes
- **Stratum-dispatched gradients** -- exact analytical gradients via topological classification, not smoothing approximations
- **Full JAX compatibility** -- every operation works with `jit`, `vmap`, and `grad`
- **Equinox PyTree integration** -- primitives are `equinox.Module` instances, composable with the JAX ecosystem

## Quick Install

```bash
pip install brepax
```

## Minimal Example

```python
import jax
from brepax.io.step import read_step
from brepax.brep.triangulate import triangulate_shape, divergence_volume

shape = read_step("part.step")
tris, params = triangulate_shape(shape)
vol = divergence_volume(tris)
grad = jax.grad(divergence_volume)(tris)  # d(volume)/d(vertices)
```

## Documentation

- [Quickstart Tutorial](tutorials/01_quickstart.md) -- create primitives, evaluate SDF, differentiate
- [First Optimization](tutorials/02_first_optimization.md) -- optimize a mold pull direction
- [Stratum Tracking](tutorials/03_stratum_tracking.md) -- understand gradient dispatch
- [API Reference](reference/index.md) -- full module documentation
