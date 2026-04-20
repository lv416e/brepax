# BRepAX

[![CI](https://github.com/lv416e/brepax/actions/workflows/ci.yaml/badge.svg)](https://github.com/lv416e/brepax/actions/workflows/ci.yaml)
[![PyPI](https://img.shields.io/pypi/v/brepax.svg)](https://pypi.org/project/brepax/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**JAX-native differentiable B-Rep kernel with NURBS support.**

BRepAX loads STEP files into a JAX computation graph, enabling gradient-based optimization of CAD geometry through Boolean operations. It provides stratum-aware differentiation that handles topological transitions at Boolean boundaries, and supports both analytical primitives and freeform B-spline surfaces.

## Installation

```bash
pip install brepax
```

## Quick Start

```python
import jax
import jax.numpy as jnp
from brepax.io.step import read_step
from brepax.brep.triangulate import (
    triangulate_shape, divergence_volume,
    mesh_surface_area, mesh_center_of_mass,
)

# Load STEP file and compute volume via divergence theorem
shape = read_step("part.step")
tris, params = triangulate_shape(shape)

vol = divergence_volume(tris)           # exact for watertight mesh
area = mesh_surface_area(tris)          # sum of triangle areas
com = mesh_center_of_mass(tris)         # surface integral (Eberly 2002)

# Gradient of volume w.r.t. all triangle vertices
grad = jax.grad(divergence_volume)(tris)
```

### Parametric Optimization

```python
from brepax.brep.triangulate import extract_mesh_topology, evaluate_mesh

# Separate topology (one-time) from evaluation (differentiable)
topology = extract_mesh_topology(shape)

def volume_fn(radius):
    tris = evaluate_mesh(topology, {"radius": radius}, uv_scale_param="radius")
    return divergence_volume(tris)

# Gradient flows from volume through vertices to design parameter
grad = jax.grad(volume_fn)(jnp.array(5.0))
```

### Speeding up cold starts

BRepAX compiles one XLA artifact per surface type and per unique B-spline
signature on the first triangulation. On busy parts (e.g. NIST CTC-02, 664
faces) this is ~10 seconds of one-shot work per Python process. Enabling
the persistent compilation cache lets later process starts reuse those
compiled artifacts from disk:

```python
import brepax

brepax.enable_compilation_cache()  # defaults to ~/.cache/brepax/jax-compile

# First run populates the cache on disk, later runs load from it.
shape = read_step("part.step")
tris, _ = triangulate_shape(shape)
```

The cache directory can also be set explicitly or via the
`BREPAX_COMPILATION_CACHE_DIR` environment variable. In-process repeat
calls are unaffected — they already hit JAX's in-memory JIT cache.

## Features

### Primitives

9 geometric types with differentiable SDF interface: Plane, Cylinder, Sphere, Cone, Torus, Box, FiniteCylinder, Disk, and **BSplineSurface** (rational NURBS with weights).

### STEP Pipeline

- Read STEP files via OCCT (cadquery-ocp-novtk)
- Convert all face types to primitives (100% conversion on 4,080 faces across 28 test files)
- PMC-based CSG-Stump reconstruction (tested up to 664 faces)
- OCCT mesh hybrid triangulation with JAX-native vertex re-evaluation

### Volume and Mass Properties (Divergence Theorem)

Mesh-based computation via the divergence theorem, working for all surface
types including freeform B-spline. Validated on 32 models (< 0.5% error
vs OCCT GProp). All are polynomial in vertex positions, giving exact
gradients with no grid artifacts or singularities.

| Function | Formula | Degree |
|----------|---------|--------|
| `divergence_volume` | `(1/6) sum(v0 . (v1 x v2))` | 3 |
| `mesh_surface_area` | `(1/2) sum(norm(cross(e1, e2)))` | -- |
| `mesh_center_of_mass` | First moments via surface integral | 4 |
| `mesh_inertia_tensor` | Second moments (Tonon 2004) | 5 |

### Differentiable Metrics

10 metrics, all differentiable via `jax.grad`. 8 of 10 work for all surface
types; wall thickness metrics require analytical surfaces via CSG-Stump.

| Metric | Method | BSpline |
|--------|--------|:-------:|
| `divergence_volume` | Divergence theorem on mesh | Yes |
| `mesh_surface_area` | Triangle area sum | Yes |
| `mesh_center_of_mass` | Divergence theorem variant | Yes |
| `mesh_inertia_tensor` | Divergence theorem variant | Yes |
| `draft_angle_violation` | SDF gradient near surface | Yes |
| `mean_curvature` / `max_curvature` | AD Hessian of SDF | Yes |
| `thin_wall_volume` | Sigmoid indicator on SDF grid | Analytical only |
| `min_wall_thickness` | Soft-argmax on SDF grid | Analytical only |

### Parametric Optimization

`extract_mesh_topology` + `evaluate_mesh` separate watertight mesh topology
(from OCCT, one-time) from vertex evaluation (JAX-native, differentiable).
Design parameters flow through surface evaluation to volume:

- **Sphere radius**: Newton convergence in 4 steps
- **Cylinder radius**: Multi-face with disk cap tracking, 4 steps
- **BSpline control points**: Exact gradient, Newton convergence in 1 step

### Boolean Operations

Union, subtract, intersect with stratum-dispatched gradients. Analytical exact gradients for bounded primitive pairs in 3 of 4 topological configurations.

## Documentation

Full documentation: [lv416e.github.io/brepax](https://lv416e.github.io/brepax)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
