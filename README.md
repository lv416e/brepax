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
import jax.numpy as jnp
import equinox as eqx
from brepax.io.step import read_step
from brepax.brep.csg_stump import reconstruct_csg_stump, stump_to_differentiable
from brepax.metrics import surface_area, thin_wall_volume

# Load STEP file and build differentiable representation
shape = read_step("part.step")
stump = reconstruct_csg_stump(shape)
diff = stump_to_differentiable(stump)

# Compute metrics
lo, hi = jnp.array([-1.0] * 3), jnp.array([41.0, 31.0, 21.0])
vol = diff.volume(resolution=32, lo=lo, hi=hi)
area = surface_area(diff.sdf, lo=lo, hi=hi, resolution=32)
thin = thin_wall_volume(diff.sdf, 2.0, lo=lo, hi=hi, resolution=32)

# Gradient of volume w.r.t. all design parameters
grad = eqx.filter_grad(lambda d: d.volume(resolution=16, lo=lo, hi=hi))(diff)
```

## Features

### Primitives

9 geometric types with differentiable SDF interface: Plane, Cylinder, Sphere, Cone, Torus, Box, FiniteCylinder, Disk, and **BSplineSurface** (rational NURBS with weights).

### STEP Pipeline

- Read STEP files via OCCT (cadquery-ocp-novtk)
- Convert all face types to primitives (100% conversion on 4,080 faces across 28 test files)
- PMC-based CSG-Stump reconstruction (tested up to 664 faces)
- Differentiable volume, metrics, and gradients end-to-end

### Differentiable Metrics

8 metrics, all differentiable via `jax.grad`:

| Metric | Description |
|--------|------------|
| `volume` | Sigmoid indicator integral |
| `surface_area` | Sigmoid-derivative delta function |
| `center_of_mass` | Volume-weighted position average |
| `moment_of_inertia` | Inertia tensor with Richardson extrapolation |
| `thin_wall_volume` | Volume below wall thickness threshold |
| `min_wall_thickness` | Soft-argmax with sub-grid refinement |
| `draft_angle_violation` | Surface area with insufficient draft angle |
| `undercut_volume` | Surface-weighted undercut severity |

### Boolean Operations

Union, subtract, intersect with stratum-dispatched gradients. Analytical exact gradients for bounded primitive pairs in 3 of 4 topological configurations.

## Documentation

Full documentation: [lv416e.github.io/brepax](https://lv416e.github.io/brepax)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
