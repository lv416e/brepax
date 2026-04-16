# BRepAX

[![CI](https://github.com/lv416e/brepax/actions/workflows/ci.yaml/badge.svg)](https://github.com/lv416e/brepax/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Differentiable rasterizer for CAD Boolean operations.**

BRepAX is a JAX-native library that enables gradient-based optimization through B-Rep (Boundary Representation) topology changes. It translates contact dynamics formulations from differentiable physics into the CAD domain, providing stratum-aware differentiation for Boolean operations on geometric primitives.

## Installation

```bash
pip install brepax
```

With optional dependencies:

```bash
pip install "brepax[viz]"          # Visualization
pip install "brepax[persistence]"  # Persistent homology
pip install "brepax[all]"          # Everything
```

## Quick Start

```python
import jax
import jax.numpy as jnp
from brepax.primitives import Disk

# Create two disks
disk1 = Disk(center=jnp.array([0.0, 0.0]), radius=jnp.array(1.0))
disk2 = Disk(center=jnp.array([1.5, 0.0]), radius=jnp.array(1.0))

# Evaluate SDF -- fully differentiable
query = jnp.array([0.75, 0.0])
sdf_value = disk1.sdf(query)

# Gradients flow through everything
grad_fn = jax.grad(lambda q: disk1.sdf(q))
gradient = grad_fn(query)
```

## Roadmap

BRepAX is under active development. Current capabilities:

- **Primitives**: 8 geometric types with differentiable SDF interface
- **Boolean operations**: Union, subtract, intersect with stratum-dispatched gradients
- **STEP I/O**: Read STEP files, extract metadata, convert faces to primitives
- **Visualization**: 3D tessellated shape rendering
- **Applications**: Mold direction optimization demonstrator

Planned modules (scaffolded, not yet implemented):

- `persistence/` — Persistent homology integration
- `topology/` — Half-edge mesh representation

## Documentation

Full documentation: [lv416e.github.io/brepax](https://lv416e.github.io/brepax)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
