# BRepAX

**Differentiable rasterizer for CAD Boolean operations.**

BRepAX is a JAX-native library that brings differentiable computation to B-Rep
CAD operations. It translates contact dynamics formulations from differentiable
physics into the CAD domain, enabling gradient-based optimization through
topology changes where edges and faces appear or disappear.

## Key Features

- **8 geometric primitives** -- Disk, Sphere, Box, Cylinder, Cone, Torus, FiniteCylinder, Plane, all with SDF interface
- **Differentiable Boolean operations** -- union, intersection, subtraction with gradients that flow through topology changes
- **Stratum-dispatched gradients** -- exact analytical gradients via topological classification, not smoothing approximations
- **Full JAX compatibility** -- every operation works with `jit`, `vmap`, and `grad`
- **Equinox PyTree integration** -- primitives are `equinox.Module` instances, composable with the JAX ecosystem
- **Application demonstrators** -- mold direction optimizer built on top of the core library

## Quick Install

```bash
pip install brepax
```

## Minimal Example

```python
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from brepax.primitives import Disk

disk = Disk(center=jnp.array([0.0, 0.0]), radius=jnp.array(1.0))
print(disk.sdf(jnp.array([2.0, 0.0])))          # 1.0
print(jax.grad(lambda x: disk.sdf(x))(jnp.array([2.0, 0.0])))  # [1. 0.]
```

## Documentation

- [Quickstart Tutorial](tutorials/01_quickstart.md) -- create primitives, evaluate SDF, differentiate
- [First Optimization](tutorials/02_first_optimization.md) -- optimize a mold pull direction
- [Stratum Tracking](tutorials/03_stratum_tracking.md) -- understand gradient dispatch
- [API Reference](reference/index.md) -- full module documentation
