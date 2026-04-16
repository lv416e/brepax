# Quickstart

Get started with BRepAX in 5 minutes. This tutorial covers creating primitives,
evaluating signed distance functions, computing volumes, and differentiating
through everything with JAX.

## Installation

```bash
pip install brepax
```

## Setup

Enable 64-bit precision for accurate SDF computation:

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from brepax.primitives import Disk, Sphere
```

## 2D: Create a Disk

A `Disk` is a 2D primitive defined by a center and radius. Its SDF returns
negative values inside, zero on the boundary, and positive values outside.

```python
disk = Disk(center=jnp.array([0.0, 0.0]), radius=jnp.array(1.0))

print(disk.sdf(jnp.array([0.5, 0.0])))   # -0.5 (inside)
print(disk.sdf(jnp.array([1.0, 0.0])))   #  0.0 (boundary)
print(disk.sdf(jnp.array([2.0, 0.0])))   #  1.0 (outside)
```

## 3D: Create a Sphere

A `Sphere` works the same way in 3D and also provides an analytical `volume()`.

```python
sphere = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(1.0))

print(sphere.sdf(jnp.array([0.0, 0.0, 0.0])))  # -1.0 (center)
print(sphere.sdf(jnp.array([1.0, 0.0, 0.0])))  #  0.0 (surface)
print(sphere.volume())                           #  4.1888 (4/3 pi)
```

## Differentiate through SDF

JAX can differentiate through any BRepAX operation. Compute the gradient of
SDF with respect to the query point:

```python
grad_fn = jax.grad(lambda x: disk.sdf(x))
print(grad_fn(jnp.array([2.0, 0.0])))  # [1.0, 0.0]
```

Differentiate with respect to primitive parameters using Equinox:

```python
import equinox as eqx

grad_disk = eqx.filter_grad(lambda d: d.sdf(jnp.array([2.0, 0.0])))(disk)
print(grad_disk.radius)  # -1.0 (increasing radius decreases SDF outside)
```

## Boolean Operations

Combine two disks with a union and differentiate through the result:

```python
from brepax.boolean import union_area

disk_a = Disk(center=jnp.array([0.0, 0.0]), radius=jnp.array(1.0))
disk_b = Disk(center=jnp.array([1.5, 0.0]), radius=jnp.array(1.0))

area = union_area(disk_a, disk_b, method="stratum")
print(f"Union area: {area:.4f}")  # ~5.0985
```

The gradient of union area with respect to radius flows through the Boolean:

```python
grad = jax.grad(
    lambda r: union_area(
        Disk(center=jnp.array([0.0, 0.0]), radius=r),
        disk_b,
        method="stratum",
    )
)(jnp.array(1.0))
print(f"d(union_area)/d(r1): {grad:.4f}")
```

## Next Steps

- [First Optimization](02_first_optimization.md) -- optimize a mold pull direction
- [Stratum Tracking](03_stratum_tracking.md) -- understand gradient dispatch
- [API Reference](../reference/index.md) -- full module documentation
