# Stratum Tracking

BRepAX classifies the topological relationship between two primitives into
discrete *strata* and dispatches different gradient formulas for each. This
tutorial explains why that matters and demonstrates the behavior.

## Setup

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from brepax.analytical.disk_disk import (
    disk_disk_boundary_distance,
    disk_disk_stratum_label,
    disk_disk_union_area,
)
from brepax.boolean import union_area
from brepax.primitives import Disk
```

## What Are Strata?

For two disks with radii r1, r2 and center distance d, three strata exist:

| Label | Condition | Relationship |
|-------|-----------|--------------|
| 0 | d >= r1 + r2 | Disjoint |
| 1 | \|r1 - r2\| < d < r1 + r2 | Intersecting |
| 2 | d <= \|r1 - r2\| | Contained |

Each stratum has a different analytical formula for the union area and its
gradient. At stratum boundaries (where the topology changes), naive
autodiff produces incorrect gradients.

```python
c1, r1, r2 = jnp.array([0.0, 0.0]), jnp.array(1.5), jnp.array(0.5)

for name, d_val in [("disjoint", 5.0), ("intersecting", 1.5), ("contained", 0.3)]:
    c2 = jnp.array([d_val, 0.0])
    label = int(disk_disk_stratum_label(c1, r1, c2, r2))
    bdist = float(disk_disk_boundary_distance(c1, r1, c2, r2))
    print(f"{name:>13}: label={label}, boundary_dist={bdist:.2f}")
```

## Gradient Behavior per Stratum

The stratum-aware method dispatches the correct gradient formula for each
regime, producing exact analytical gradients.

```python
for name, d_val in [("disjoint", 5.0), ("intersecting", 1.5), ("contained", 0.3)]:
    c2 = jnp.array([d_val, 0.0])
    b = Disk(center=c2, radius=r2)

    grad_r = jax.grad(
        lambda r: union_area(Disk(center=c1, radius=r), b, method="stratum")
    )(r1)

    grad_c = jax.grad(
        lambda cx: union_area(
            Disk(center=c1, radius=r1),
            Disk(center=jnp.array([cx, 0.0]), radius=r2),
            method="stratum",
        )
    )(jnp.array(d_val))

    print(f"{name:>13}: d(area)/d(r1)={float(grad_r):.4f}, "
          f"d(area)/d(c2_x)={float(grad_c):.6f}")
```

## Boundary Proximity: Method (A) vs Method (C)

As two unit disks approach the external tangent boundary (d approaches r1 + r2),
Method (A) (smoothing-based) accumulates error while Method (C) (stratum-aware)
maintains precision.

```python
r1_test, r2_test = jnp.array(1.0), jnp.array(1.0)

for d_val in [1.5, 1.9, 1.99, 1.999]:
    c2 = jnp.array([d_val, 0.0])
    b = Disk(center=c2, radius=r2_test)

    exact = float(jax.grad(disk_disk_union_area, argnums=1)(
        jnp.array([0.0, 0.0]), r1_test, c2, r2_test,
    ))

    ga = float(jax.grad(lambda r: union_area(
        Disk(center=jnp.array([0.0, 0.0]), radius=r), b,
        method="smoothing", k=0.1, beta=0.1, resolution=128,
    ))(r1_test))

    gc = float(jax.grad(lambda r: union_area(
        Disk(center=jnp.array([0.0, 0.0]), radius=r), b,
        method="stratum",
    ))(r1_test))

    bdist = 2.0 - d_val
    err_a = abs(ga - exact) / max(abs(exact), 1e-12)
    err_c = abs(gc - exact) / max(abs(exact), 1e-12)
    print(f"boundary_dist={bdist:.3f}: "
          f"Method A err={err_a:.4%}, Method C err={err_c:.4%}")
```

As the boundary distance shrinks toward zero, Method (A) error grows while
Method (C) remains near machine precision. This is the core advantage of
stratum-dispatched gradients.

## Next Steps

- [Quickstart](01_quickstart.md) -- basics of primitives and SDF
- [API Reference](../reference/boolean.md) -- Boolean operations API docs
