# First Optimization

This tutorial walks through a mold direction optimization: finding the best
pull direction for an injection-molded part to minimize undercuts. BRepAX
provides a differentiable undercut metric, enabling gradient descent on the
unit sphere.

## Setup

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from brepax.experimental.applications.mold_direction import (
    optimize_mold_direction,
    undercut_volume,
)
from brepax.primitives import Box, Cylinder
```

## Define a Composite Shape

Build an L-bracket by subtracting a notch and a through-hole from a body box.
The CSG tree uses `max/min` on SDF values, and every operation is differentiable.

```python
body = Box(
    center=jnp.array([0.0, 0.0, 0.0]),
    half_extents=jnp.array([1.5, 1.0, 1.0]),
)
notch = Box(
    center=jnp.array([1.0, 0.0, 0.5]),
    half_extents=jnp.array([0.6, 1.1, 0.6]),
)
hole = Cylinder(
    point=jnp.array([0.0, 0.0, 0.0]),
    axis=jnp.array([0.0, 1.0, 0.0]),
    radius=jnp.array(0.3),
)

def bracket_sdf(x):
    """L-bracket with through-hole: body - notch - hole."""
    return jnp.maximum(jnp.maximum(body.sdf(x), -notch.sdf(x)), -hole.sdf(x))
```

## Compare Pull Directions

The undercut metric evaluates how much surface area opposes a given pull
direction. Lower values mean easier mold release.

```python
lo = jnp.array([-2.5, -2.0, -2.0])
hi = jnp.array([2.5, 2.0, 2.0])

for name, d in [("+z", jnp.array([0., 0., 1.])),
                ("-z", jnp.array([0., 0., -1.]))]:
    uc = undercut_volume(bracket_sdf, d, lo=lo, hi=hi, resolution=48)
    print(f"{name}: undercut = {float(uc):.4f}")
```

## Run the Optimizer

Start from a sub-optimal direction (into the notch) and let projected gradient
descent on the sphere find a better one.

```python
result = optimize_mold_direction(
    bracket_sdf,
    initial_direction=jnp.array([1.0, 0.0, 1.0]),
    lo=lo,
    hi=hi,
    resolution=48,
    steps=200,
    lr=0.02,
)
```

## Interpret Results

```python
d_init = jnp.array([1.0, 0.0, 1.0]) / jnp.sqrt(2.0)
d_final = result.direction

uc_initial = float(undercut_volume(bracket_sdf, d_init, lo=lo, hi=hi, resolution=48))
uc_final = float(undercut_volume(bracket_sdf, d_final, lo=lo, hi=hi, resolution=48))
reduction = (uc_initial - uc_final) / uc_initial

print(f"Initial undercut: {uc_initial:.4f}")
print(f"Final undercut:   {uc_final:.4f}")
print(f"Reduction:        {reduction:.1%}")   # ~22% reduction
print(f"Converged:        {result.converged}")
```

The optimizer rotates the pull direction away from the notch, reducing the
undercut by approximately 22%. The gradient flows through the SDF, through
the Boolean subtract, through the surface integral, and back to the direction
vector on the sphere.

## Next Steps

- [Stratum Tracking](03_stratum_tracking.md) -- how gradient dispatch works
- [API Reference](../reference/applications.md) -- mold direction API docs
