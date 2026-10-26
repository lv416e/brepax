# BRepAX

**JAX-native differentiable geometry and STEP/B-Rep evaluation.**

BRepAX combines OCCT preparation with JAX numerical kernels for CAD
experiments and geometry evaluation. Start with the
[API overview](reference/index.md#capabilities-and-limits) for supported
inputs, differentiation assumptions, and known limits.

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
