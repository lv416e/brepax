# BRepAX

[![CI](https://github.com/lv416e/brepax/actions/workflows/ci.yaml/badge.svg)](https://github.com/lv416e/brepax/actions/workflows/ci.yaml)
[![PyPI](https://img.shields.io/pypi/v/brepax.svg)](https://pypi.org/project/brepax/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**JAX-native differentiable geometry and STEP/B-Rep evaluation.**

BRepAX combines host-side OCCT extraction with JAX numerical evaluation of
analytical primitives and B-spline surfaces. It provides mesh-based mass
properties, parameter gradients on fixed mesh topology, and experimental
stratum-dispatched Boolean measures.

## Installation

```bash
pip install brepax
```

## Quick Start

```python
import jax
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

### Speeding up cold starts

BRepAX compiles one XLA artifact per surface type and per unique B-spline
signature on the first triangulation. On busy parts (e.g. NIST CTC-02, 664
faces) this is ~10 seconds of one-shot work per Python process. Enabling
the persistent compilation cache lets later process starts reuse those
compiled artifacts from disk:

```python
import brepax
from brepax.io.step import read_step
from brepax.brep.triangulate import triangulate_shape

brepax.enable_compilation_cache()  # defaults to ~/.cache/brepax/jax-compile

# First run populates the cache on disk, later runs load from it.
shape = read_step("part.step")
tris, _ = triangulate_shape(shape)
```

The cache directory can also be set explicitly or via the
`BREPAX_COMPILATION_CACHE_DIR` environment variable. In-process repeat
calls are unaffected — they already hit JAX's in-memory JIT cache.

## Capabilities and limits

See the [API overview](https://lv416e.github.io/brepax/reference/) for
supported STEP surfaces, the host/JAX boundary, mass-property assumptions,
and the scope of Boolean gradients. The
[integration report](https://lv416e.github.io/brepax/reference/benchmarks/)
records results on its fixture set; it is not a guarantee for arbitrary CAD.

## Documentation

Full documentation: [lv416e.github.io/brepax](https://lv416e.github.io/brepax)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
