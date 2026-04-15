# ADR-0007: OCCT (cadquery-ocp-novtk) as Core Dependency

## Status

Accepted

## Context

BRepAX's core identity is a differentiable kernel for **real B-Rep operations**, not just implicit/SDF representations. This requires access to OCCT (Open CASCADE Technology) for parametric surface evaluation, trim curves, tolerance handling, and STEP I/O. Without OCCT, BRepAX cannot operate on true B-Rep geometry and becomes indistinguishable from existing implicit shape libraries.

The original spec listed `pythonocc-core` as an optional dependency, but it is only distributed via conda-forge. The actively maintained 7.9.x is not available on PyPI, making it incompatible with a `uv`/pip-based project.

`cadquery-ocp-novtk` is a CadQuery-maintained OCCT Python wrapper distributed on PyPI with wheels for Python 3.10-3.14 on Linux, macOS (Intel + ARM), and Windows.

## Decision

Use `cadquery-ocp-novtk` as a **core dependency** (not optional extras).

Reasons:

- **Core identity**: BRepAX handles real B-Rep geometry; OCCT is the industry-standard kernel for this
- **PyPI availability**: Installable via `pip install` / `uv add`, no conda required
- **Platform coverage**: Wheels for all supported platforms including Apple Silicon
- **Lightweight**: The `novtk` variant excludes VTK (visualization uses matplotlib/plotly instead)
- **Industry precedent**: CadQuery and build123d use the same Apache 2.0 wrapper + LGPL OCCT structure without adoption issues

Trade-offs:

- **LGPL transitive dependency**: OCCT is LGPL 2.1. BRepAX itself remains Apache 2.0. Dynamic linking satisfies LGPL obligations for typical use cases (academic, personal, OSS, commercial internal tools). Closed-source redistribution requires maintaining OCCT replaceability.
- **Wheel size**: 80-150MB (comparable to JAX, much smaller than PyTorch CUDA builds)
- **API style**: Imports use `OCP.STEPControl` rather than pythonocc's `OCC.Core.STEPControl`

## Consequences

- `pip install brepax` includes OCCT out of the box
- `src/brepax/brep/` provides the OCP-to-JAX bridge layer
- `src/brepax/io/step.py` uses OCP imports for STEP I/O
- README and docs must clearly document the LGPL transitive dependency
- THIRD_PARTY_NOTICES.md documents OCCT licensing
