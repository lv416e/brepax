# Why B-Rep

Why BRepAX operates on true Boundary Representation geometry rather than implicit (SDF) representations alone.

## The Limitation of Implicit Representations

SDF-based and implicit shape representations are powerful for differentiable geometry, but they cannot capture:

- **Parametric surfaces**: NURBS, Bezier patches, and analytic surfaces (planes, cylinders, cones, tori) with exact evaluation
- **Trim curves**: The boundaries that define which portion of a surface is active in a face
- **Tolerance handling**: Manufacturing tolerances that determine whether edges are coincident or distinct
- **Face/edge/vertex incidence**: The combinatorial structure that encodes how geometric entities connect

These are precisely the structures that manufacturing cost analysis depends on (undercuts, wall thickness, rib detection, mold direction, bend analysis).

## Existing Work in the Implicit Space

Several projects already address differentiable implicit geometry:

- Marschner et al.'s pseudo-SDF correction
- DreamCAD and related neural implicit CAD methods
- MeshSDF and DMTet for differentiable mesh extraction

BRepAX's differentiation from these approaches comes from operating directly on B-Rep topology, not from JAX-nativeness alone.

## OCCT as the Foundation

Open CASCADE Technology (OCCT) is the industry-standard open-source B-Rep kernel, used by FreeCAD, CadQuery, build123d, and many commercial systems. By building on OCCT, BRepAX gains:

- Exact parametric surface evaluation (no tessellation artifacts)
- Standard STEP file I/O for interoperability with industrial CAD systems
- Robust Boolean operations as a reference implementation
- Access to the full vocabulary of B-Rep topology
