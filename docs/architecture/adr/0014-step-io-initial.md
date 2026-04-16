# ADR-0014: STEP I/O: Read and Visualization

## Status

Accepted

## Context

BRepAX operates on JAX-native SDF representations, but real-world CAD
parts are exchanged as STEP files (ISO 10303-21).  Without the ability
to ingest STEP geometry, every analysis must be built from primitives
in code -- precluding use on existing CAD models.

STEP read support enables:

- Loading industry-standard part files for face classification.
- Extracting topological metadata (face/edge/vertex counts, surface
  types, bounding boxes) as a first step toward B-Rep-to-SDF conversion.
- Tessellated 3D visualization for quick visual verification.

## Decision

Implement read-only STEP support with metadata extraction and
visualization.  Writing, modification, and gradient computation on
imported geometry are deferred to future work.

### Scope

**Included:**

- `read_step()` in `brepax.io.step` -- reads a STEP file via the OCCT
  abstraction layer (ADR-0008) and returns a `TopoDS_Shape`.
- `shape_metadata()` in `brepax.brep.convert` -- extracts face, edge,
  and vertex counts, classifies face surface types, and computes the
  axis-aligned bounding box.
- `plot_shape()` in `brepax.viz.plot3d` -- tessellates the shape with
  `BRepMesh_IncrementalMesh` and renders a 3D matplotlib plot with
  faces colored by surface type.
- Sample STEP fixtures (`sample_box.step`, `sample_cylinder.step`)
  generated programmatically from OCP primitives for deterministic
  testing.
- Unit tests for all three functions and an example notebook
  (`08_step_io_demo.py`).

**Deferred:**

- STEP write support.
- Conversion of OCCT geometry to JAX-differentiable SDF representations.
- Gradient computation through imported shapes.
- Support for assemblies or multi-body STEP files.

### Design choices

**All OCP imports in `_occt/backend.py`** per ADR-0008.  The new modules
(`io.step`, `brep.convert`, `viz.plot3d`) import exclusively from the
backend, keeping the OCCT binding swappable.

**Metadata as a plain dataclass** rather than an Equinox module.
`ShapeMetadata` contains only Python scalars and dicts, not JAX arrays,
so the PyTree registration overhead is unnecessary.

**Matplotlib for visualization** rather than a WebGL viewer.  Matplotlib
is already a project dependency (via the `viz` extra) and produces
publication-quality static figures adequate for verification.

## Consequences

- BRepAX can load and inspect real STEP parts, providing a concrete
  answer to "can I use this on my CAD files?"
- The OCCT abstraction layer (ADR-0008) gains its first substantive
  usage, validating the centralized-import design.
- Future work on B-Rep-to-SDF conversion can build on the `read_step()`
  and `shape_metadata()` foundations.
- The `viz` optional dependency is exercised beyond SDF isosurface
  plotting.
