"""B-Rep bridge layer: OCP-to-JAX conversions for faces, edges, and vertices."""

from brepax.brep.convert import (
    ShapeMetadata,
    face_to_primitive,
    faces_to_primitives,
    shape_metadata,
)

__all__ = [
    "ShapeMetadata",
    "face_to_primitive",
    "faces_to_primitives",
    "shape_metadata",
]
