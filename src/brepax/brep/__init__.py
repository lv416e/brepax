"""B-Rep bridge layer: OCP-to-JAX conversions for faces, edges, and vertices."""

from brepax.brep.convert import (
    ShapeMetadata,
    face_to_primitive,
    faces_to_primitives,
    shape_metadata,
)
from brepax.brep.topology import (
    FaceAdjacencyGraph,
    build_adjacency_graph,
    face_degree,
    neighbors,
    shared_edges,
)

__all__ = [
    "FaceAdjacencyGraph",
    "ShapeMetadata",
    "build_adjacency_graph",
    "face_degree",
    "face_to_primitive",
    "faces_to_primitives",
    "neighbors",
    "shape_metadata",
    "shared_edges",
]
