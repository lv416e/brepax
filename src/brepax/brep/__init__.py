"""B-Rep bridge layer: OCP-to-JAX conversions for faces, edges, and vertices."""

from brepax.brep.convert import (
    ShapeMetadata,
    face_to_primitive,
    faces_to_primitives,
    shape_metadata,
)
from brepax.brep.csg import (
    CSGLeaf,
    CSGNode,
    CSGOperation,
    reconstruct_stock_minus_features,
)
from brepax.brep.csg_eval import (
    DifferentiableCSG,
    csg_to_differentiable,
    evaluate_csg_sdf,
    evaluate_csg_volume,
)
from brepax.brep.topology import (
    FaceAdjacencyGraph,
    build_adjacency_graph,
    face_degree,
    neighbors,
    shared_edges,
)

__all__ = [
    "CSGLeaf",
    "CSGNode",
    "CSGOperation",
    "DifferentiableCSG",
    "FaceAdjacencyGraph",
    "ShapeMetadata",
    "build_adjacency_graph",
    "csg_to_differentiable",
    "evaluate_csg_sdf",
    "evaluate_csg_volume",
    "face_degree",
    "face_to_primitive",
    "faces_to_primitives",
    "neighbors",
    "reconstruct_stock_minus_features",
    "shape_metadata",
    "shared_edges",
]
