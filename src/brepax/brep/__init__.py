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
from brepax.brep.csg_stump import (
    CSGStump,
    DifferentiableCSGStump,
    compact_stump,
    csg_tree_to_stump,
    evaluate_stump_sdf,
    evaluate_stump_volume,
    group_stump_primitives,
    reconstruct_csg_stump,
    stump_to_differentiable,
)
from brepax.brep.topology import (
    FaceAdjacencyGraph,
    build_adjacency_graph,
    face_degree,
    neighbors,
    shared_edges,
)
from brepax.brep.triangulate import (
    divergence_volume,
    evaluate_mesh,
    extract_mesh_topology,
    mesh_center_of_mass,
    mesh_inertia_tensor,
    mesh_surface_area,
    triangulate_shape,
)

__all__ = [
    "CSGLeaf",
    "CSGNode",
    "CSGOperation",
    "CSGStump",
    "DifferentiableCSG",
    "DifferentiableCSGStump",
    "FaceAdjacencyGraph",
    "ShapeMetadata",
    "build_adjacency_graph",
    "compact_stump",
    "csg_to_differentiable",
    "csg_tree_to_stump",
    "divergence_volume",
    "evaluate_csg_sdf",
    "evaluate_csg_volume",
    "evaluate_mesh",
    "evaluate_stump_sdf",
    "evaluate_stump_volume",
    "extract_mesh_topology",
    "face_degree",
    "face_to_primitive",
    "faces_to_primitives",
    "group_stump_primitives",
    "mesh_center_of_mass",
    "mesh_inertia_tensor",
    "mesh_surface_area",
    "neighbors",
    "reconstruct_csg_stump",
    "reconstruct_stock_minus_features",
    "shape_metadata",
    "shared_edges",
    "stump_to_differentiable",
    "triangulate_shape",
]
