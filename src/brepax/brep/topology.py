"""Face adjacency graph extracted from B-Rep topology.

Uses OCCT's TopExp.MapShapesAndAncestors to build an edge-to-face
incidence map, then derives face-level adjacency from shared edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from brepax._occt.backend import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopExp,
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_IndexedMapOfShape,
)
from brepax._occt.types import TopoDS_Shape


@dataclass
class FaceAdjacencyGraph:
    """Graph of face-to-face connections through shared edges.

    Faces and edges are 0-indexed.  Each entry in ``adjacency``
    maps a face id to a list of ``(neighbor_face_id, edge_id)``
    pairs.  Seam edges (where a face is adjacent to itself) are
    included.

    Attributes:
        n_faces: Number of topological faces.
        n_edges: Number of topological edges.
        adjacency: Mapping from face id to list of (neighbor, edge) pairs.
    """

    n_faces: int
    n_edges: int
    adjacency: dict[int, list[tuple[int, int]]] = field(default_factory=dict)


def build_adjacency_graph(shape: TopoDS_Shape) -> FaceAdjacencyGraph:
    """Build a face adjacency graph from a B-Rep shape.

    Iterates over all edges and records which faces share each edge.
    Seam edges (same face on both sides) are included in the result.

    Args:
        shape: An OCCT topological shape.

    Returns:
        A :class:`FaceAdjacencyGraph` with 0-indexed face and edge ids.

    Examples:
        >>> from brepax.io import read_step
        >>> from brepax.brep import build_adjacency_graph
        >>> shape = read_step("part.step")
        >>> graph = build_adjacency_graph(shape)
        >>> neighbors(graph, 0)
        [1, 2, 3, 4]
    """
    face_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, face_map)

    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    n_faces = face_map.Extent()
    n_edges = edge_face_map.Extent()
    adjacency: dict[int, list[tuple[int, int]]] = {fi: [] for fi in range(n_faces)}

    for ei_ocp in range(1, n_edges + 1):
        ei = ei_ocp - 1
        face_list = edge_face_map.FindFromIndex(ei_ocp)
        face_ids = [face_map.FindIndex(f) - 1 for f in face_list]

        for i in range(len(face_ids)):
            for j in range(i + 1, len(face_ids)):
                a, b = face_ids[i], face_ids[j]
                adjacency[a].append((b, ei))
                if a != b:
                    adjacency[b].append((a, ei))

    return FaceAdjacencyGraph(
        n_faces=n_faces,
        n_edges=n_edges,
        adjacency=adjacency,
    )


def neighbors(graph: FaceAdjacencyGraph, face_id: int) -> list[int]:
    """Return distinct neighbor face ids for a given face.

    Args:
        graph: A face adjacency graph.
        face_id: 0-indexed face identifier.

    Returns:
        Sorted list of unique neighbor face ids.
    """
    return sorted({n for n, _ in graph.adjacency.get(face_id, [])})


def shared_edges(graph: FaceAdjacencyGraph, face_a: int, face_b: int) -> list[int]:
    """Return edge ids shared between two faces.

    Args:
        graph: A face adjacency graph.
        face_a: 0-indexed face identifier.
        face_b: 0-indexed face identifier.

    Returns:
        Sorted list of edge ids connecting the two faces.
    """
    return sorted({e for n, e in graph.adjacency.get(face_a, []) if n == face_b})


def face_degree(graph: FaceAdjacencyGraph, face_id: int) -> int:
    """Return the number of distinct faces adjacent to a given face.

    Args:
        graph: A face adjacency graph.
        face_id: 0-indexed face identifier.

    Returns:
        Number of distinct neighbor faces (including self for seam edges).
    """
    return len({n for n, _ in graph.adjacency.get(face_id, [])})


__all__ = [
    "FaceAdjacencyGraph",
    "build_adjacency_graph",
    "face_degree",
    "neighbors",
    "shared_edges",
]
