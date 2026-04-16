"""CSG tree reconstruction from B-Rep topology and analytical primitives.

Reconstructs a constructive solid geometry tree from a B-Rep shape
by classifying faces into stock (bounding box) and subtractive features
(cylindrical holes), then building a subtract tree.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax._occt.backend import (
    BRepAdaptor_Surface,
    TopAbs_FACE,
    TopExp,
    TopoDS,
    TopTools_IndexedMapOfShape,
)
from brepax._occt.types import TopoDS_Face, TopoDS_Shape
from brepax.brep.convert import face_to_primitive
from brepax.brep.topology import FaceAdjacencyGraph, build_adjacency_graph
from brepax.primitives import Box, Cylinder, FiniteCylinder, Plane
from brepax.primitives._base import Primitive

_DIRECTION_TOL = 0.01


@dataclass
class CSGLeaf:
    """Terminal node wrapping a single primitive.

    Attributes:
        primitive: The geometric primitive.
        face_ids: B-Rep face indices that contributed to this node.
    """

    primitive: Primitive
    face_ids: list[int] = field(default_factory=list)


@dataclass
class CSGOperation:
    """Binary Boolean operation combining two subtrees.

    Attributes:
        op: One of ``"union"``, ``"subtract"``, or ``"intersect"``.
        left: Left operand subtree.
        right: Right operand subtree.
    """

    op: Literal["union", "subtract", "intersect"]
    left: CSGNode
    right: CSGNode


CSGNode = CSGLeaf | CSGOperation


def reconstruct_stock_minus_features(
    shape: TopoDS_Shape,
) -> CSGNode | None:
    """Reconstruct a CSG tree assuming a stock-minus-features pattern.

    Identifies a bounding box stock from planar faces and subtractive
    cylindrical features from non-planar faces.  Returns ``None`` if the
    shape does not match the expected pattern.

    Args:
        shape: An OCCT topological shape.

    Returns:
        A CSG tree representing stock minus features, or ``None`` if the
        pattern is not recognized.

    Examples:
        >>> from brepax.io import read_step
        >>> from brepax.brep.csg import reconstruct_stock_minus_features
        >>> shape = read_step("box_with_holes.step")
        >>> tree = reconstruct_stock_minus_features(shape)
        >>> isinstance(tree, CSGOperation) and tree.op == "subtract"
        True
    """
    faces = _extract_indexed_faces(shape)
    primitives = [face_to_primitive(f) for f in faces]
    graph = build_adjacency_graph(shape)

    result = _classify_faces_and_build_box(primitives)
    if result is None:
        return None

    box, box_face_ids, feature_face_ids = result

    if not feature_face_ids:
        return CSGLeaf(primitive=box, face_ids=box_face_ids)

    feature_groups = _group_connected_faces(feature_face_ids, graph)

    feature_nodes: list[CSGLeaf] = []
    for group in feature_groups:
        node = _reconstruct_cylinder_feature(group, faces, primitives)
        if node is not None:
            feature_nodes.append(node)

    if not feature_nodes:
        return CSGLeaf(primitive=box, face_ids=box_face_ids)

    tree: CSGNode = CSGLeaf(primitive=box, face_ids=box_face_ids)
    for feat in feature_nodes:
        tree = CSGOperation(op="subtract", left=tree, right=feat)

    return tree


def _extract_indexed_faces(shape: TopoDS_Shape) -> list[TopoDS_Face]:
    """Extract faces in IndexedMap order (consistent with adjacency graph)."""
    face_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, face_map)
    return [TopoDS.Face_s(face_map.FindKey(i)) for i in range(1, face_map.Extent() + 1)]


def _classify_faces_and_build_box(
    primitives: list[Primitive | None],
) -> tuple[Box, list[int], list[int]] | None:
    """Classify faces into stock box and feature faces.

    Groups planar faces by normal direction, takes the outermost pair
    in each of three orthogonal directions to form the stock box.
    Remaining faces (non-planar and inner planes) become features.

    Returns ``(box, box_face_ids, feature_face_ids)`` or ``None``.
    """
    planar: list[tuple[int, Float[Array, 3], float]] = []
    non_planar: list[int] = []

    for i, prim in enumerate(primitives):
        if prim is None:
            continue
        if isinstance(prim, Plane):
            planar.append((i, prim.normal, float(prim.offset)))
        else:
            non_planar.append(i)

    groups = _group_planes_by_direction(planar)
    if len(groups) != 3:
        return None

    box_face_ids: list[int] = []
    inner_plane_ids: list[int] = []
    center = jnp.zeros(3)
    half_extents = jnp.zeros(3)

    for canonical_normal, group_planes in groups:
        if len(group_planes) < 2:
            return None

        sorted_planes = sorted(group_planes, key=lambda x: x[1])
        min_fid, min_offset = sorted_planes[0]
        max_fid, max_offset = sorted_planes[-1]

        box_face_ids.extend([min_fid, max_fid])

        for fid, _ in sorted_planes[1:-1]:
            inner_plane_ids.append(fid)

        half_ext = (max_offset - min_offset) / 2.0
        center_comp = (max_offset + min_offset) / 2.0

        center = center + canonical_normal * center_comp
        half_extents = half_extents + jnp.abs(canonical_normal) * half_ext

    feature_ids = sorted(non_planar + inner_plane_ids)
    box = Box(center=center, half_extents=half_extents)
    return box, sorted(box_face_ids), feature_ids


def _group_planes_by_direction(
    planes: list[tuple[int, Float[Array, 3], float]],
) -> list[tuple[Float[Array, 3], list[tuple[int, float]]]]:
    """Group planes by normal direction (parallel or antiparallel).

    Returns a list of ``(canonical_normal, [(face_id, canonical_offset), ...])``.
    Antiparallel normals are flipped to match the canonical direction.
    """
    groups: list[tuple[Float[Array, 3], list[tuple[int, float]]]] = []
    assigned: set[int] = set()

    for i, (fid_i, n_i, d_i) in enumerate(planes):
        if i in assigned:
            continue

        canonical = n_i
        group: list[tuple[int, float]] = [(fid_i, d_i)]
        assigned.add(i)

        for j, (fid_j, n_j, d_j) in enumerate(planes):
            if j in assigned:
                continue
            dot = float(jnp.dot(canonical, n_j))
            if abs(dot) > 1.0 - _DIRECTION_TOL:
                canonical_offset = d_j if dot > 0 else -d_j
                group.append((fid_j, canonical_offset))
                assigned.add(j)

        groups.append((canonical, group))

    return groups


def _group_connected_faces(
    face_ids: list[int],
    graph: FaceAdjacencyGraph,
) -> list[list[int]]:
    """Group face ids into connected components via the adjacency graph."""
    face_set = set(face_ids)
    visited: set[int] = set()
    groups: list[list[int]] = []

    for fid in face_ids:
        if fid in visited:
            continue
        group: list[int] = []
        queue = [fid]
        while queue:
            current = queue.pop(0)
            if current in visited or current not in face_set:
                continue
            visited.add(current)
            group.append(current)
            for neighbor, _ in graph.adjacency.get(current, []):
                if neighbor in face_set and neighbor not in visited:
                    queue.append(neighbor)
        if group:
            groups.append(sorted(group))

    return groups


def _reconstruct_cylinder_feature(
    face_ids: list[int],
    faces: list[TopoDS_Face],
    primitives: list[Primitive | None],
) -> CSGLeaf | None:
    """Convert a connected group of cylindrical faces to a FiniteCylinder.

    Extracts height from the OCCT face parametric V-bounds and computes
    the center position along the cylinder axis.
    """
    cyl_faces: list[tuple[int, Cylinder]] = []
    for fid in face_ids:
        prim = primitives[fid]
        if isinstance(prim, Cylinder):
            cyl_faces.append((fid, prim))

    if not cyl_faces:
        warnings.warn(
            f"Feature group {face_ids} has no cylindrical faces, skipping",
            stacklevel=2,
        )
        return None

    _, cyl = cyl_faces[0]

    v_min = float("inf")
    v_max = float("-inf")
    for fid, _ in cyl_faces:
        adaptor = BRepAdaptor_Surface(faces[fid])
        v_min = min(v_min, adaptor.FirstVParameter())
        v_max = max(v_max, adaptor.LastVParameter())

    height = v_max - v_min
    if height <= 0:
        return None

    center = cyl.point + cyl.axis * (v_min + v_max) / 2.0

    finite_cyl = FiniteCylinder(
        center=center,
        axis=cyl.axis,
        radius=cyl.radius,
        height=jnp.array(height),
    )
    return CSGLeaf(primitive=finite_cyl, face_ids=face_ids)


__all__ = [
    "CSGLeaf",
    "CSGNode",
    "CSGOperation",
    "reconstruct_stock_minus_features",
]
