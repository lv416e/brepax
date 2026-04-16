"""Conversion between OCP B-Rep entities and JAX-friendly representations."""

from __future__ import annotations

from dataclasses import dataclass, field

from brepax._occt.backend import (
    Bnd_Box,
    BRepAdaptor_Surface,
    BRepBndLib,
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_OtherSurface,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_VERTEX,
    TopExp_Explorer,
    TopoDS,
)
from brepax._occt.types import TopoDS_Shape

# Readable names for OCCT surface type enums.
_SURFACE_TYPE_NAMES: dict[object, str] = {
    GeomAbs_Plane: "planar",
    GeomAbs_Cylinder: "cylindrical",
    GeomAbs_Sphere: "spherical",
    GeomAbs_Cone: "conical",
    GeomAbs_Torus: "toroidal",
    GeomAbs_BSplineSurface: "bspline",
    GeomAbs_BezierSurface: "bezier",
    GeomAbs_OtherSurface: "other",
}


@dataclass
class ShapeMetadata:
    """Summary of a B-Rep shape's topology and geometry."""

    n_faces: int
    n_edges: int
    n_vertices: int
    face_types: dict[str, int] = field(default_factory=dict)
    bbox_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_max: tuple[float, float, float] = (0.0, 0.0, 0.0)


def shape_metadata(shape: TopoDS_Shape) -> ShapeMetadata:
    """Extract topological and geometric metadata from a shape.

    Counts faces, edges, and vertices, classifies each face by surface
    type, and computes the axis-aligned bounding box.

    Args:
        shape: An OCCT topological shape.

    Returns:
        A :class:`ShapeMetadata` summarising the shape.
    """
    n_faces = 0
    face_types: dict[str, int] = {}
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        n_faces += 1
        face = TopoDS.Face_s(explorer.Current())
        adaptor = BRepAdaptor_Surface(face)
        stype = adaptor.GetType()
        name = _SURFACE_TYPE_NAMES.get(stype, "other")
        face_types[name] = face_types.get(name, 0) + 1
        explorer.Next()

    n_edges = 0
    edge_exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_exp.More():
        n_edges += 1
        edge_exp.Next()

    n_vertices = 0
    vert_exp = TopExp_Explorer(shape, TopAbs_VERTEX)
    while vert_exp.More():
        n_vertices += 1
        vert_exp.Next()

    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

    return ShapeMetadata(
        n_faces=n_faces,
        n_edges=n_edges,
        n_vertices=n_vertices,
        face_types=face_types,
        bbox_min=(xmin, ymin, zmin),
        bbox_max=(xmax, ymax, zmax),
    )


__all__ = ["ShapeMetadata", "shape_metadata"]
