"""3D tessellated B-Rep and SDF isosurface visualization."""

from __future__ import annotations

import numpy as np

from brepax._occt.backend import (
    BRep_Tool,
    BRepAdaptor_Surface,
    BRepMesh_IncrementalMesh,
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_OtherSurface,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    TopAbs_FACE,
    TopExp_Explorer,
    TopLoc_Location,
    TopoDS,
)
from brepax._occt.types import TopoDS_Shape

# Face type to color mapping for visualization.
_FACE_TYPE_COLORS: dict[object, str] = {
    GeomAbs_Plane: "#4e79a7",
    GeomAbs_Cylinder: "#f28e2b",
    GeomAbs_Sphere: "#e15759",
    GeomAbs_Cone: "#76b7b2",
    GeomAbs_Torus: "#59a14f",
    GeomAbs_BSplineSurface: "#edc948",
    GeomAbs_BezierSurface: "#b07aa1",
    GeomAbs_OtherSurface: "#aaaaaa",
}

_DEFAULT_FACE_COLOR = "#aaaaaa"


def plot_shape(
    shape: TopoDS_Shape,
    *,
    face_colors: bool = True,
    linear_deflection: float = 0.1,
) -> None:
    """Visualize a tessellated B-Rep shape using matplotlib.

    Tessellates the shape with :class:`BRepMesh_IncrementalMesh` and
    renders each face as a :class:`Poly3DCollection` colored by surface
    type.

    Args:
        shape: The OCCT shape to visualize.
        face_colors: Color faces by surface type when ``True``.
        linear_deflection: Tessellation quality (smaller = finer mesh).
    """
    # Lazy import keeps matplotlib optional at module level.
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    BRepMesh_IncrementalMesh(shape, linear_deflection)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    all_pts: list[np.ndarray[tuple[int], np.dtype[np.float64]]] = []

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        explorer.Next()

        if tri is None:
            continue

        trsf = loc.Transformation()
        identity = loc.IsIdentity()

        # Extract node coordinates, applying location transform.
        n_nodes = tri.NbNodes()
        nodes = np.empty((n_nodes, 3))
        for i in range(1, n_nodes + 1):
            pt = tri.Node(i)
            if not identity:
                pt.Transform(trsf)
            nodes[i - 1] = (pt.X(), pt.Y(), pt.Z())

        all_pts.append(nodes)

        # Build triangle polygons (1-indexed in OCCT).
        n_tri = tri.NbTriangles()
        polys = []
        for i in range(1, n_tri + 1):
            t = tri.Triangle(i)
            n1, n2, n3 = t.Get()
            polys.append([nodes[n1 - 1], nodes[n2 - 1], nodes[n3 - 1]])

        if face_colors:
            adaptor = BRepAdaptor_Surface(face)
            stype = adaptor.GetType()
            color = _FACE_TYPE_COLORS.get(stype, _DEFAULT_FACE_COLOR)
        else:
            color = "#4e79a7"

        collection = Poly3DCollection(
            polys, alpha=0.7, facecolor=color, edgecolor="#333333", linewidth=0.3
        )
        ax.add_collection3d(collection)

    # Set axis limits from all collected points.
    if all_pts:
        combined = np.concatenate(all_pts, axis=0)
        mins = combined.min(axis=0)
        maxs = combined.max(axis=0)
        center = (mins + maxs) / 2.0
        extent = (maxs - mins).max() / 2.0 * 1.1
        ax.set_xlim(center[0] - extent, center[0] + extent)
        ax.set_ylim(center[1] - extent, center[1] + extent)
        ax.set_zlim(center[2] - extent, center[2] + extent)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("B-Rep Shape")
    plt.tight_layout()
    plt.show()


__all__ = ["plot_shape"]
