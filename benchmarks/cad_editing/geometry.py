"""Supplied CAD family and independent integral reference, in working units.

The roof is z = 10 + a*4*(x/40)*(1-x/40), extruded over y=[0,20],
with a vertical through-hole centred at (20,10). Parameters are the roof
amplitude a and hole radius r. The tested edit box is a=[3.5,6.5], r=[2.25,3.75].
This is a supplied construction, not recovery of a design history from STEP.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from OCP.BRep import BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_NurbsConvert,
)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism
from OCP.Geom import Geom_BezierCurve
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS

from brepax._occt.types import TopoDS_Shape

LENGTH, WIDTH, HEIGHT = 40.0, 20.0, 10.0
BASE_DIMENSIONS = np.array([5.0, 3.0])
EDIT_POINTS = BASE_DIMENSIONS * np.array([[0.8, 0.85], [1.0, 1.0], [1.2, 1.15]])


def make_shape(dimensions: NDArray[np.float64]) -> TopoDS_Shape:
    """Construct and cut only AFTER external dimensions have been normalized."""
    amplitude, radius = dimensions
    if not (3.5 <= amplitude <= 6.5 and 2.25 <= radius <= 3.75):
        raise ValueError("Dimensions outside the supplied fixed-topology edit box")
    points = [
        gp_Pnt(0, 0, 0),
        gp_Pnt(LENGTH, 0, 0),
        gp_Pnt(LENGTH, 0, HEIGHT),
        gp_Pnt(0, 0, HEIGHT),
    ]
    poles = TColgp_Array1OfPnt(1, 3)
    for i, point in enumerate(
        [points[2], gp_Pnt(LENGTH / 2, 0, HEIGHT + 2 * amplitude), points[3]], 1
    ):
        poles.SetValue(i, point)
    wire = BRepBuilderAPI_MakeWire()
    for edge in (
        BRepBuilderAPI_MakeEdge(points[0], points[1]),
        BRepBuilderAPI_MakeEdge(points[1], points[2]),
        BRepBuilderAPI_MakeEdge(Geom_BezierCurve(poles)),
        BRepBuilderAPI_MakeEdge(points[3], points[0]),
    ):
        wire.Add(edge.Edge())
    profile = BRepBuilderAPI_MakeFace(wire.Wire()).Face()
    stock = BRepPrimAPI_MakePrism(profile, gp_Vec(0, WIDTH, 0)).Shape()
    # Convert stock before cutting. Converting the cylinder to a periodic
    # B-spline would exceed the current surface extraction contract.
    stock = BRepBuilderAPI_NurbsConvert(stock, True).Shape()
    tool = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(LENGTH / 2, WIDTH / 2, -1), gp_Dir(0, 0, 1)),
        float(radius),
        float(HEIGHT + amplitude + 2),
    ).Shape()
    operation = BRepAlgoAPI_Cut(stock, tool)
    if not operation.IsDone():
        raise RuntimeError("CAD Boolean cut failed")
    shape = operation.Shape()
    if shape.IsNull() or not BRepCheck_Analyzer(shape).IsValid():
        raise RuntimeError("Invalid rebuilt CAD solid")
    return shape


def cad_measures(shape: TopoDS_Shape) -> NDArray[np.float64]:
    """Adaptive OCCT [volume, area] integration, independent of the mesh."""
    volume, area = GProp_GProps(), GProp_GProps()
    # Default non-adaptive GProp integration is too noisy for this FD oracle.
    BRepGProp.VolumeProperties_s(shape, volume, 1e-10)
    BRepGProp.SurfaceProperties_s(shape, area, 1e-10)
    return np.array([volume.Mass(), area.Mass()])


def boundary_tolerance(shape: TopoDS_Shape) -> float:
    """Largest actual OCCT edge/vertex tolerance in working length units."""
    values = []
    for kind, cast in ((TopAbs_EDGE, TopoDS.Edge_s), (TopAbs_VERTEX, TopoDS.Vertex_s)):
        explorer = TopExp_Explorer(shape, kind)
        while explorer.More():
            values.append(BRep_Tool.Tolerance_s(cast(explorer.Current())))
            explorer.Next()
    return float(max(values))


def bottom_landmark_error(
    shape: TopoDS_Shape, expected: NDArray[np.float64], length_scale: float
) -> float:
    """Largest distance to known unchanged bottom corners, in working units."""
    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    vertices = []
    while explorer.More():
        point = BRep_Tool.Pnt_s(TopoDS.Vertex_s(explorer.Current()))
        vertices.append([point.X(), point.Y(), point.Z()])
        explorer.Next()
    distances = np.linalg.norm(expected[:, None] - np.asarray(vertices), axis=2)
    error = float(distances.min(axis=1).max() / length_scale)
    if not np.isfinite(error):
        raise FloatingPointError("Nonfinite CAD landmark error")
    return error


def reference_measures(
    dimensions: NDArray[np.float64] | NDArray[np.complex128],
) -> NDArray[np.float64] | NDArray[np.complex128]:
    """Independent graph/disk integrals; complex inputs support complex-step AD.

    The roof and removed circular patch use 96-point Gauss-Legendre quadrature.
    The prism section, removed volume, and cylindrical wall use closed formulas.
    No CAD, shared mesh or JAX evaluation participates in this reference.
    """
    amplitude, radius = dimensions
    nodes, weights = np.polynomial.legendre.leggauss(96)
    arc = (
        LENGTH / 2 * np.dot(weights, np.sqrt(1 + (4 * amplitude * nodes / LENGTH) ** 2))
    )
    section = LENGTH * (HEIGHT + 2 * amplitude / 3)
    volume = WIDTH * section
    area = WIDTH * (LENGTH + 2 * HEIGHT + arc) + 2 * section
    removed_volume = (
        np.pi * radius**2 * (HEIGHT + amplitude * (1 - radius**2 / LENGTH**2))
    )
    phi = nodes * np.pi / 2
    removed_roof = (
        np.pi
        * radius**2
        * np.dot(
            weights,
            np.cos(phi) ** 2
            * np.sqrt(1 + (8 * amplitude * radius * np.sin(phi) / LENGTH**2) ** 2),
        )
    )
    wall = 2 * np.pi * radius * (HEIGHT + amplitude * (1 - 2 * radius**2 / LENGTH**2))
    return np.array(
        [volume - removed_volume, area - np.pi * radius**2 - removed_roof + wall]
    )


def reference_jacobian(dimensions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Complex-step derivatives of the independent integrals w.r.t. lengths."""
    return np.stack(
        [
            np.imag(reference_measures(dimensions.astype(complex) + 1e-25j * direction))
            / 1e-25
            for direction in np.eye(2)
        ],
        axis=1,
    )
