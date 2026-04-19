"""Thin wrapper around cadquery-ocp-novtk providing the OCCT subset BRepAX needs.

All OCP imports are centralized here so that swapping the underlying
OCCT binding (e.g., to a custom pybind11 wrapper) requires changing
only this module.
"""

from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve2d, BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
from OCP.GeomAbs import (
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_OtherSurface,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCP.gp import gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_IN,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_IndexedMapOfShape,
)

__all__ = [
    "BRepAdaptor_Curve2d",
    "BRepAdaptor_Surface",
    "BRepBndLib",
    "BRepClass3d_SolidClassifier",
    "BRepMesh_IncrementalMesh",
    "BRepTools",
    "BRepTools_WireExplorer",
    "BRep_Tool",
    "Bnd_Box",
    "GeomAbs_BSplineSurface",
    "GeomAbs_BezierSurface",
    "GeomAbs_Cone",
    "GeomAbs_Cylinder",
    "GeomAbs_OtherSurface",
    "GeomAbs_Plane",
    "GeomAbs_Sphere",
    "GeomAbs_Torus",
    "IFSelect_RetDone",
    "STEPControl_Reader",
    "TopAbs_EDGE",
    "TopAbs_FACE",
    "TopAbs_FORWARD",
    "TopAbs_IN",
    "TopAbs_SOLID",
    "TopAbs_VERTEX",
    "TopAbs_WIRE",
    "TopExp",
    "TopExp_Explorer",
    "TopLoc_Location",
    "TopTools_IndexedDataMapOfShapeListOfShape",
    "TopTools_IndexedMapOfShape",
    "TopoDS",
    "gp_Pnt",
]
