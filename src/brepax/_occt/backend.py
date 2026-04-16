"""Thin wrapper around cadquery-ocp-novtk providing the OCCT subset BRepAX needs.

All OCP imports are centralized here so that swapping the underlying
OCCT binding (e.g., to a custom pybind11 wrapper) requires changing
only this module.
"""

from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
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
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

__all__ = [
    "BRepAdaptor_Surface",
    "BRepBndLib",
    "BRepMesh_IncrementalMesh",
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
    "TopAbs_VERTEX",
    "TopExp_Explorer",
    "TopLoc_Location",
    "TopoDS",
]
