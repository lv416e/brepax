"""Thin wrapper around cadquery-ocp-novtk providing the OCCT subset BRepAX needs.

All OCP imports are centralized here so that swapping the underlying
OCCT binding (e.g., to a custom pybind11 wrapper) requires changing
only this module.
"""

from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Curve2d, BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_Copy,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_Transform,
)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakePrism,
    BRepPrimAPI_MakeSphere,
    BRepPrimAPI_MakeTorus,
)
from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
from OCP.GeomAbs import (
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Line,
    GeomAbs_OtherSurface,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCP.gp import gp_Ax2, gp_Dir, gp_Pln, gp_Pnt, gp_Trsf, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_IN,
    TopAbs_OUT,
    TopAbs_REVERSED,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Compound
from OCP.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_IndexedMapOfShape,
)

__all__ = [
    "BRepAdaptor_Curve",
    "BRepAdaptor_Curve2d",
    "BRepAdaptor_Surface",
    "BRepAlgoAPI_Common",
    "BRepAlgoAPI_Cut",
    "BRepAlgoAPI_Fuse",
    "BRepBndLib",
    "BRepBuilderAPI_Copy",
    "BRepBuilderAPI_MakeFace",
    "BRepBuilderAPI_MakePolygon",
    "BRepBuilderAPI_MakeVertex",
    "BRepBuilderAPI_Transform",
    "BRepCheck_Analyzer",
    "BRepClass3d_SolidClassifier",
    "BRepClass_FaceClassifier",
    "BRepExtrema_DistShapeShape",
    "BRepGProp",
    "BRepMesh_IncrementalMesh",
    "BRepPrimAPI_MakeBox",
    "BRepPrimAPI_MakeCone",
    "BRepPrimAPI_MakeCylinder",
    "BRepPrimAPI_MakePrism",
    "BRepPrimAPI_MakeSphere",
    "BRepPrimAPI_MakeTorus",
    "BRepTools",
    "BRepTools_WireExplorer",
    "BRep_Builder",
    "BRep_Tool",
    "Bnd_Box",
    "GProp_GProps",
    "GeomAbs_BSplineSurface",
    "GeomAbs_BezierSurface",
    "GeomAbs_Cone",
    "GeomAbs_Cylinder",
    "GeomAbs_Line",
    "GeomAbs_OtherSurface",
    "GeomAbs_Plane",
    "GeomAbs_Sphere",
    "GeomAbs_Torus",
    "IFSelect_RetDone",
    "STEPControl_AsIs",
    "STEPControl_Reader",
    "STEPControl_Writer",
    "TColStd_Array1OfInteger",
    "TColStd_Array1OfReal",
    "TopAbs_EDGE",
    "TopAbs_FACE",
    "TopAbs_FORWARD",
    "TopAbs_IN",
    "TopAbs_OUT",
    "TopAbs_REVERSED",
    "TopAbs_SOLID",
    "TopAbs_VERTEX",
    "TopAbs_WIRE",
    "TopExp",
    "TopExp_Explorer",
    "TopLoc_Location",
    "TopTools_IndexedDataMapOfShapeListOfShape",
    "TopTools_IndexedMapOfShape",
    "TopoDS",
    "TopoDS_Compound",
    "gp_Ax2",
    "gp_Dir",
    "gp_Pln",
    "gp_Pnt",
    "gp_Trsf",
    "gp_Vec",
]
