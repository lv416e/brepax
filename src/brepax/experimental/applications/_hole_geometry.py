"""Host-side recognition and reconstruction for bounded through-hole edits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from brepax._occt.backend import (
    Bnd_Box,
    BRep_Tool,
    BRepAdaptor_Surface,
    BRepAlgoAPI_Common,
    BRepAlgoAPI_Cut,
    BRepAlgoAPI_Fuse,
    BRepBndLib,
    BRepBuilderAPI_Copy,
    BRepBuilderAPI_Transform,
    BRepCheck_Analyzer,
    BRepClass_FaceClassifier,
    BRepExtrema_DistShapeShape,
    BRepGProp,
    BRepPrimAPI_MakeCylinder,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GProp_GProps,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_OUT,
    TopAbs_REVERSED,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopExp,
    TopoDS,
    TopTools_IndexedMapOfShape,
    gp_Ax2,
    gp_Dir,
    gp_Pnt,
    gp_Trsf,
)
from brepax._occt.types import TopoDS_Face, TopoDS_Shape
from brepax.brep.topology import build_adjacency_graph, neighbors
from brepax.experimental.applications.cad_frame import CadFrame


@dataclass(frozen=True)
class ThroughHole:
    """Recognized hole in input units; face IDs belong to the prepared input only.

    ``base`` is one axis endpoint, ``axis`` a unit direction toward the other,
    and ``clearance`` the distance from the cylinder to other boundary faces,
    excluding its two planar end faces. No identity persistence is promised
    after Boolean rebuilding or STEP serialization.
    """

    face_id: int
    radius: float
    base: tuple[float, float, float]
    axis: tuple[float, float, float]
    height: float
    clearance: float


@dataclass(frozen=True)
class RejectedHole:
    """An inward cylindrical face excluded from the editable feature set."""

    face_id: int
    reason: str


def require_solid(shape: TopoDS_Shape) -> None:
    """Reject null, invalid, multiple, reversed or incomplete solid inputs."""
    if shape.IsNull() or not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError("Expected a valid non-null solid")
    solids = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_SOLID, solids)
    if solids.Extent() != 1:
        raise ValueError("Expected exactly one solid")
    # Do not silently discard orphan faces/edges in a STEP compound.
    for kind in (TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX):
        whole, solid = TopTools_IndexedMapOfShape(), TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape, kind, whole)
        TopExp.MapShapes_s(solids.FindKey(1), kind, solid)
        if whole.Extent() != solid.Extent():
            raise ValueError("Input contains geometry outside its solid")
    if cad_measures(shape)[0] <= 0:
        raise ValueError("Expected a positive-volume, outward-oriented solid")


def normalize_shape(
    shape: TopoDS_Shape, frame: CadFrame | None
) -> tuple[TopoDS_Shape, CadFrame]:
    require_solid(shape)
    if frame is None:
        box = Bnd_Box()
        BRepBndLib.AddOptimal_s(shape, box, False, False)
        bounds = np.asarray(box.Get()).reshape(2, 3)
        # A fixed characteristic size keeps OCCT away from tiny/huge units.
        scale = float(np.max(bounds[1] - bounds[0]) / 40)
        frame = CadFrame(scale, tuple((bounds[0] + bounds[1]) / 2))
    scale = 1 / frame.length_scale
    x, y, z = -np.asarray(frame.origin) * scale
    transform = gp_Trsf()
    transform.SetValues(scale, 0, 0, x, 0, scale, 0, y, 0, 0, scale, z)
    working = BRepBuilderAPI_Transform(shape, transform, True, False).Shape()
    require_solid(working)
    return working, frame


def cad_measures(shape: TopoDS_Shape) -> NDArray[np.float64]:
    volume, area = GProp_GProps(), GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, volume)
    BRepGProp.SurfaceProperties_s(shape, area)
    return np.array([volume.Mass(), area.Mass()])


def _point(value: gp_Pnt | gp_Dir) -> NDArray[np.float64]:
    return np.array([value.X(), value.Y(), value.Z()])


def _recognize(
    shape: TopoDS_Shape,
    index: int,
    faces: list[TopoDS_Face],
    caps: list[int],
    tolerance: float,
) -> ThroughHole:
    face = faces[index]
    adaptor = BRepAdaptor_Surface(face)
    cylinder = adaptor.Cylinder()
    radius = float(cylinder.Radius())
    axis = _point(cylinder.Axis().Direction())
    low, high = adaptor.FirstVParameter(), adaptor.LastVParameter()
    base = _point(cylinder.Location()) + low * axis
    height = float(high - low)
    if (
        not np.isfinite([radius, height, *base]).all()
        or min(radius, height) <= 100 * tolerance
    ):
        raise ValueError("Hole dimensions are too small relative to CAD tolerance")
    if abs(adaptor.LastUParameter() - adaptor.FirstUParameter() - 2 * np.pi) > 1e-8:
        raise ValueError("Not a complete cylindrical hole")
    if len(caps) != 2:
        raise ValueError("Hole must have two planar ends")
    ends = []
    for cap in caps:
        plane = BRepAdaptor_Surface(faces[cap])
        if plane.GetType() != GeomAbs_Plane:
            raise ValueError("Nonplanar hole end")
        normal = _point(plane.Plane().Axis().Direction())
        if abs(abs(normal @ axis) - 1) > 1e-8:
            raise ValueError("Oblique hole end")
        z = (_point(plane.Plane().Location()) - base) @ axis
        if min(abs(z), abs(z - height)) > tolerance:
            raise ValueError("End plane does not match cylinder extent")
        end = int(abs(z - height) < abs(z))
        ends.append(end)
        outward = normal if faces[cap].Orientation() == TopAbs_FORWARD else -normal
        # A counterbore shoulder can surround an empty axis but faces into
        # the large bore. Its area law and boundary motion are different.
        if abs(outward @ axis - (2 * end - 1)) > 1e-8:
            raise ValueError("End face orientation does not bound a through-hole")
        if (
            BRepClass_FaceClassifier(
                faces[cap], gp_Pnt(*(base + z * axis)), tolerance
            ).State()
            != TopAbs_OUT
        ):
            raise ValueError("Blind hole: end plane covers the hole axis")
    if sorted(ends) != [0, 1]:
        raise ValueError("Hole ends must be on opposite planes")
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    if not np.isclose(props.Mass(), 2 * np.pi * radius * height, rtol=1e-8):
        raise ValueError("Cylinder face has additional trims")
    distances = []
    for other, boundary in enumerate(faces):
        if other in [index, *caps]:
            continue
        distance = BRepExtrema_DistShapeShape(face, boundary)
        if not distance.IsDone():
            raise RuntimeError("OCCT boundary distance failed")
        distances.append(float(distance.Value()))
    clearance = min(distances, default=0.0)
    if not np.isfinite(clearance) or clearance <= 100 * tolerance:
        raise ValueError("Insufficient boundary clearance")
    # A connected peg can enter an otherwise full cylindrical wall from an
    # opening. Cap classification alone would accept it; fill/re-cut would
    # then erase that material. Require the entire cylinder to be void.
    tool = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(*base), gp_Dir(*axis)), radius, height
    ).Shape()
    overlap = BRepAlgoAPI_Common(BRepBuilderAPI_Copy(shape, True, False).Shape(), tool)
    if not overlap.IsDone():
        raise RuntimeError("OCCT hole interior classification failed")
    occupied = GProp_GProps()
    BRepGProp.VolumeProperties_s(overlap.Shape(), occupied)
    if abs(occupied.Mass()) > 1e-9 * np.pi * radius**2 * height:
        raise ValueError("Hole cylinder contains other solid material")
    return ThroughHole(index, radius, tuple(base), tuple(axis), height, clearance)


def recognize_holes(
    shape: TopoDS_Shape,
) -> tuple[tuple[ThroughHole, ...], tuple[RejectedHole, ...], float]:
    face_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, face_map)
    faces = [TopoDS.Face_s(face_map.FindKey(i + 1)) for i in range(face_map.Extent())]
    tolerance = 1e-6
    for kind, cast in (
        (TopAbs_FACE, TopoDS.Face_s),
        (TopAbs_EDGE, TopoDS.Edge_s),
        (TopAbs_VERTEX, TopoDS.Vertex_s),
    ):
        items = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape, kind, items)
        for i in range(1, items.Extent() + 1):
            tolerance = max(
                tolerance, float(BRep_Tool.Tolerance_s(cast(items.FindKey(i))))
            )
    graph = build_adjacency_graph(shape)
    holes, rejected = [], []
    for index, face in enumerate(faces):
        if (
            BRepAdaptor_Surface(face).GetType() != GeomAbs_Cylinder
            or face.Orientation() != TopAbs_REVERSED
        ):
            continue
        caps = [i for i in neighbors(graph, index) if i != index]
        try:
            holes.append(_recognize(shape, index, faces, caps, tolerance))
        except ValueError as error:
            rejected.append(RejectedHole(index, str(error)))
    return tuple(holes), tuple(rejected), tolerance


def external_hole(hole: ThroughHole, frame: CadFrame) -> ThroughHole:
    scale = frame.length_scale
    return ThroughHole(
        hole.face_id,
        hole.radius * scale,
        tuple(np.asarray(hole.base) * scale + frame.origin),
        hole.axis,
        hole.height * scale,
        hole.clearance * scale,
    )


def rebuild_holes(
    shape: TopoDS_Shape, holes: tuple[ThroughHole, ...], radii: NDArray[np.float64]
) -> TopoDS_Shape:
    # Each call owns its CAD data; Boolean builders may update input tolerances.
    changed = BRepBuilderAPI_Copy(shape, True, False).Shape()
    for operation, sizes in (
        (BRepAlgoAPI_Fuse, [hole.radius for hole in holes]),
        (BRepAlgoAPI_Cut, radii),
    ):
        for hole, radius in zip(holes, sizes, strict=True):
            tool = BRepPrimAPI_MakeCylinder(
                gp_Ax2(gp_Pnt(*hole.base), gp_Dir(*hole.axis)),
                float(radius),
                hole.height,
            ).Shape()
            builder = operation(changed, tool)
            if not builder.IsDone():
                raise RuntimeError("OCCT hole reconstruction failed")
            changed = builder.Shape()
    require_solid(changed)
    old = np.array([hole.radius for hole in holes])
    height = np.array([hole.height for hole in holes])
    squared = radii**2 - old**2
    expected = cad_measures(shape) + np.array(
        [
            -np.pi * np.sum(squared * height),
            np.sum(-2 * np.pi * squared + 2 * np.pi * height * (radii - old)),
        ]
    )
    if not np.allclose(cad_measures(changed), expected, rtol=1e-7, atol=1e-9):
        raise RuntimeError("Rebuilt CAD does not match the isolated-hole edit")
    return changed
