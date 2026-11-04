"""Host-side recognition, intervals and rebuilds of planar face offsets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from brepax._occt.backend import (
    Bnd_Box,
    BRepAdaptor_Surface,
    BRepAlgoAPI_Common,
    BRepAlgoAPI_Cut,
    BRepAlgoAPI_Fuse,
    BRepBndLib,
    BRepBuilderAPI_Copy,
    BRepBuilderAPI_Transform,
    BRepGProp,
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakePrism,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GProp_GProps,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_REVERSED,
    TopAbs_SOLID,
    TopAbs_WIRE,
    TopExp,
    TopExp_Explorer,
    TopoDS,
    TopTools_IndexedMapOfShape,
    gp_Ax2,
    gp_Dir,
    gp_Pnt,
    gp_Trsf,
    gp_Vec,
)
from brepax._occt.types import TopoDS_Face, TopoDS_Shape
from brepax.brep.topology import build_adjacency_graph, neighbors
from brepax.experimental.applications._edit_validation import (
    LocalEdit,
    local_edit_error,
    volume_and_moment,
)
from brepax.experimental.applications._hole_geometry import (
    require_solid,
    shape_tolerance,
    unify_same_domain,
)

# Neighbor directions must agree with the face normal to this cosine slack.
_ANGLE_SLACK = 1e-9
# Offset of the Boolean probe that screens out faces whose Booleans already
# disagree with the law, in working units (the working part spans 40).
_PROBE = 1e-3
# Longest travel searched for an event, in working units.
_REACH = 80.0


@dataclass(frozen=True)
class PlanarFace:
    """A planar face that can move along its outward normal.

    Lengths are in the units of the shape the record was made from.
    ``lower`` <= 0 <= ``upper`` bound the offset; ``area``, ``centroid`` and
    ``perimeter`` describe the face before the move.
    """

    face_id: int
    normal: tuple[float, float, float]
    centroid: tuple[float, float, float]
    area: float
    perimeter: float
    lower: float
    upper: float


@dataclass(frozen=True)
class RejectedFace:
    face_id: int
    reason: str


def _point(value: gp_Pnt | gp_Dir) -> NDArray[np.float64]:
    return np.array([value.X(), value.Y(), value.Z()])


def _faces(shape: TopoDS_Shape) -> list[TopoDS_Face]:
    faces = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, faces)
    return [TopoDS.Face_s(faces.FindKey(i + 1)) for i in range(faces.Extent())]


def _count(shape: TopoDS_Shape, kind: object) -> int:
    found = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, found)
    return int(found.Extent())


def _to_face_frame(origin: NDArray[np.float64], normal: NDArray[np.float64]) -> gp_Trsf:
    """World to a frame with the face plane at z=0 and the normal along +z."""
    helper = np.eye(3)[int(np.argmin(np.abs(normal)))]
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    rows = np.stack([u, v, normal])
    shift = -rows @ origin
    transform = gp_Trsf()
    transform.SetValues(*rows[0], shift[0], *rows[1], shift[1], *rows[2], shift[2])
    return transform


def _height_range(shape: TopoDS_Shape, frame: gp_Trsf) -> tuple[float, float] | None:
    moved = BRepBuilderAPI_Transform(shape, frame, True).Shape()
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(moved, box, False, False)
    if box.IsVoid():
        return None
    bounds = box.Get()
    return float(bounds[2]), float(bounds[5])


def _prism(
    face: TopoDS_Face, normal: NDArray[np.float64], offset: float
) -> TopoDS_Shape:
    return BRepPrimAPI_MakePrism(face, gp_Vec(*(normal * offset))).Shape()


def _outward_normal(face: TopoDS_Face) -> NDArray[np.float64]:
    normal = _point(BRepAdaptor_Surface(face).Plane().Axis().Direction())
    return -normal if face.Orientation() == TopAbs_REVERSED else normal


def _supported_neighbors(
    faces: list[TopoDS_Face], neighbor_ids: list[int], normal: NDArray[np.float64]
) -> bool:
    """Neighbors stay put as the face moves: perpendicular planes or cylinders
    whose axes run along the normal."""
    for j in neighbor_ids:
        surface = BRepAdaptor_Surface(faces[j])
        kind = surface.GetType()
        if kind == GeomAbs_Plane:
            direction = _point(surface.Plane().Axis().Direction())
            if abs(direction @ normal) >= _ANGLE_SLACK:
                return False
        elif kind == GeomAbs_Cylinder:
            direction = _point(surface.Cylinder().Axis().Direction())
            if abs(direction @ normal) <= 1 - _ANGLE_SLACK:
                return False
        else:
            return False
    return True


def _events(
    solid: TopoDS_Shape,
    faces: list[TopoDS_Face],
    face_id: int,
    normal: NDArray[np.float64],
    centroid: NDArray[np.float64],
    neighbor_ids: list[int],
    tolerance: float,
) -> tuple[float, float] | None:
    """Conservative travel (outward, inward) before the move meets an event.

    Outward, the face sweeps through air until the swept prism meets
    material; inward, it sweeps through material until the prism leaves the
    solid. Heights of those pieces are bounded by boxes in the face frame,
    which can only underestimate the travel. A neighbor that lies on the
    moving side ends where its edges away from the face begin; edges along
    the normal start at the face and are skipped. Each travel keeps a margin
    of 10 CAD tolerances.
    """
    face = faces[face_id]
    frame = _to_face_frame(centroid, normal)
    floor = 1e-12 * _REACH**3
    shared = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(face, TopAbs_EDGE, shared)
    travel = []
    for sign in (1.0, -1.0):
        limit = _REACH
        tool = _prism(face, normal, sign * _REACH)
        copy = BRepBuilderAPI_Copy(solid, True, False).Shape()
        builder = (
            BRepAlgoAPI_Common(copy, tool) if sign > 0 else BRepAlgoAPI_Cut(tool, copy)
        )
        if not builder.IsDone():
            return None
        explorer = TopExp_Explorer(builder.Shape(), TopAbs_SOLID)
        while explorer.More():
            piece = explorer.Current()
            if volume_and_moment(piece)[0] > floor:
                heights = _height_range(piece, frame)
                if heights is not None:
                    limit = min(limit, heights[0] if sign > 0 else -heights[1])
            explorer.Next()
        for j in neighbor_ids:
            heights = _height_range(faces[j], frame)
            if heights is None:
                continue
            if (heights[1] if sign > 0 else -heights[0]) <= tolerance:
                continue
            explorer = TopExp_Explorer(faces[j], TopAbs_EDGE)
            while explorer.More():
                edge = explorer.Current()
                if not shared.Contains(edge):
                    span = _height_range(edge, frame)
                    if span is not None:
                        far = span[0] if sign > 0 else -span[1]
                        if far > tolerance:
                            limit = min(limit, far)
                explorer.Next()
        travel.append(max(limit - 10 * tolerance, 0.0))
    return travel[0], travel[1]


def _face_properties(
    face: TopoDS_Face,
) -> tuple[float, NDArray[np.float64], float]:
    surface, length = GProp_GProps(), GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, surface, 1e-9)
    BRepGProp.LinearProperties_s(face, length)
    return surface.Mass(), _point(surface.CentreOfMass()), length.Mass()


def _slab_terms(
    faces: tuple[PlanarFace, ...], offsets: NDArray[np.float64]
) -> tuple[float, NDArray[np.float64]]:
    """Added volume and first moment of the swept slabs (exact law)."""
    added = 0.0
    moment = np.zeros(3)
    for face, offset in zip(faces, offsets, strict=True):
        normal = np.asarray(face.normal)
        added += face.area * offset
        moment += face.area * offset * (np.asarray(face.centroid) + normal * offset / 2)
    return added, moment


def _slab_box(
    shape: TopoDS_Shape, face: PlanarFace, low: float, high: float, margin: float
) -> TopoDS_Shape:
    """Box in the face frame around the face, over heights [low, high]."""
    normal = np.asarray(face.normal)
    frame = _to_face_frame(np.asarray(face.centroid), normal)
    flat = _height_range_xy(shape, frame)
    corner = np.array([flat[0] - margin, flat[1] - margin, low - margin])
    size = np.array(
        [flat[2] - flat[0] + 2 * margin, flat[3] - flat[1] + 2 * margin, 0.0]
    )
    size[2] = high - low + 2 * margin
    inverse = frame.Inverted()
    origin = gp_Pnt(*corner).Transformed(inverse)
    axes = gp_Ax2(
        origin,
        gp_Dir(*normal),
        gp_Dir(1, 0, 0).Transformed(inverse),
    )
    return BRepPrimAPI_MakeBox(axes, *size).Shape()


def _height_range_xy(
    shape: TopoDS_Shape, frame: gp_Trsf
) -> tuple[float, float, float, float]:
    moved = BRepBuilderAPI_Transform(shape, frame, True).Shape()
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(moved, box, False, False)
    bounds = box.Get()
    return float(bounds[0]), float(bounds[1]), float(bounds[3]), float(bounds[4])


def rebuild_faces(
    shape: TopoDS_Shape,
    faces: tuple[PlanarFace, ...],
    offsets: NDArray[np.float64],
) -> TopoDS_Shape:
    """Offset independent planar faces and validate the result.

    Each face is swept by its offset and the prism is fused (outward) or cut
    (inward). The result must match the slab law: whole-solid volume and
    first moment first, else inside a box around each slab with the outside
    unchanged (see :func:`local_edit_error`). Same-domain faces are unified
    afterwards so the extended neighbors are single faces again.
    """
    edited = [i for i, offset in enumerate(offsets) if offset != 0]
    faces = tuple(faces[i] for i in edited)
    offsets = np.asarray(offsets, dtype=np.float64)[edited]
    changed = BRepBuilderAPI_Copy(shape, True, False).Shape()
    if not faces:
        return changed
    all_faces = _faces(shape)
    for face, offset in zip(faces, offsets, strict=True):
        tool = _prism(all_faces[face.face_id], np.asarray(face.normal), offset)
        builder = (BRepAlgoAPI_Fuse if offset > 0 else BRepAlgoAPI_Cut)(changed, tool)
        if not builder.IsDone():
            raise RuntimeError("OCCT face offset failed")
        changed = builder.Shape()
    require_solid(changed)
    error = _offset_error(shape, changed, faces, offsets)
    if error is not None:
        raise RuntimeError(error)
    return unify_same_domain(changed)


def _offset_error(
    shape: TopoDS_Shape,
    changed: TopoDS_Shape,
    faces: tuple[PlanarFace, ...],
    offsets: NDArray[np.float64],
) -> str | None:
    tolerance = shape_tolerance(shape)
    added, added_moment = _slab_terms(faces, offsets)
    before = volume_and_moment(shape)
    after = volume_and_moment(changed)
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    reach = float(np.abs(np.asarray(box.Get())).max()) + float(np.abs(offsets).max())
    if abs(after[0] - before[0] - added) <= 1e-7 * abs(before[0]) and np.all(
        np.abs(after[1] - before[1] - added_moment) <= 1e-7 * abs(before[0]) * reach
    ):
        return None
    margin = 10 * tolerance
    envelope = None
    zones = []
    volume_bound = moment_bound = 0.0
    for face, offset in zip(faces, offsets, strict=True):
        slab = _slab_box(
            _faces(shape)[face.face_id],
            face,
            min(0.0, offset),
            max(0.0, offset),
            margin,
        )
        bounds = Bnd_Box()
        BRepBndLib.Add_s(slab, bounds)
        zones.append(np.asarray(bounds.Get()).reshape(2, 3))
        if envelope is None:
            envelope = slab
        else:
            builder = BRepAlgoAPI_Fuse(envelope, slab)
            if not builder.IsDone():
                return "OCCT could not build the face offset envelope"
            envelope = builder.Shape()
        # Neighbors may deviate from perpendicular within the CAD tolerance
        # along the swept perimeter, and the face from its plane over its area.
        bound = tolerance * (face.perimeter * abs(offset) + 2 * face.area)
        bound += 1e-12 * abs(before[0])
        volume_bound += bound
        moment_bound += bound * reach
    if envelope is None:
        return None
    return local_edit_error(
        shape,
        changed,
        LocalEdit(
            envelope=envelope,
            zones=tuple(zones),
            removed_volume=-added,
            removed_moment=-added_moment,
            volume_bound=volume_bound,
            moment_bound=moment_bound,
            mismatch_message="Rebuilt face regions do not match the offset law",
            away_message="Rebuilt CAD changed material away from the offset faces",
        ),
    )


def recognize_faces(
    shape: TopoDS_Shape, face_ids: tuple[int, ...] | None = None
) -> tuple[tuple[PlanarFace, ...], tuple[RejectedFace, ...]]:
    """Find planar faces with a nonempty offset interval in a working solid.

    A face qualifies when it is planar with a single boundary loop and every
    neighbor is a perpendicular plane or a cylinder whose axis runs along the
    face normal. Its interval comes from :func:`_events`. A face whose
    Boolean already breaks the law at a 1e-3 working-unit probe in either
    direction is rejected. ``face_ids`` restricts the search; by default
    every planar face is examined and every rejected planar face reported.
    """
    tolerance = shape_tolerance(shape)
    faces = _faces(shape)
    graph = build_adjacency_graph(shape)
    found: list[PlanarFace] = []
    rejected: list[RejectedFace] = []
    candidates = range(len(faces)) if face_ids is None else face_ids
    for face_id in candidates:
        face = faces[face_id]
        if BRepAdaptor_Surface(face).GetType() != GeomAbs_Plane:
            rejected.append(RejectedFace(face_id, "not planar"))
            continue
        if _count(face, TopAbs_WIRE) != 1:
            rejected.append(RejectedFace(face_id, "inner boundary loops"))
            continue
        normal = _outward_normal(face)
        neighbor_ids = [j for j in neighbors(graph, face_id) if j != face_id]
        if not _supported_neighbors(faces, neighbor_ids, normal):
            rejected.append(
                RejectedFace(
                    face_id,
                    "a neighbor is neither a perpendicular plane nor a cylinder "
                    "along the normal",
                )
            )
            continue
        area, centroid, perimeter = _face_properties(face)
        travel = _events(
            shape, faces, face_id, normal, centroid, neighbor_ids, tolerance
        )
        if travel is None:
            rejected.append(RejectedFace(face_id, "OCCT could not sweep the face"))
            continue
        record = PlanarFace(
            face_id,
            (float(normal[0]), float(normal[1]), float(normal[2])),
            (float(centroid[0]), float(centroid[1]), float(centroid[2])),
            area,
            perimeter,
            -travel[1],
            travel[0],
        )
        if record.upper < _PROBE and -record.lower < _PROBE:
            rejected.append(RejectedFace(face_id, "no admissible offset"))
            continue
        failure = None
        for probe in (_PROBE, -_PROBE):
            if not record.lower <= probe <= record.upper:
                continue
            try:
                rebuild_faces(shape, (record,), np.array([probe]))
            except (RuntimeError, ValueError) as error:
                failure = str(error)
                break
        if failure is not None:
            rejected.append(
                RejectedFace(face_id, f"Boolean probe failed validation: {failure}")
            )
            continue
        found.append(record)
    return tuple(found), tuple(rejected)


def interacting(first: PlanarFace, second: PlanarFace, shape: TopoDS_Shape) -> bool:
    """Whether two faces' full offset ranges could touch each other."""
    faces = _faces(shape)
    boxes = []
    for face in (first, second):
        slab = _slab_box(faces[face.face_id], face, face.lower, face.upper, 0.0)
        bounds = Bnd_Box()
        BRepBndLib.Add_s(slab, bounds)
        boxes.append(np.asarray(bounds.Get()).reshape(2, 3))
    return bool(
        np.all(boxes[0][0] <= boxes[1][1]) and np.all(boxes[1][0] <= boxes[0][1])
    )
