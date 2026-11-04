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
    ShapeUpgrade_UnifySameDomain,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_OUT,
    TopAbs_REVERSED,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopExp,
    TopExp_Explorer,
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
    # Count occurrences, not unique entities: a loose duplicate of a solid's
    # own face/edge is absent from the mesh but can corrupt CAD mass properties.
    for kind in (TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX):
        counts = []
        for source in (shape, solids.FindKey(1)):
            explorer = TopExp_Explorer(source, kind)
            count = 0
            while explorer.More():
                count += 1
                explorer.Next()
            counts.append(count)
        if counts[0] != counts[1]:
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


def shape_tolerance(shape: TopoDS_Shape) -> float:
    """Largest face, edge or vertex tolerance, floored at 1e-6 working units."""
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
    return tolerance


def unify_same_domain(shape: TopoDS_Shape) -> TopoDS_Shape:
    """Merge coincident same-domain faces and edges of a working solid.

    STEP exporters commonly split one cylindrical wall into two half-cylinder
    faces, and a coplanar seam can also split one axially; either split makes
    ``recognize_holes`` reject the wall as an incomplete cylinder because no
    single face spans the full 2*pi range. Unifying merges such splits (and
    any other same-domain faces/edges elsewhere in the solid) into one face
    without moving the boundary, so recognition can see the merged periodic
    cylinder. The input's face partition is not preserved: callers must take
    face IDs from the returned shape, not the original.

    Raises ValueError (via ``require_solid``) if the result is not a single
    valid solid. Raises RuntimeError if the measured volume or surface area
    moved beyond a bound derived from the shape's own CAD tolerance
    (``shape_tolerance``), which would indicate the merge changed geometry
    rather than only its face partition.
    """
    tolerance = shape_tolerance(shape)
    before_volume, before_area = cad_measures(shape)
    length = GProp_GProps()
    BRepGProp.LinearProperties_s(shape, length)
    unifier = ShapeUpgrade_UnifySameDomain(shape, True, True, False)
    unifier.Build()
    merged = unifier.Shape()
    require_solid(merged)
    after_volume, after_area = cad_measures(merged)
    # Same-domain merging only removes seams between faces/edges already
    # continuous within the shape's own tolerance; it should not move the
    # boundary. Bound the allowed drift as one tolerance-thick layer over the
    # shape's own surface area (for volume) or total edge length (for area),
    # plus a roundoff floor, instead of an unexplained fixed constant.
    volume_bound = tolerance * before_area + 1e-9 * abs(before_volume)
    area_bound = tolerance * length.Mass() + 1e-9 * abs(before_area)
    if (
        abs(after_volume - before_volume) > volume_bound
        or abs(after_area - before_area) > area_bound
    ):
        raise RuntimeError(
            "Same-domain unification changed the solid's volume or area "
            "beyond the tolerance-derived bound"
        )
    return merged


def recognize_holes(
    shape: TopoDS_Shape,
) -> tuple[tuple[ThroughHole, ...], tuple[RejectedHole, ...], float]:
    face_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, face_map)
    faces = [TopoDS.Face_s(face_map.FindKey(i + 1)) for i in range(face_map.Extent())]
    tolerance = shape_tolerance(shape)
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
    shape: TopoDS_Shape,
    holes: tuple[ThroughHole, ...],
    radii: NDArray[np.float64],
    shifts: NDArray[np.float64],
) -> TopoDS_Shape:
    # Each call owns its CAD data; Boolean builders may update input tolerances.
    changed = BRepBuilderAPI_Copy(shape, True, False).Shape()
    for operation, sizes, offsets in (
        (BRepAlgoAPI_Fuse, [hole.radius for hole in holes], np.zeros_like(shifts)),
        (BRepAlgoAPI_Cut, radii, shifts),
    ):
        for hole, radius, offset in zip(holes, sizes, offsets, strict=True):
            tool = BRepPrimAPI_MakeCylinder(
                gp_Ax2(gp_Pnt(*(np.asarray(hole.base) + offset)), gp_Dir(*hole.axis)),
                float(radius),
                hole.height,
            ).Shape()
            builder = operation(changed, tool)
            if not builder.IsDone():
                raise RuntimeError("OCCT hole reconstruction failed")
            # Remove coplanar fill seams before a shifted bore crosses them.
            # Otherwise a valid full cylinder can acquire several end faces,
            # preventing the exported feature from being prepared again.
            builder.SimplifyResult()
            changed = builder.Shape()
    require_solid(changed)
    if _global_edit_error(shape, changed, holes, radii, shifts) is not None:
        # Whole-solid properties are the cheaper check; they can fail through
        # measurement drift alone, so the local regions decide in that case.
        error = _local_edit_error(shape, changed, holes, radii, shifts)
        if error is not None:
            raise RuntimeError(error)
    return changed


def _edit_terms(
    holes: tuple[ThroughHole, ...],
    radii: NDArray[np.float64],
    shifts: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """Removed volume and its first moment for an isolated-hole edit."""
    old = np.array([hole.radius for hole in holes])
    height = np.array([hole.height for hole in holes])
    centers = (
        np.array([hole.base for hole in holes])
        + height[:, None] * np.array([hole.axis for hole in holes]) / 2
    )
    removed = float(np.pi * np.sum((radii**2 - old**2) * height))
    moment = np.pi * np.sum(
        height[:, None]
        * (radii[:, None] ** 2 * (centers + shifts) - old[:, None] ** 2 * centers),
        axis=0,
    )
    return removed, moment


def _global_edit_error(
    shape: TopoDS_Shape,
    changed: TopoDS_Shape,
    holes: tuple[ThroughHole, ...],
    radii: NDArray[np.float64],
    shifts: NDArray[np.float64],
) -> str | None:
    """Compare whole-solid mass properties with the isolated-hole edit law."""
    old = np.array([hole.radius for hole in holes])
    height = np.array([hole.height for hole in holes])
    removed, removed_moment = _edit_terms(holes, radii, shifts)
    expected = cad_measures(shape) + np.array(
        [
            -removed,
            np.sum(
                -2 * np.pi * (radii**2 - old**2) + 2 * np.pi * height * (radii - old)
            ),
        ]
    )
    if not np.allclose(cad_measures(changed), expected, rtol=1e-7, atol=1e-9):
        return "Rebuilt CAD does not match the isolated-hole edit"
    # V/A cannot detect misplaced cylinders. Check first moments independently
    # in the normalized frame, including simultaneous radius/position changes.
    source, result = GProp_GProps(), GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, source)
    BRepGProp.VolumeProperties_s(changed, result)
    moment = source.Mass() * _point(source.CentreOfMass()) - removed_moment
    if not np.allclose(
        _point(result.CentreOfMass()), moment / expected[0], rtol=1e-7, atol=1e-8
    ):
        return "Rebuilt CAD center does not match the isolated-hole edit"
    return None


def _local_edit_error(
    shape: TopoDS_Shape,
    changed: TopoDS_Shape,
    holes: tuple[ThroughHole, ...],
    radii: NDArray[np.float64],
    shifts: NDArray[np.float64],
) -> str | None:
    """Check the edit through the material it actually removed and added.

    Whole-solid properties can drift by about 1e-7 relative when a Boolean
    rebuilds a face bounded by B-spline edges far from the hole, although the
    geometry is unchanged there. Both difference regions are bounded only by
    the hole cylinders and their end planes, so their properties are measured
    without that drift. Recognition accepts end planes within the CAD
    tolerance of the cylinder extent, so each end may add or remove a slice
    of that thickness over the changed cross-section; the comparison allows
    exactly that and roundoff. Each region must also stay beside a bore.
    """
    regions = []
    for minuend, subtrahend in ((shape, changed), (changed, shape)):
        builder = BRepAlgoAPI_Cut(
            BRepBuilderAPI_Copy(minuend, True, False).Shape(),
            BRepBuilderAPI_Copy(subtrahend, True, False).Shape(),
        )
        if not builder.IsDone():
            return "OCCT could not isolate the rebuilt hole regions"
        regions.append(builder.Shape())
    props = []
    for region in regions:
        measured = GProp_GProps()
        BRepGProp.VolumeProperties_s(region, measured)
        props.append(
            (measured.Mass(), measured.Mass() * _point(measured.CentreOfMass()))
        )
    expected, expected_moment = _edit_terms(holes, radii, shifts)
    volume = props[0][0] - props[1][0]
    moment = props[0][1] - props[1][1]
    tolerance = shape_tolerance(shape)
    volume_bound = moment_bound = 0.0
    for hole, radius, offset in zip(holes, radii, shifts, strict=True):
        larger = max(radius, hole.radius)
        # Bound on the area between the old and new bore cross-sections.
        changed_area = np.pi * abs(radius**2 - hole.radius**2) + 4 * larger * float(
            np.linalg.norm(offset)
        )
        center = np.asarray(hole.base) + hole.height * np.asarray(hole.axis) / 2
        reach = float(np.linalg.norm(center)) + hole.height / 2 + larger
        slices = 2 * tolerance * changed_area
        roundoff = 1e-12 * np.pi * larger**2 * hole.height
        volume_bound += slices + roundoff
        moment_bound += (slices + roundoff) * reach
    if abs(volume - expected) > volume_bound or np.any(
        np.abs(moment - expected_moment) > moment_bound
    ):
        return "Rebuilt hole regions do not match the isolated-hole edit"
    bores = []
    for hole, radius, offset in zip(holes, radii, shifts, strict=True):
        for size, center in ((hole.radius, np.zeros(3)), (radius, offset)):
            box = Bnd_Box()
            tool = BRepPrimAPI_MakeCylinder(
                gp_Ax2(gp_Pnt(*(np.asarray(hole.base) + center)), gp_Dir(*hole.axis)),
                float(size),
                hole.height,
            ).Shape()
            BRepBndLib.Add_s(tool, box)
            box.Enlarge(10 * tolerance)
            bores.append(box)
    for region in regions:
        explorer = TopExp_Explorer(region, TopAbs_SOLID)
        while explorer.More():
            box = Bnd_Box()
            BRepBndLib.Add_s(explorer.Current(), box)
            corners = np.asarray(box.Get()).reshape(2, 3)
            if not any(
                np.all(corners[0] >= np.asarray(b.Get())[:3])
                and np.all(corners[1] <= np.asarray(b.Get())[3:])
                for b in bores
            ):
                return "Rebuilt CAD changed material away from the edited holes"
            explorer.Next()
    return None
