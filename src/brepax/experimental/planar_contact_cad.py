"""Host preparation of a nominal contact patch from selected planar CAD faces.

This module prepares a planar contact model, not an exact-contact certificate
or a repaired B-Rep. Source gaps and numerical residuals remain visible.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from brepax._occt.backend import (
    BRep_Tool,
    BRepAdaptor_Curve,
    BRepAdaptor_Curve2d,
    BRepAdaptor_Surface,
    BRepCheck_Analyzer,
    BRepGProp,
    BRepTools,
    BRepTools_WireExplorer,
    GeomAbs_Line,
    GeomAbs_Plane,
    GProp_GProps,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_REVERSED,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
    TopExp_Explorer,
    TopoDS,
)
from brepax._occt.types import TopoDS_Face, TopoDS_Shape
from brepax.experimental.planar_contact import (
    PlanarContactPatch,
    prepare_planar_contact_patch,
)

_Array = NDArray[np.float64]
_Mode = Literal["zero_residual", "projected_nominal"]
_ROUND_OFF = 128 * np.finfo(np.float64).eps


@dataclass(frozen=True)
class PlanarContactDiagnostics:
    """Source-CAD evidence, in input length units unless dimensionless.

    ``plane_distance_range`` contains signed moving-vertex distances from the
    fixed plane, whose outward normal points towards the moving solid.
    ``signed_plane_gap`` uses the underlying plane origins. With imperfect
    opposition there is no single gap, so both quantities are retained.
    Normal residual is dimensionless. Bounds concern floating-point geometric
    consistency and never replace a nonzero gap by exact source contact.
    """

    mode: _Mode
    face_ids: tuple[int, int]
    signed_plane_gap: float
    plane_distance_range: tuple[float, float]
    opposed_normal_residual: float
    face_tolerances: tuple[float, float]
    max_wire_geometry_residual: float
    max_support_residual: float
    roundoff_length_bound: float
    reference_kind: Literal["projected_support_polygons"] = field(
        default="projected_support_polygons", init=False
    )
    original_cad_contact_certified: Literal[False] = field(default=False, init=False)


@dataclass(frozen=True)
class PreparedPlanarContact:
    """Numerical reference patch and host-only source diagnostics."""

    patch: PlanarContactPatch
    diagnostics: PlanarContactDiagnostics


@dataclass(frozen=True)
class _FacePolygon:
    origin: _Array
    basis: _Array
    normal: _Array
    vertices: _Array
    polygon: _Array
    area: float
    tolerance: float
    wire_residual: float


@dataclass(frozen=True)
class _SupportedSolid:
    selected: _FacePolygon
    support_residual: float
    wire_residual: float


def _coordinates(point: Any) -> _Array:
    return np.array([point.X(), point.Y(), point.Z()], dtype=np.float64)


def _subshapes(shape: TopoDS_Shape, kind: Any) -> list[TopoDS_Shape]:
    explorer = TopExp_Explorer(shape, kind)
    result = []
    while explorer.More():
        result.append(explorer.Current())
        explorer.Next()
    return result


def _signed_area(polygon: _Array) -> float:
    # Subtraction avoids area cancellation when the surface parameter origin
    # is distant from the trimmed face.
    local = polygon - polygon[0]
    following = np.roll(local, -1, axis=0)
    return float(
        np.sum(local[:, 0] * following[:, 1] - local[:, 1] * following[:, 0]) / 2
    )


def _cross2(first: _Array, second: _Array) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _extract_face(face: TopoDS_Face) -> _FacePolygon:
    if not BRepCheck_Analyzer(face, True, False, True).IsValid():
        raise ValueError("Invalid CAD face")
    adaptor = BRepAdaptor_Surface(face)
    if adaptor.GetType() != GeomAbs_Plane:
        raise ValueError("Contact preparation supports only planar faces")
    if face.Orientation() not in (TopAbs_FORWARD, TopAbs_REVERSED):
        raise ValueError("Unsupported face orientation")
    if len(_subshapes(face, TopAbs_WIRE)) != 1:
        raise ValueError("Faces must have one wire and no holes")
    plane = adaptor.Plane().Position()
    origin = _coordinates(plane.Location())
    normal = _coordinates(plane.Direction())
    if face.Orientation() == TopAbs_REVERSED:
        normal = -normal
    first_axis = _coordinates(plane.XDirection())
    basis = np.column_stack((first_axis, np.cross(normal, first_axis)))
    wire = BRepTools.OuterWire_s(face)
    if wire.IsNull():
        raise ValueError("Face has no outer wire")
    explorer = BRepTools_WireExplorer(wire, face)
    starts, ends, residuals = [], [], []
    while explorer.More():
        edge = explorer.Current()
        if edge.Orientation() not in (TopAbs_FORWARD, TopAbs_REVERSED):
            raise ValueError("Unsupported edge orientation")
        curve = BRepAdaptor_Curve2d(edge, face)
        spatial_curve = BRepAdaptor_Curve(edge)
        if curve.GetType() != GeomAbs_Line or spatial_curve.GetType() != GeomAbs_Line:
            raise ValueError("Faces must have straight edges")
        first, last = curve.FirstParameter(), curve.LastParameter()
        if edge.Orientation() == TopAbs_REVERSED:
            first, last = last, first
        uv_start, uv_end = curve.Value(first), curve.Value(last)
        start = _coordinates(adaptor.Value(uv_start.X(), uv_start.Y()))
        end = _coordinates(adaptor.Value(uv_end.X(), uv_end.Y()))
        vertex = _coordinates(BRep_Tool.Pnt_s(explorer.CurrentVertex()))
        starts.append(start)
        ends.append(end)
        residuals.extend(
            [
                float(np.linalg.norm(vertex - start)),
                float(np.linalg.norm(_coordinates(spatial_curve.Value(first)) - start)),
                float(np.linalg.norm(_coordinates(spatial_curve.Value(last)) - end)),
            ]
        )
        explorer.Next()
    if len(starts) != len(_subshapes(wire, TopAbs_EDGE)):
        raise ValueError("Incomplete wire traversal")
    vertices = np.asarray(starts, dtype=np.float64)
    if len(vertices) < 3 or not np.isfinite(vertices).all():
        raise ValueError("Face polygon is degenerate or nonfinite")
    length = float(np.max(np.linalg.norm(vertices - vertices[0], axis=1)))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Face has no finite extent")
    bound = _ROUND_OFF * length
    closure = float(
        np.max(np.linalg.norm(np.asarray(ends) - np.roll(vertices, -1, axis=0), axis=1))
    )
    wire_residual = max(closure, *residuals)
    if not np.isfinite(wire_residual) or wire_residual > bound:
        raise ValueError("Wire geometry exceeds the roundoff consistency bound")
    polygon = (vertices - origin) @ basis
    signed_area = _signed_area(polygon)
    if not np.isfinite(signed_area) or abs(signed_area) <= bound * length:
        raise ValueError("Face has numerically degenerate area")
    if signed_area < 0:
        vertices, polygon = vertices[::-1], polygon[::-1]
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        edge = end - start
        edge_length = float(np.linalg.norm(edge))
        if edge_length <= bound:
            raise ValueError("Face has a numerically degenerate edge")
        distances = [_cross2(edge, point - start) for point in polygon]
        if min(distances) < -bound * edge_length:
            raise ValueError("Faces must be convex")
    properties = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, properties)
    if abs(abs(signed_area) - properties.Mass()) > 8 * bound * length:
        raise ValueError("Polygon and CAD face areas disagree")
    return _FacePolygon(
        origin,
        basis,
        normal,
        vertices,
        polygon,
        abs(signed_area),
        float(BRep_Tool.Tolerance_s(face)),
        wire_residual,
    )


def _supported_solid(shape: TopoDS_Shape, face_id: int) -> _SupportedSolid:
    if shape is None or shape.IsNull():
        raise ValueError("Expected a non-null finite solid")
    if not BRepCheck_Analyzer(shape, True, False, True).IsValid():
        raise ValueError("Invalid CAD solid")
    solids = _subshapes(shape, TopAbs_SOLID)
    if len(solids) != 1 or any(
        len(_subshapes(shape, kind)) != len(_subshapes(solids[0], kind))
        for kind in (TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX)
    ):
        raise ValueError("Expected one solid without additional geometry")
    faces = _subshapes(shape, TopAbs_FACE)
    if isinstance(face_id, bool) or not isinstance(face_id, int):
        raise ValueError("Face ID must be an integer")
    if face_id < 0 or face_id >= len(faces):
        raise ValueError("Selected face ID is outside the shape")
    polygons = [_extract_face(TopoDS.Face_s(face)) for face in faces]
    vertices = np.concatenate([polygon.vertices for polygon in polygons])
    length = float(np.max(np.linalg.norm(vertices - vertices[0], axis=1)))
    bound = 2 * _ROUND_OFF * length
    support_residual = 0.0
    for polygon in polygons:
        distances = (vertices - polygon.origin) @ polygon.normal
        residual = float(np.max(distances))
        support_residual = max(support_residual, residual)
        if not np.isfinite(residual) or residual > bound:
            raise ValueError("Solid is not convex in its oriented face halfspaces")
    selected = polygons[face_id]
    for index, polygon in enumerate(polygons):
        if index != face_id:
            distances = (polygon.vertices - selected.origin) @ selected.normal
            if np.all(np.abs(distances) <= bound):
                raise ValueError("Selected support plane contains multiple faces")
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    if not np.isfinite(properties.Mass()) or properties.Mass() <= 0:
        raise ValueError("Solid must have positive finite volume")
    return _SupportedSolid(
        selected, support_residual, max(p.wire_residual for p in polygons)
    )


def _intersection_polygon(subject: _Array, clipper: _Array) -> _Array:
    output = list(subject)
    for first, last in zip(clipper, np.roll(clipper, -1, axis=0), strict=True):
        previous, output = output, []
        if not previous:
            break
        start = previous[-1]
        start_distance = _cross2(last - first, start - first)
        for end in previous:
            end_distance = _cross2(last - first, end - first)
            if (end_distance >= 0) != (start_distance >= 0):
                output.append(
                    start
                    + start_distance / (start_distance - end_distance) * (end - start)
                )
            if end_distance >= 0:
                output.append(end)
            start, start_distance = end, end_distance
    # Exact boundary hits can emit the same vertex twice. Removing identical
    # neighbours changes neither the patch nor its physical reference.
    unique: list[_Array] = []
    for point in output:
        if not unique or not np.array_equal(point, unique[-1]):
            unique.append(point)
    if len(unique) > 1 and np.array_equal(unique[0], unique[-1]):
        unique.pop()
    return np.asarray(unique, dtype=np.float64).reshape(-1, 2)


def prepare_planar_contact_reference(
    fixed_shape: TopoDS_Shape,
    moving_shape: TopoDS_Shape,
    *,
    fixed_face_id: int,
    moving_face_id: int,
    mode: _Mode,
) -> PreparedPlanarContact:
    """Prepare explicitly selected convex support polygons on the fixed plane.

    Both inputs must be valid finite convex polyhedral solids with planar,
    convex, single-wire faces and straight edges. Face IDs index the current
    ``TopExp_Explorer(shape, TopAbs_FACE)`` order; rebuilding CAD can change IDs.
    The selected faces must oppose, be the unique faces on their support
    planes, and have positive-area overlap after projection. Whole-solid
    checks exclude competing contacts outside this convex supporting model.

    ``mode`` is required. ``"zero_residual"`` rejects every computed nonzero
    plane or normal-opposition residual. ``"projected_nominal"`` permits only
    roundoff-sized residuals and explicitly projects the moving support polygon
    onto the fixed plane. Both return the *projected support-polygon reference*,
    not a repaired solid or a certified derivative of the original CAD. Even a
    positive source gap below either OCCT tolerance or roundoff has a different
    contact derivative; inspect the returned diagnostics and establish actual
    contact independently before applying the reference law to source CAD.

    Lengths retain input units; linear velocities and pivots passed to the
    evaluator must use these units. Preparation requires JAX x64 and runs on
    the host. The resulting patch supports the numerical evaluator's JAX
    transformations. Rigid placement and positive unit scaling preserve the
    reference geometry up to extraction roundoff. Extreme origin/feature
    conditioning can fail the consistency checks instead of being snapped.

    Raises ``ValueError`` for unsupported geometry, selection, residuals, or
    mode. No contact search, face healing, or original-solid modification occurs.

    Example, given two known contacting boxes and their selected face IDs::

        prepared = prepare_planar_contact_reference(
            fixed, moving, fixed_face_id=5, moving_face_id=4,
            mode="zero_residual",
        )
        rate = planar_contact_volume_direction(
            prepared.patch, velocity, angular_velocity, pivot,
        )
    """
    if mode not in ("zero_residual", "projected_nominal"):
        raise ValueError("Mode must be zero_residual or projected_nominal")
    if jnp.asarray(0.0).dtype != np.dtype("float64"):
        raise ValueError("CAD contact preparation requires JAX x64")
    fixed = _supported_solid(fixed_shape, fixed_face_id)
    moving = _supported_solid(moving_shape, moving_face_id)
    first, second = fixed.selected, moving.selected
    opposed_residual = float(np.linalg.norm(first.normal + second.normal))
    distances = (second.vertices - first.origin) @ first.normal
    gap = float((second.origin - first.origin) @ first.normal)
    max_distance = float(np.max(np.abs(distances)))
    length = max(np.sqrt(first.area), np.sqrt(second.area))
    bound = _ROUND_OFF * length
    if not np.isfinite([opposed_residual, gap, max_distance]).all():
        raise ValueError("Nonfinite contact residual")
    if opposed_residual > _ROUND_OFF:
        raise ValueError("Selected outward normals must oppose")
    if mode == "zero_residual":
        if opposed_residual != 0 or max_distance != 0 or gap != 0:
            raise ValueError(
                f"Nonzero contact residual: signed gap={gap:.17g}, "
                f"max plane distance={max_distance:.17g}, "
                f"normal opposition={opposed_residual:.17g}"
            )
    elif max(max_distance, abs(gap)) > bound:
        raise ValueError("Source plane residual exceeds the nominal roundoff bound")
    projected = (second.vertices - first.origin) @ first.basis
    if _signed_area(projected) < 0:
        projected = projected[::-1]
    polygon = _intersection_polygon(first.polygon, projected)
    if len(polygon) < 3 or _signed_area(polygon) <= bound * length:
        raise ValueError("Selected faces have no positive-area contact patch")
    patch = prepare_planar_contact_patch(
        polygon, origin=first.origin, basis=first.basis
    )
    diagnostics = PlanarContactDiagnostics(
        mode=mode,
        face_ids=(fixed_face_id, moving_face_id),
        signed_plane_gap=gap,
        plane_distance_range=(float(np.min(distances)), float(np.max(distances))),
        opposed_normal_residual=opposed_residual,
        face_tolerances=(first.tolerance, second.tolerance),
        max_wire_geometry_residual=max(fixed.wire_residual, moving.wire_residual),
        max_support_residual=max(fixed.support_residual, moving.support_residual),
        roundoff_length_bound=bound,
    )
    return PreparedPlanarContact(patch, diagnostics)
