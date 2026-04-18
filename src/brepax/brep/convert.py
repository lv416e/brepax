"""Conversion between OCP B-Rep entities and JAX-friendly representations."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np

from brepax._occt.backend import (
    Bnd_Box,
    BRepAdaptor_Curve2d,
    BRepAdaptor_Surface,
    BRepBndLib,
    BRepTools,
    BRepTools_WireExplorer,
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
    TopAbs_FORWARD,
    TopAbs_VERTEX,
    TopExp_Explorer,
    TopoDS,
)
from brepax._occt.types import TopoDS_Face, TopoDS_Shape
from brepax.primitives import BSplineSurface as BSplinePrim
from brepax.primitives import Cone as ConePrim
from brepax.primitives import Cylinder as CylinderPrim
from brepax.primitives import Plane as PlanePrim
from brepax.primitives import Sphere as SpherePrim
from brepax.primitives import Torus as TorusPrim
from brepax.primitives._base import Primitive

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


def _gp_pnt_to_array(pnt: Any) -> jnp.ndarray:
    """Convert an OCP gp_Pnt to a JAX array."""
    return jnp.array([pnt.X(), pnt.Y(), pnt.Z()])


def _gp_dir_to_array(direction: Any) -> jnp.ndarray:
    """Convert an OCP gp_Dir to a JAX array."""
    return jnp.array([direction.X(), direction.Y(), direction.Z()])


def face_to_primitive(face: TopoDS_Face) -> Primitive | None:
    """Convert a single OCP face to a BRepAX Primitive.

    Maps the underlying surface type (plane, cylinder, sphere, cone, torus)
    to the corresponding BRepAX primitive, extracting geometric parameters
    via OCCT adaptor classes.

    Returns None with a warning for unsupported surface types (NURBS, etc.).

    Args:
        face: An OCCT topological face.

    Returns:
        A :class:`Primitive` instance, or ``None`` if the surface type
        is not supported.
    """
    adaptor = BRepAdaptor_Surface(face)
    stype = adaptor.GetType()

    if stype == GeomAbs_Plane:
        gp_plane = adaptor.Plane()
        point = _gp_pnt_to_array(gp_plane.Location())
        normal = _gp_dir_to_array(gp_plane.Axis().Direction())
        offset = jnp.dot(normal, point)
        return PlanePrim(normal=normal, offset=offset)

    if stype == GeomAbs_Cylinder:
        gp_cyl = adaptor.Cylinder()
        point = _gp_pnt_to_array(gp_cyl.Location())
        axis = _gp_dir_to_array(gp_cyl.Axis().Direction())
        radius = jnp.array(gp_cyl.Radius())
        return CylinderPrim(point=point, axis=axis, radius=radius)

    if stype == GeomAbs_Sphere:
        gp_sph = adaptor.Sphere()
        center = _gp_pnt_to_array(gp_sph.Location())
        radius = jnp.array(gp_sph.Radius())
        return SpherePrim(center=center, radius=radius)

    if stype == GeomAbs_Cone:
        gp_cone = adaptor.Cone()
        apex = _gp_pnt_to_array(gp_cone.Apex())
        axis = _gp_dir_to_array(gp_cone.Axis().Direction())
        angle = jnp.array(abs(gp_cone.SemiAngle()))
        return ConePrim(apex=apex, axis=axis, angle=angle)

    if stype == GeomAbs_Torus:
        gp_torus = adaptor.Torus()
        center = _gp_pnt_to_array(gp_torus.Location())
        axis = _gp_dir_to_array(gp_torus.Axis().Direction())
        major_radius = jnp.array(gp_torus.MajorRadius())
        minor_radius = jnp.array(gp_torus.MinorRadius())
        return TorusPrim(
            center=center,
            axis=axis,
            major_radius=major_radius,
            minor_radius=minor_radius,
        )

    if stype == GeomAbs_BSplineSurface:
        return _convert_bspline_face(adaptor, face)

    type_name = _SURFACE_TYPE_NAMES.get(stype, "unknown")
    warnings.warn(
        f"Unsupported surface type: {type_name}, skipping face",
        stacklevel=2,
    )
    return None


def _expand_knots(unique_knots: list[float], multiplicities: list[int]) -> jnp.ndarray:
    """Expand unique knots + multiplicities to repeated knot vector.

    OCCT stores knots as (unique values, multiplicities).  The De Boor
    algorithm expects the fully repeated form.
    """
    repeated: list[float] = []
    for knot, mult in zip(unique_knots, multiplicities, strict=True):
        repeated.extend([knot] * mult)
    return jnp.array(repeated)


_TRIM_SAMPLES_PER_EDGE = 8


def _precompute_coarse_grid(
    control_points: jnp.ndarray,
    knots_u: jnp.ndarray,
    knots_v: jnp.ndarray,
    degree_u: int,
    degree_v: int,
    weights: jnp.ndarray | None,
    sign_flip: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute surface samples and normals for fast sign estimation.

    Returns positions ``(G*G, 3)`` and orientation-corrected normals
    ``(G*G, 3)`` on a uniform parametric grid, where ``G`` is
    :data:`~brepax.nurbs.projection._COARSE_GRID`.
    """
    import jax

    from brepax.nurbs.evaluate import evaluate_surface_derivs
    from brepax.nurbs.projection import _COARSE_GRID

    u_lo, u_hi = knots_u[degree_u], knots_u[-degree_u - 1]
    v_lo, v_hi = knots_v[degree_v], knots_v[-degree_v - 1]
    us = jnp.linspace(u_lo, u_hi, _COARSE_GRID)
    vs = jnp.linspace(v_lo, v_hi, _COARSE_GRID)
    u_grid, v_grid = jnp.meshgrid(us, vs, indexing="ij")
    u_flat = u_grid.ravel()
    v_flat = v_grid.ravel()

    def _eval_derivs(u: jnp.ndarray, v: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        return evaluate_surface_derivs(
            control_points, knots_u, knots_v, degree_u, degree_v, u, v, weights
        )

    points, dus, dvs = jax.vmap(_eval_derivs)(u_flat, v_flat)
    normals = jnp.cross(dus, dvs)
    norms = jnp.linalg.norm(normals, axis=-1, keepdims=True) + 1e-10
    normals = normals / norms * sign_flip

    return points, normals


def _extract_trim_polygon(
    face: TopoDS_Face,
) -> np.ndarray | None:
    """Extract the outer wire of a face as a polyline in 2D parameter space.

    Traverses the wire edges in topological order using
    ``BRepTools_WireExplorer``, sampling each edge's pcurve at
    regular intervals.  Returns ``None`` for faces with fewer than
    3 edges (degenerate) or no outer wire.

    Returns:
        Array of shape ``(n_vertices, 2)`` with (u, v) coordinates,
        or ``None`` if the wire cannot be extracted.
    """
    wire = BRepTools.OuterWire_s(face)
    if wire.IsNull():
        return None

    vertices: list[list[float]] = []
    we = BRepTools_WireExplorer(wire, face)
    while we.More():
        edge = we.Current()
        curve2d = BRepAdaptor_Curve2d(edge, face)
        t0 = curve2d.FirstParameter()
        t1 = curve2d.LastParameter()
        is_forward = edge.Orientation() == TopAbs_FORWARD

        for i in range(_TRIM_SAMPLES_PER_EDGE):
            frac = i / _TRIM_SAMPLES_PER_EDGE
            t = t0 + (t1 - t0) * frac if is_forward else t1 - (t1 - t0) * frac
            pt = curve2d.Value(t)
            vertices.append([pt.X(), pt.Y()])

        we.Next()

    if len(vertices) < 3:
        return None
    return np.array(vertices)


def _convert_bspline_face(
    adaptor: Any, face: TopoDS_Face | None = None
) -> BSplinePrim | None:
    """Convert an OCCT B-spline surface to a BSplineSurface primitive.

    Extracts control points, knot vectors, and degree from the OCCT
    Geom_BSplineSurface handle.  Supports both non-rational and
    rational (weighted NURBS) surfaces.
    """
    bspl = adaptor.BSpline()
    is_rational = bspl.IsURational() or bspl.IsVRational()

    n_u = bspl.NbUPoles()
    n_v = bspl.NbVPoles()
    deg_u = bspl.UDegree()
    deg_v = bspl.VDegree()

    # Extract control points (OCCT uses 1-based indexing)
    poles = np.zeros((n_u, n_v, 3))
    for i in range(1, n_u + 1):
        for j in range(1, n_v + 1):
            pt = bspl.Pole(i, j)
            poles[i - 1, j - 1] = [pt.X(), pt.Y(), pt.Z()]

    # Extract knot vectors: OCCT (unique + multiplicities) → repeated
    from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal

    uk = TColStd_Array1OfReal(1, bspl.NbUKnots())
    um = TColStd_Array1OfInteger(1, bspl.NbUKnots())
    bspl.UKnots(uk)
    bspl.UMultiplicities(um)
    u_knots_unique = [uk.Value(i) for i in range(1, bspl.NbUKnots() + 1)]
    u_mults = [um.Value(i) for i in range(1, bspl.NbUKnots() + 1)]

    vk = TColStd_Array1OfReal(1, bspl.NbVKnots())
    vm = TColStd_Array1OfInteger(1, bspl.NbVKnots())
    bspl.VKnots(vk)
    bspl.VMultiplicities(vm)
    v_knots_unique = [vk.Value(i) for i in range(1, bspl.NbVKnots() + 1)]
    v_mults = [vm.Value(i) for i in range(1, bspl.NbVKnots() + 1)]

    knots_u = _expand_knots(u_knots_unique, u_mults)
    knots_v = _expand_knots(v_knots_unique, v_mults)

    # Extract weights for rational B-splines
    weights = None
    if is_rational:
        w = np.zeros((n_u, n_v))
        for i in range(1, n_u + 1):
            for j in range(1, n_v + 1):
                w[i - 1, j - 1] = bspl.Weight(i, j)
        weights = jnp.array(w)

    # Extract parametric trim bounds from the face adaptor
    eps_param = 1e-6
    u_full_lo, u_full_hi = float(knots_u[deg_u]), float(knots_u[-deg_u - 1])
    v_full_lo, v_full_hi = float(knots_v[deg_v]), float(knots_v[-deg_v - 1])
    u_face_lo = adaptor.FirstUParameter()
    u_face_hi = adaptor.LastUParameter()
    v_face_lo = adaptor.FirstVParameter()
    v_face_hi = adaptor.LastVParameter()

    param_u_range = None
    param_v_range = None
    if abs(u_face_lo - u_full_lo) > eps_param or abs(u_face_hi - u_full_hi) > eps_param:
        param_u_range = (u_face_lo, u_face_hi)
    if abs(v_face_lo - v_full_lo) > eps_param or abs(v_face_hi - v_full_hi) > eps_param:
        param_v_range = (v_face_lo, v_face_hi)

    # OCCT face orientation determines inside/outside convention.
    # BRepAdaptor reflects orientation for analytical surfaces but
    # returns the underlying parametrization unchanged for BSpline.
    sign_flip = 1.0
    if face is not None and face.Orientation() != TopAbs_FORWARD:
        sign_flip = -1.0

    # Precompute coarse grid samples for fast sign estimation in PMC
    cp_arr = jnp.array(poles)
    coarse_positions, coarse_normals = _precompute_coarse_grid(
        cp_arr, knots_u, knots_v, deg_u, deg_v, weights, sign_flip
    )

    # Extract 2D trim polygon for non-rectangular face boundaries
    trim_polygon = None
    trim_mask = None
    if face is not None:
        poly_np = _extract_trim_polygon(face)
        if poly_np is not None:
            trim_polygon = jnp.array(poly_np)
            trim_mask = jnp.ones(len(poly_np))

    return BSplinePrim(
        control_points=cp_arr,
        knots_u=knots_u,
        knots_v=knots_v,
        degree_u=deg_u,
        degree_v=deg_v,
        weights=weights,
        sign_flip=sign_flip,
        param_u_range=param_u_range,
        param_v_range=param_v_range,
        coarse_positions=coarse_positions,
        coarse_normals=coarse_normals,
        trim_polygon=trim_polygon,
        trim_mask=trim_mask,
    )


def faces_to_primitives(shape: TopoDS_Shape) -> list[Primitive | None]:
    """Convert all faces in a shape to BRepAX Primitives.

    Iterates over the topological faces of the shape and converts each
    to the corresponding BRepAX primitive.  Unsupported face types
    produce ``None`` entries in the returned list.

    Args:
        shape: An OCCT topological shape.

    Returns:
        A list with one entry per face.  Unsupported faces are ``None``.
    """
    primitives: list[Primitive | None] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        primitives.append(face_to_primitive(face))
        explorer.Next()
    return primitives


__all__ = [
    "ShapeMetadata",
    "face_to_primitive",
    "faces_to_primitives",
    "shape_metadata",
]
