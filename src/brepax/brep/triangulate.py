"""Triangulation of B-Rep faces for divergence theorem volume.

Uses OCCT BRepMesh for watertight mesh topology, then re-evaluates
vertex positions inside the JAX computation graph using parametric
surface functions.  This gives ``jax.grad`` flow from volume through
triangle vertices to primitive parameters (control points, radius,
etc.) while preserving mesh watertightness.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from brepax._occt.backend import (
    BRep_Tool,
    BRepAdaptor_Surface,
    BRepMesh_IncrementalMesh,
    GeomAbs_BSplineSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_SOLID,
    TopExp_Explorer,
    TopLoc_Location,
    TopoDS,
)
from brepax._occt.types import TopoDS_Shape
from brepax.nurbs.evaluate import evaluate_surface

_DEFAULT_DEFLECTION = 0.01


def _eval_plane(
    origin: jnp.ndarray,
    xdir: jnp.ndarray,
    ydir: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate plane at parametric coordinates (u, v)."""
    result: jnp.ndarray = origin + u * xdir + v * ydir
    return result


def _eval_cylinder(
    center: jnp.ndarray,
    xdir: jnp.ndarray,
    ydir: jnp.ndarray,
    axis: jnp.ndarray,
    radius: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate cylinder surface at (u=theta, v=height)."""
    result: jnp.ndarray = (
        center + radius * (jnp.cos(u) * xdir + jnp.sin(u) * ydir) + v * axis
    )
    return result


def _eval_sphere(
    center: jnp.ndarray,
    xdir: jnp.ndarray,
    ydir: jnp.ndarray,
    axis: jnp.ndarray,
    radius: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate sphere at (u=theta, v=phi)."""
    result: jnp.ndarray = center + radius * (
        jnp.cos(v) * jnp.cos(u) * xdir
        + jnp.cos(v) * jnp.sin(u) * ydir
        + jnp.sin(v) * axis
    )
    return result


def _eval_cone(
    location: jnp.ndarray,
    xdir: jnp.ndarray,
    ydir: jnp.ndarray,
    axis: jnp.ndarray,
    ref_radius: jnp.ndarray,
    semi_angle: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate cone at (u=theta, v=distance along ruling)."""
    r = ref_radius + v * jnp.sin(semi_angle)
    h = v * jnp.cos(semi_angle)
    result: jnp.ndarray = (
        location + r * (jnp.cos(u) * xdir + jnp.sin(u) * ydir) + h * axis
    )
    return result


def _eval_torus(
    center: jnp.ndarray,
    xdir: jnp.ndarray,
    ydir: jnp.ndarray,
    axis: jnp.ndarray,
    major_r: jnp.ndarray,
    minor_r: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate torus at (u=major angle, v=minor angle)."""
    radial = jnp.cos(u) * xdir + jnp.sin(u) * ydir
    ring_center = center + major_r * radial
    result: jnp.ndarray = ring_center + minor_r * (
        jnp.cos(v) * radial + jnp.sin(v) * axis
    )
    return result


def _extract_ax3(
    position: Any,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract origin, xdir, ydir, axis from an OCCT gp_Ax3 position."""
    loc = position.Location()
    origin = jnp.array([loc.X(), loc.Y(), loc.Z()])
    d = position.Direction()
    axis = jnp.array([d.X(), d.Y(), d.Z()])
    xd = position.XDirection()
    xdir = jnp.array([xd.X(), xd.Y(), xd.Z()])
    yd = position.YDirection()
    ydir = jnp.array([yd.X(), yd.Y(), yd.Z()])
    return origin, xdir, ydir, axis


def _make_face_eval(
    adaptor: Any,
) -> tuple[Any, dict[str, Any]] | None:
    """Build a JAX eval function for a face's surface type.

    Returns ``(eval_fn, params)`` where ``eval_fn(u, v)`` evaluates
    the surface at parametric coordinates, or ``None`` for unsupported
    surface types.
    """
    stype = adaptor.GetType()

    if stype == GeomAbs_Plane:
        gp = adaptor.Plane()
        origin, xdir, ydir, _axis = _extract_ax3(gp.Position())

        def eval_fn(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
            return _eval_plane(origin, xdir, ydir, u, v)

        return eval_fn, {"origin": origin}

    if stype == GeomAbs_Cylinder:
        gp = adaptor.Cylinder()
        center, xdir, ydir, axis = _extract_ax3(gp.Position())
        radius = jnp.array(gp.Radius())

        def eval_fn(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
            return _eval_cylinder(center, xdir, ydir, axis, radius, u, v)

        return eval_fn, {"center": center, "axis": axis, "radius": radius}

    if stype == GeomAbs_Sphere:
        gp = adaptor.Sphere()
        center, xdir, ydir, axis = _extract_ax3(gp.Position())
        radius = jnp.array(gp.Radius())

        def eval_fn(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
            return _eval_sphere(center, xdir, ydir, axis, radius, u, v)

        return eval_fn, {"center": center, "radius": radius}

    if stype == GeomAbs_Cone:
        gp = adaptor.Cone()
        location, xdir, ydir, axis = _extract_ax3(gp.Position())
        ref_radius = jnp.array(gp.RefRadius())
        semi_angle = jnp.array(gp.SemiAngle())

        def eval_fn(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
            return _eval_cone(location, xdir, ydir, axis, ref_radius, semi_angle, u, v)

        return eval_fn, {
            "location": location,
            "axis": axis,
            "ref_radius": ref_radius,
            "semi_angle": semi_angle,
        }

    if stype == GeomAbs_Torus:
        gp = adaptor.Torus()
        center, xdir, ydir, axis = _extract_ax3(gp.Position())
        major_r = jnp.array(gp.MajorRadius())
        minor_r = jnp.array(gp.MinorRadius())

        def eval_fn(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
            return _eval_torus(center, xdir, ydir, axis, major_r, minor_r, u, v)

        return eval_fn, {
            "center": center,
            "axis": axis,
            "major_radius": major_r,
            "minor_radius": minor_r,
        }

    if stype == GeomAbs_BSplineSurface:
        return _make_bspline_eval(adaptor)

    return None


def _make_bspline_eval(
    adaptor: Any,
) -> tuple[Any, dict[str, Any]]:
    """Build a JAX eval function for a BSpline face."""
    bspl = adaptor.BSpline()
    n_u, n_v = bspl.NbUPoles(), bspl.NbVPoles()
    deg_u, deg_v = bspl.UDegree(), bspl.VDegree()

    poles = np.zeros((n_u, n_v, 3))
    for i in range(1, n_u + 1):
        for j in range(1, n_v + 1):
            pt = bspl.Pole(i, j)
            poles[i - 1, j - 1] = [pt.X(), pt.Y(), pt.Z()]
    control_points = jnp.array(poles)

    from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal

    from brepax.brep.convert import _expand_knots

    uk = TColStd_Array1OfReal(1, bspl.NbUKnots())
    um = TColStd_Array1OfInteger(1, bspl.NbUKnots())
    bspl.UKnots(uk)
    bspl.UMultiplicities(um)
    knots_u = _expand_knots(
        [uk.Value(i) for i in range(1, bspl.NbUKnots() + 1)],
        [um.Value(i) for i in range(1, bspl.NbUKnots() + 1)],
    )

    vk = TColStd_Array1OfReal(1, bspl.NbVKnots())
    vm = TColStd_Array1OfInteger(1, bspl.NbVKnots())
    bspl.VKnots(vk)
    bspl.VMultiplicities(vm)
    knots_v = _expand_knots(
        [vk.Value(i) for i in range(1, bspl.NbVKnots() + 1)],
        [vm.Value(i) for i in range(1, bspl.NbVKnots() + 1)],
    )

    is_rational = bspl.IsURational() or bspl.IsVRational()
    weights = None
    if is_rational:
        w = np.zeros((n_u, n_v))
        for i in range(1, n_u + 1):
            for j in range(1, n_v + 1):
                w[i - 1, j - 1] = bspl.Weight(i, j)
        weights = jnp.array(w)

    def eval_fn(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return evaluate_surface(
            control_points, knots_u, knots_v, deg_u, deg_v, u, v, weights
        )

    params = {"control_points": control_points, "knots_u": knots_u, "knots_v": knots_v}
    if weights is not None:
        params["weights"] = weights
    return eval_fn, params


def triangulate_shape(
    shape: TopoDS_Shape,
    *,
    deflection: float = _DEFAULT_DEFLECTION,
) -> tuple[jnp.ndarray, list[dict[str, Any]]]:
    """Triangulate all faces of a shape with JAX-native vertices.

    Uses OCCT BRepMesh for watertight mesh topology, then re-evaluates
    each vertex position using JAX-native parametric surface functions.
    This ensures the divergence theorem gives correct volume while
    ``jax.grad`` flows through the vertex positions.

    Args:
        shape: An OCCT topological shape.
        deflection: Mesh deflection tolerance for OCCT tessellation.

    Returns:
        Tuple of ``(triangles, params_list)`` where triangles has
        shape ``(n_total, 3, 3)`` and params_list contains one
        parameter dict per face.
    """
    BRepMesh_IncrementalMesh(shape, deflection)

    all_tris: list[jnp.ndarray] = []
    all_params: list[dict[str, Any]] = []

    # Iterate faces per-Solid to exclude orphan faces/shells that
    # are not part of any solid and would break the divergence theorem.
    # Fall back to all faces when the shape contains no Solids
    # (e.g. a single surface or open shell).
    face_sources: list[Any] = []
    exp_solid = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp_solid.More():
        face_sources.append(TopoDS.Solid_s(exp_solid.Current()))
        exp_solid.Next()
    if not face_sources:
        face_sources.append(shape)

    for source in face_sources:
        exp = TopExp_Explorer(source, TopAbs_FACE)
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            adaptor = BRepAdaptor_Surface(face)

            result = _make_face_eval(adaptor)
            if result is None:
                exp.Next()
                continue

            eval_fn, params = result

            loc = TopLoc_Location()
            poly_tri = BRep_Tool.Triangulation_s(face, loc)
            if poly_tri is None:
                exp.Next()
                continue

            n_nodes = poly_tri.NbNodes()
            n_tris = poly_tri.NbTriangles()

            # Batch-extract UV coordinates and connectivity as numpy arrays
            us_np = np.empty(n_nodes)
            vs_np = np.empty(n_nodes)
            for i in range(1, n_nodes + 1):
                uv = poly_tri.UVNode(i)
                us_np[i - 1] = uv.X()
                vs_np[i - 1] = uv.Y()

            conn = np.empty((n_tris, 3), dtype=np.int32)
            for i in range(1, n_tris + 1):
                n1, n2, n3 = poly_tri.Triangle(i).Get()
                conn[i - 1] = [n1 - 1, n2 - 1, n3 - 1]

            # Re-evaluate vertex positions in JAX graph
            positions = jax.vmap(eval_fn)(jnp.array(us_np), jnp.array(vs_np))

            # Assemble triangles via fancy indexing (no Python per-triangle loop)
            reverse = face.Orientation() != TopAbs_FORWARD
            idx = conn[:, [0, 2, 1]] if reverse else conn
            face_tris = positions[idx]

            all_tris.append(face_tris)
            all_params.append(params)

            exp.Next()

    if not all_tris:
        return jnp.zeros((0, 3, 3)), []

    return jnp.concatenate(all_tris, axis=0), all_params


def divergence_volume(triangles: jnp.ndarray) -> jnp.ndarray:
    """Compute volume of a closed triangle mesh via the divergence theorem.

    Uses the signed-tetrahedra formula: each triangle and the origin
    form a tetrahedron whose signed volume is ``v0 . (v1 x v2) / 6``.
    The sum over all triangles gives the enclosed volume for a
    watertight mesh with outward-facing normals.

    The result is a polynomial function of vertex positions, so
    ``jax.grad`` gives exact gradients with no singularities.

    Args:
        triangles: Triangle vertices, shape ``(n, 3, 3)``.

    Returns:
        Enclosed volume (scalar).

    Examples:
        >>> shape = read_step("model.step")
        >>> tris, params = triangulate_shape(shape)
        >>> vol = divergence_volume(tris)
        >>> grad = jax.grad(divergence_volume)(tris)
    """
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    return jnp.sum(v0 * jnp.cross(v1, v2)) / 6.0


__all__ = [
    "divergence_volume",
    "triangulate_shape",
]
