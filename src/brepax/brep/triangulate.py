"""Triangulation of B-Rep faces for divergence theorem volume.

Uses OCCT BRepMesh for watertight mesh topology, then re-evaluates
vertex positions inside the JAX computation graph using parametric
surface functions.  This gives ``jax.grad`` flow from volume through
triangle vertices to primitive parameters (control points, radius,
etc.) while preserving mesh watertightness.
"""

from __future__ import annotations

import functools
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


# Module-level vmap-jit for each analytical surface type. Runtime args keep
# JIT cache hot across faces of the same type (663 fresh closures -> 5 jits).


@jax.jit
def _plane_vmap(
    origin: jnp.ndarray,
    xdir: jnp.ndarray,
    ydir: jnp.ndarray,
    us: jnp.ndarray,
    vs: jnp.ndarray,
) -> jnp.ndarray:
    return jax.vmap(lambda u, v: _eval_plane(origin, xdir, ydir, u, v))(us, vs)


@jax.jit
def _cylinder_vmap(
    center: jnp.ndarray,
    xdir: jnp.ndarray,
    ydir: jnp.ndarray,
    axis: jnp.ndarray,
    radius: jnp.ndarray,
    us: jnp.ndarray,
    vs: jnp.ndarray,
) -> jnp.ndarray:
    return jax.vmap(
        lambda u, v: _eval_cylinder(center, xdir, ydir, axis, radius, u, v)
    )(us, vs)


@jax.jit
def _sphere_vmap(
    center: jnp.ndarray,
    xdir: jnp.ndarray,
    ydir: jnp.ndarray,
    axis: jnp.ndarray,
    radius: jnp.ndarray,
    us: jnp.ndarray,
    vs: jnp.ndarray,
) -> jnp.ndarray:
    return jax.vmap(lambda u, v: _eval_sphere(center, xdir, ydir, axis, radius, u, v))(
        us, vs
    )


@jax.jit
def _cone_vmap(
    location: jnp.ndarray,
    xdir: jnp.ndarray,
    ydir: jnp.ndarray,
    axis: jnp.ndarray,
    ref_radius: jnp.ndarray,
    semi_angle: jnp.ndarray,
    us: jnp.ndarray,
    vs: jnp.ndarray,
) -> jnp.ndarray:
    return jax.vmap(
        lambda u, v: _eval_cone(
            location, xdir, ydir, axis, ref_radius, semi_angle, u, v
        )
    )(us, vs)


@jax.jit
def _torus_vmap(
    center: jnp.ndarray,
    xdir: jnp.ndarray,
    ydir: jnp.ndarray,
    axis: jnp.ndarray,
    major_r: jnp.ndarray,
    minor_r: jnp.ndarray,
    us: jnp.ndarray,
    vs: jnp.ndarray,
) -> jnp.ndarray:
    return jax.vmap(
        lambda u, v: _eval_torus(center, xdir, ydir, axis, major_r, minor_r, u, v)
    )(us, vs)


@functools.cache
def _get_bspline_vmap(deg_u: int, deg_v: int, is_rational: bool) -> Any:
    """One JIT per (deg_u, deg_v, rational). Runtime args carry face-specific data."""
    if is_rational:

        @jax.jit
        def _fn(
            control_points: jnp.ndarray,
            knots_u: jnp.ndarray,
            knots_v: jnp.ndarray,
            weights: jnp.ndarray,
            us: jnp.ndarray,
            vs: jnp.ndarray,
        ) -> jnp.ndarray:
            return jax.vmap(
                lambda u, v: evaluate_surface(
                    control_points, knots_u, knots_v, deg_u, deg_v, u, v, weights
                )
            )(us, vs)

        return _fn

    @jax.jit
    def _fn_nr(
        control_points: jnp.ndarray,
        knots_u: jnp.ndarray,
        knots_v: jnp.ndarray,
        us: jnp.ndarray,
        vs: jnp.ndarray,
    ) -> jnp.ndarray:
        return jax.vmap(
            lambda u, v: evaluate_surface(
                control_points, knots_u, knots_v, deg_u, deg_v, u, v, None
            )
        )(us, vs)

    return _fn_nr


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

    Returns ``(eval_fn, params)`` where ``eval_fn(us, vs)`` evaluates
    the surface at a batch of parametric coordinates, or ``None`` for
    unsupported surface types. ``eval_fn`` forwards to a module-level
    jitted function so the JIT cache is shared across faces of the
    same type / BSpline signature.
    """
    stype = adaptor.GetType()

    if stype == GeomAbs_Plane:
        gp = adaptor.Plane()
        origin, xdir, ydir, _axis = _extract_ax3(gp.Position())

        def eval_fn(us: jnp.ndarray, vs: jnp.ndarray) -> jnp.ndarray:
            out: jnp.ndarray = _plane_vmap(origin, xdir, ydir, us, vs)
            return out

        return eval_fn, {"origin": origin}

    if stype == GeomAbs_Cylinder:
        gp = adaptor.Cylinder()
        center, xdir, ydir, axis = _extract_ax3(gp.Position())
        radius = jnp.array(gp.Radius())

        def eval_fn(us: jnp.ndarray, vs: jnp.ndarray) -> jnp.ndarray:
            out: jnp.ndarray = _cylinder_vmap(center, xdir, ydir, axis, radius, us, vs)
            return out

        return eval_fn, {"center": center, "axis": axis, "radius": radius}

    if stype == GeomAbs_Sphere:
        gp = adaptor.Sphere()
        center, xdir, ydir, axis = _extract_ax3(gp.Position())
        radius = jnp.array(gp.Radius())

        def eval_fn(us: jnp.ndarray, vs: jnp.ndarray) -> jnp.ndarray:
            out: jnp.ndarray = _sphere_vmap(center, xdir, ydir, axis, radius, us, vs)
            return out

        return eval_fn, {"center": center, "radius": radius}

    if stype == GeomAbs_Cone:
        gp = adaptor.Cone()
        location, xdir, ydir, axis = _extract_ax3(gp.Position())
        ref_radius = jnp.array(gp.RefRadius())
        semi_angle = jnp.array(gp.SemiAngle())

        def eval_fn(us: jnp.ndarray, vs: jnp.ndarray) -> jnp.ndarray:
            out: jnp.ndarray = _cone_vmap(
                location, xdir, ydir, axis, ref_radius, semi_angle, us, vs
            )
            return out

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

        def eval_fn(us: jnp.ndarray, vs: jnp.ndarray) -> jnp.ndarray:
            out: jnp.ndarray = _torus_vmap(
                center, xdir, ydir, axis, major_r, minor_r, us, vs
            )
            return out

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

    shared = _get_bspline_vmap(deg_u, deg_v, weights is not None)

    if weights is not None:
        weights_arr: jnp.ndarray = weights

        def eval_fn(us: jnp.ndarray, vs: jnp.ndarray) -> jnp.ndarray:
            out: jnp.ndarray = shared(
                control_points, knots_u, knots_v, weights_arr, us, vs
            )
            return out
    else:

        def eval_fn(us: jnp.ndarray, vs: jnp.ndarray) -> jnp.ndarray:
            out: jnp.ndarray = shared(control_points, knots_u, knots_v, us, vs)
            return out

    params: dict[str, Any] = {
        "control_points": control_points,
        "knots_u": knots_u,
        "knots_v": knots_v,
        "deg_u": deg_u,
        "deg_v": deg_v,
    }
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

            # Re-evaluate vertex positions via module-level jit (cache-shared
            # across faces of the same surface type / BSpline signature).
            positions = eval_fn(jnp.array(us_np), jnp.array(vs_np))

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


def mesh_surface_area(triangles: jnp.ndarray) -> jnp.ndarray:
    """Compute surface area of a triangle mesh.

    Sums the area of each triangle: ``(1/2) ||cross(e1, e2)||``.
    Differentiable via ``jax.grad``.

    Args:
        triangles: Triangle vertices, shape ``(n, 3, 3)``.

    Returns:
        Total surface area (scalar).

    Examples:
        >>> tris, _ = triangulate_shape(shape)
        >>> area = mesh_surface_area(tris)
    """
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    cross = jnp.cross(e1, e2)
    return jnp.sum(jnp.sqrt(jnp.sum(cross**2, axis=-1) + 1e-20)) / 2.0


def mesh_center_of_mass(triangles: jnp.ndarray) -> jnp.ndarray:
    """Compute center of mass of a closed triangle mesh.

    Uses the divergence theorem with ``F = (x^2/2, 0, 0)`` etc.
    to compute first moments, then divides by volume.  Polynomial
    in vertex positions (degree 4 ratio).

    Args:
        triangles: Triangle vertices, shape ``(n, 3, 3)``.

    Returns:
        Center of mass, shape ``(3,)``.

    Examples:
        >>> tris, _ = triangulate_shape(shape)
        >>> com = mesh_center_of_mass(tris)
    """
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    vol = jnp.sum(v0 * jnp.cross(v1, v2)) / 6.0
    n = jnp.cross(v1 - v0, v2 - v0)
    sq_sum = v0**2 + v1**2 + v2**2 + v0 * v1 + v0 * v2 + v1 * v2
    moments = jnp.sum(n * sq_sum, axis=0) / 24.0
    return moments / (vol + 1e-20)


def mesh_inertia_tensor(triangles: jnp.ndarray) -> jnp.ndarray:
    """Compute the inertia tensor of a closed triangle mesh about its center of mass.

    Uses the divergence theorem to compute second moments via
    surface integrals (Tonon 2004 / Eberly 2002), then applies
    the parallel axis theorem to shift from origin to CoM.
    Polynomial in vertex positions (degree 5).

    Assumes uniform density ``rho = 1``.

    Args:
        triangles: Triangle vertices, shape ``(n, 3, 3)``.

    Returns:
        Symmetric 3x3 inertia tensor about center of mass.

    Examples:
        >>> tris, _ = triangulate_shape(shape)
        >>> I = mesh_inertia_tensor(tris)
    """
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    n = jnp.cross(b - a, c - a)

    vol = jnp.sum(a * jnp.cross(b, c)) / 6.0

    # Third-order canonical integrals per axis (Tonon 2004)
    f3 = (
        a**3
        + b**3
        + c**3
        + a**2 * b
        + a**2 * c
        + b**2 * a
        + b**2 * c
        + c**2 * a
        + c**2 * b
        + a * b * c
    )
    # Second moments about origin: integral of x_k^2 dV
    xx = jnp.sum(n[:, 0] * f3[:, 0]) / 60.0
    yy = jnp.sum(n[:, 1] * f3[:, 1]) / 60.0
    zz = jnp.sum(n[:, 2] * f3[:, 2]) / 60.0

    # Cross-moment terms about origin
    def _cross_moment(
        ai: jnp.ndarray,
        bi: jnp.ndarray,
        ci: jnp.ndarray,
        aj: jnp.ndarray,
        bj: jnp.ndarray,
        cj: jnp.ndarray,
        nk: jnp.ndarray,
    ) -> jnp.ndarray:
        return (
            jnp.sum(
                nk
                * (
                    2 * ai * aj
                    + 2 * bi * bj
                    + 2 * ci * cj
                    + ai * bj
                    + ai * cj
                    + bi * aj
                    + bi * cj
                    + ci * aj
                    + ci * bj
                )
            )
            / 120.0
        )

    xy = _cross_moment(a[:, 0], b[:, 0], c[:, 0], a[:, 1], b[:, 1], c[:, 1], n[:, 2])
    yz = _cross_moment(a[:, 1], b[:, 1], c[:, 1], a[:, 2], b[:, 2], c[:, 2], n[:, 0])
    xz = _cross_moment(a[:, 0], b[:, 0], c[:, 0], a[:, 2], b[:, 2], c[:, 2], n[:, 1])

    # Inertia about origin
    i_origin = jnp.array(
        [
            [yy + zz, -xy, -xz],
            [-xy, xx + zz, -yz],
            [-xz, -yz, xx + yy],
        ]
    )

    # Parallel axis theorem: I_com = I_origin - M * (|r|^2 I - r outer r)
    com = mesh_center_of_mass(triangles)
    r2 = jnp.dot(com, com)
    shift = vol * (r2 * jnp.eye(3) - jnp.outer(com, com))
    return i_origin - shift


_SURFACE_TYPE_NAMES = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Sphere: "sphere",
    GeomAbs_Cone: "cone",
    GeomAbs_Torus: "torus",
    GeomAbs_BSplineSurface: "bspline",
}


def _extract_face_data(
    adaptor: Any,
) -> dict[str, Any] | None:
    """Extract surface type and parameters from an OCCT face adaptor.

    Returns a dict with ``surface_type`` (str) and type-specific
    parameters as JAX arrays, or ``None`` for unsupported types.
    """
    stype = adaptor.GetType()
    name = _SURFACE_TYPE_NAMES.get(stype)
    if name is None:
        return None

    if name == "plane":
        gp = adaptor.Plane()
        origin, xdir, ydir, _axis = _extract_ax3(gp.Position())
        return {"surface_type": name, "origin": origin, "xdir": xdir, "ydir": ydir}

    if name == "cylinder":
        gp = adaptor.Cylinder()
        center, xdir, ydir, axis = _extract_ax3(gp.Position())
        return {
            "surface_type": name,
            "center": center,
            "xdir": xdir,
            "ydir": ydir,
            "axis": axis,
            "radius": jnp.array(gp.Radius()),
        }

    if name == "sphere":
        gp = adaptor.Sphere()
        center, xdir, ydir, axis = _extract_ax3(gp.Position())
        return {
            "surface_type": name,
            "center": center,
            "xdir": xdir,
            "ydir": ydir,
            "axis": axis,
            "radius": jnp.array(gp.Radius()),
        }

    if name == "cone":
        gp = adaptor.Cone()
        location, xdir, ydir, axis = _extract_ax3(gp.Position())
        return {
            "surface_type": name,
            "location": location,
            "xdir": xdir,
            "ydir": ydir,
            "axis": axis,
            "ref_radius": jnp.array(gp.RefRadius()),
            "semi_angle": jnp.array(gp.SemiAngle()),
        }

    if name == "torus":
        gp = adaptor.Torus()
        center, xdir, ydir, axis = _extract_ax3(gp.Position())
        return {
            "surface_type": name,
            "center": center,
            "xdir": xdir,
            "ydir": ydir,
            "axis": axis,
            "major_radius": jnp.array(gp.MajorRadius()),
            "minor_radius": jnp.array(gp.MinorRadius()),
        }

    if name == "bspline":
        _, bspline_params = _make_bspline_eval(adaptor)
        bspline_params["surface_type"] = name
        return bspline_params

    return None


def _evaluate_face_at(
    surface_type: str,
    params: dict[str, Any],
    u: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate a surface point given explicit parameters.

    Unlike ``_make_face_eval`` closures, all parameters are explicit
    arguments so that ``jax.grad`` can flow through them.
    """
    if surface_type == "plane":
        return _eval_plane(params["origin"], params["xdir"], params["ydir"], u, v)
    if surface_type == "cylinder":
        return _eval_cylinder(
            params["center"],
            params["xdir"],
            params["ydir"],
            params["axis"],
            params["radius"],
            u,
            v,
        )
    if surface_type == "sphere":
        return _eval_sphere(
            params["center"],
            params["xdir"],
            params["ydir"],
            params["axis"],
            params["radius"],
            u,
            v,
        )
    if surface_type == "cone":
        return _eval_cone(
            params["location"],
            params["xdir"],
            params["ydir"],
            params["axis"],
            params["ref_radius"],
            params["semi_angle"],
            u,
            v,
        )
    if surface_type == "torus":
        return _eval_torus(
            params["center"],
            params["xdir"],
            params["ydir"],
            params["axis"],
            params["major_radius"],
            params["minor_radius"],
            u,
            v,
        )
    if surface_type == "bspline":
        return evaluate_surface(
            params["control_points"],
            params["knots_u"],
            params["knots_v"],
            params["deg_u"],
            params["deg_v"],
            u,
            v,
            params.get("weights"),
        )
    msg = f"Unsupported surface type: {surface_type}"
    raise ValueError(msg)


def extract_mesh_topology(
    shape: TopoDS_Shape,
    *,
    deflection: float = _DEFAULT_DEFLECTION,
) -> list[dict[str, Any]]:
    """Extract watertight mesh topology from an OCCT shape.

    Tessellates the shape via OCCT BRepMesh, then extracts per-face
    UV coordinates, triangle connectivity, and surface parameters.
    The returned data is static (not in the JAX graph) and serves
    as input to :func:`evaluate_mesh` for differentiable vertex
    re-evaluation.

    Args:
        shape: An OCCT topological shape.
        deflection: Mesh deflection tolerance for OCCT tessellation.

    Returns:
        List of face dicts, each containing ``us``, ``vs`` (numpy),
        ``conn`` (numpy int32), ``reverse`` (bool), and surface
        parameters (JAX arrays).

    Examples:
        >>> topology = extract_mesh_topology(shape)
        >>> triangles = evaluate_mesh(topology)
        >>> vol = divergence_volume(triangles)
    """
    BRepMesh_IncrementalMesh(shape, deflection)
    faces: list[dict[str, Any]] = []

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

            face_data = _extract_face_data(adaptor)
            if face_data is None:
                exp.Next()
                continue

            loc = TopLoc_Location()
            poly_tri = BRep_Tool.Triangulation_s(face, loc)
            if poly_tri is None:
                exp.Next()
                continue

            n_nodes = poly_tri.NbNodes()
            n_tris = poly_tri.NbTriangles()

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

            face_data["us"] = us_np
            face_data["vs"] = vs_np
            face_data["conn"] = conn
            face_data["reverse"] = face.Orientation() != TopAbs_FORWARD

            # For plane faces, store the max UV distance from origin.
            # This is used by evaluate_mesh to scale cap vertices when
            # a linked parameter (e.g. cylinder radius) changes.
            if face_data["surface_type"] == "plane" and us_np.size > 0:
                extent = float(np.max(np.sqrt(us_np**2 + vs_np**2)))
                face_data["uv_scale_ref"] = extent

            faces.append(face_data)
            exp.Next()

    return faces


def evaluate_mesh(
    topology: list[dict[str, Any]],
    overrides: dict[str, jnp.ndarray] | None = None,
    *,
    uv_scale_param: str | None = None,
) -> jnp.ndarray:
    """Re-evaluate mesh vertices from topology with parameter overrides.

    For each face, evaluates the surface function at the stored UV
    coordinates using surface parameters.  Any parameter in
    ``overrides`` replaces the corresponding stored value, allowing
    ``jax.grad`` to flow from volume through vertices to the
    overridden design parameter.

    When ``uv_scale_param`` is set, plane faces scale their UV
    coordinates by ``overrides[uv_scale_param] / uv_scale_ref`` so
    that flat caps follow curved-surface parameter changes (e.g.
    disk caps scale with cylinder radius).

    Args:
        topology: Face list from :func:`extract_mesh_topology`.
        overrides: Parameter name-value pairs to override.
        uv_scale_param: Override key used to scale plane UV
            coordinates.  Required for multi-face optimization
            where flat caps must track a curved-surface parameter.

    Returns:
        Triangle vertices, shape ``(n_total, 3, 3)``.

    Examples:
        >>> topology = extract_mesh_topology(cylinder_shape)
        >>> def volume_fn(radius):
        ...     return divergence_volume(
        ...         evaluate_mesh(topology, {"radius": radius},
        ...                       uv_scale_param="radius")
        ...     )
        >>> grad = jax.grad(volume_fn)(jnp.array(5.0))
    """
    if overrides is None:
        overrides = {}

    all_tris: list[jnp.ndarray] = []

    for face in topology:
        stype = face["surface_type"]
        us = jnp.array(face["us"])
        vs = jnp.array(face["vs"])
        conn = face["conn"]
        reverse = face["reverse"]

        # Build effective params: stored values + overrides
        params = {
            k: v
            for k, v in face.items()
            if k not in ("surface_type", "us", "vs", "conn", "reverse", "uv_scale_ref")
        }

        if stype == "plane" and uv_scale_param and uv_scale_param in overrides:
            ref = face.get("uv_scale_ref", 1.0)
            if ref > 1e-10:
                scale = overrides[uv_scale_param] / ref
                us = us * scale
                vs = vs * scale

        if overrides:
            params.update(overrides)

        positions = jax.vmap(
            lambda u, v, _st=stype, _p=params: _evaluate_face_at(_st, _p, u, v)
        )(us, vs)

        idx = conn[:, [0, 2, 1]] if reverse else conn
        all_tris.append(positions[idx])

    if not all_tris:
        return jnp.zeros((0, 3, 3))

    return jnp.concatenate(all_tris, axis=0)


__all__ = [
    "divergence_volume",
    "evaluate_mesh",
    "extract_mesh_topology",
    "mesh_center_of_mass",
    "mesh_inertia_tensor",
    "mesh_surface_area",
    "triangulate_shape",
]
