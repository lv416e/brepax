"""Triangulation of B-Rep faces for divergence theorem volume.

Uses OCCT BRepMesh for watertight mesh topology, then re-evaluates
vertex positions inside the JAX computation graph using parametric
surface functions.  This gives ``jax.grad`` flow from volume through
triangle vertices to primitive parameters (control points, radius,
etc.) while preserving mesh watertightness.

``triangulate_shape`` evaluates each triangle corner from its own
face's parameters independently, so a vertex on a shared edge gets two
numerically close but distinct copies: watertight as a surface
covering, but not at the index level. ``triangulate_shape_watertight``
additionally welds those copies into one vertex per physical point,
identified from OCCT edge topology rather than a distance tolerance;
``extract_watertight_topology`` and ``evaluate_watertight_mesh`` split
that into frozen host-side topology and a pure JAX re-evaluation
function, for differentiating with respect to specific per-face
surface parameters without repeating the OCCT work. See their
docstrings and ADR-0024 for the design rationale.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

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
    TColStd_Array1OfInteger,
    TColStd_Array1OfReal,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopExp,
    TopExp_Explorer,
    TopLoc_Location,
    TopoDS,
    TopTools_IndexedMapOfShape,
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


# Module-level vmap-of-vmap JITs: batch N faces of the same type/signature
# into a single XLA dispatch. Each call returns (N_faces, max_points, 3).


@jax.jit
def _plane_batched_vmap(
    origins: jnp.ndarray,
    xdirs: jnp.ndarray,
    ydirs: jnp.ndarray,
    us_mat: jnp.ndarray,
    vs_mat: jnp.ndarray,
) -> jnp.ndarray:
    out: jnp.ndarray = jax.vmap(_plane_vmap)(origins, xdirs, ydirs, us_mat, vs_mat)
    return out


@jax.jit
def _cylinder_batched_vmap(
    centers: jnp.ndarray,
    xdirs: jnp.ndarray,
    ydirs: jnp.ndarray,
    axes: jnp.ndarray,
    radii: jnp.ndarray,
    us_mat: jnp.ndarray,
    vs_mat: jnp.ndarray,
) -> jnp.ndarray:
    out: jnp.ndarray = jax.vmap(_cylinder_vmap)(
        centers, xdirs, ydirs, axes, radii, us_mat, vs_mat
    )
    return out


@jax.jit
def _sphere_batched_vmap(
    centers: jnp.ndarray,
    xdirs: jnp.ndarray,
    ydirs: jnp.ndarray,
    axes: jnp.ndarray,
    radii: jnp.ndarray,
    us_mat: jnp.ndarray,
    vs_mat: jnp.ndarray,
) -> jnp.ndarray:
    out: jnp.ndarray = jax.vmap(_sphere_vmap)(
        centers, xdirs, ydirs, axes, radii, us_mat, vs_mat
    )
    return out


@jax.jit
def _cone_batched_vmap(
    locations: jnp.ndarray,
    xdirs: jnp.ndarray,
    ydirs: jnp.ndarray,
    axes: jnp.ndarray,
    ref_radii: jnp.ndarray,
    semi_angles: jnp.ndarray,
    us_mat: jnp.ndarray,
    vs_mat: jnp.ndarray,
) -> jnp.ndarray:
    out: jnp.ndarray = jax.vmap(_cone_vmap)(
        locations, xdirs, ydirs, axes, ref_radii, semi_angles, us_mat, vs_mat
    )
    return out


@jax.jit
def _torus_batched_vmap(
    centers: jnp.ndarray,
    xdirs: jnp.ndarray,
    ydirs: jnp.ndarray,
    axes: jnp.ndarray,
    major_rs: jnp.ndarray,
    minor_rs: jnp.ndarray,
    us_mat: jnp.ndarray,
    vs_mat: jnp.ndarray,
) -> jnp.ndarray:
    out: jnp.ndarray = jax.vmap(_torus_vmap)(
        centers, xdirs, ydirs, axes, major_rs, minor_rs, us_mat, vs_mat
    )
    return out


@functools.cache
def _get_bspline_batched_vmap(deg_u: int, deg_v: int, is_rational: bool) -> Any:
    """Batched vmap over N faces sharing the same BSpline signature."""
    per_face = _get_bspline_vmap(deg_u, deg_v, is_rational)

    if is_rational:

        @jax.jit
        def _fn(
            control_points_b: jnp.ndarray,
            knots_u_b: jnp.ndarray,
            knots_v_b: jnp.ndarray,
            weights_b: jnp.ndarray,
            us_mat: jnp.ndarray,
            vs_mat: jnp.ndarray,
        ) -> jnp.ndarray:
            out: jnp.ndarray = jax.vmap(per_face)(
                control_points_b, knots_u_b, knots_v_b, weights_b, us_mat, vs_mat
            )
            return out

        return _fn

    @jax.jit
    def _fn_nr(
        control_points_b: jnp.ndarray,
        knots_u_b: jnp.ndarray,
        knots_v_b: jnp.ndarray,
        us_mat: jnp.ndarray,
        vs_mat: jnp.ndarray,
    ) -> jnp.ndarray:
        out: jnp.ndarray = jax.vmap(per_face)(
            control_points_b, knots_u_b, knots_v_b, us_mat, vs_mat
        )
        return out

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


def _extract_face_geometric_params(adaptor: Any) -> tuple[str, dict[str, Any]] | None:
    """Extract surface type and raw geometric parameters for batching.

    Separates the analytical parameter tuple from the JIT-wrapping
    responsibility so ``triangulate_shape`` can group and stack
    parameters across faces before any JAX dispatch.
    """
    stype = adaptor.GetType()

    if stype == GeomAbs_Plane:
        origin, xdir, ydir, _ = _extract_ax3(adaptor.Plane().Position())
        return "plane", {"origin": origin, "xdir": xdir, "ydir": ydir}

    if stype == GeomAbs_Cylinder:
        gp = adaptor.Cylinder()
        center, xdir, ydir, axis = _extract_ax3(gp.Position())
        return "cylinder", {
            "center": center,
            "xdir": xdir,
            "ydir": ydir,
            "axis": axis,
            "radius": jnp.array(gp.Radius()),
        }

    if stype == GeomAbs_Sphere:
        gp = adaptor.Sphere()
        center, xdir, ydir, axis = _extract_ax3(gp.Position())
        return "sphere", {
            "center": center,
            "xdir": xdir,
            "ydir": ydir,
            "axis": axis,
            "radius": jnp.array(gp.Radius()),
        }

    if stype == GeomAbs_Cone:
        gp = adaptor.Cone()
        location, xdir, ydir, axis = _extract_ax3(gp.Position())
        return "cone", {
            "location": location,
            "xdir": xdir,
            "ydir": ydir,
            "axis": axis,
            "ref_radius": jnp.array(gp.RefRadius()),
            "semi_angle": jnp.array(gp.SemiAngle()),
        }

    if stype == GeomAbs_Torus:
        gp = adaptor.Torus()
        center, xdir, ydir, axis = _extract_ax3(gp.Position())
        return "torus", {
            "center": center,
            "xdir": xdir,
            "ydir": ydir,
            "axis": axis,
            "major_radius": jnp.array(gp.MajorRadius()),
            "minor_radius": jnp.array(gp.MinorRadius()),
        }

    if stype == GeomAbs_BSplineSurface:
        from brepax.brep.convert import _clamped_bspline_surface, _expand_knots

        # The flat-knot evaluator needs the clamped form of periodic surfaces.
        bspl = _clamped_bspline_surface(adaptor.BSpline())
        n_u, n_v = bspl.NbUPoles(), bspl.NbVPoles()
        deg_u, deg_v = bspl.UDegree(), bspl.VDegree()

        poles = np.zeros((n_u, n_v, 3))
        for i in range(1, n_u + 1):
            for j in range(1, n_v + 1):
                pt = bspl.Pole(i, j)
                poles[i - 1, j - 1] = [pt.X(), pt.Y(), pt.Z()]

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
        weights_arr: jnp.ndarray | None = None
        if is_rational:
            w = np.zeros((n_u, n_v))
            for i in range(1, n_u + 1):
                for j in range(1, n_v + 1):
                    w[i - 1, j - 1] = bspl.Weight(i, j)
            weights_arr = jnp.array(w)

        bspline_params: dict[str, Any] = {
            "control_points": jnp.array(poles),
            "knots_u": knots_u,
            "knots_v": knots_v,
            "deg_u": deg_u,
            "deg_v": deg_v,
        }
        if weights_arr is not None:
            bspline_params["weights"] = weights_arr
        return "bspline", bspline_params

    return None


def _face_group_key(surface_type: str, params: dict[str, Any]) -> tuple[Any, ...]:
    """Signature that makes two faces share a batched JIT compile."""
    if surface_type == "bspline":
        return (
            "bspline",
            params["deg_u"],
            params["deg_v"],
            params["control_points"].shape,
            params["knots_u"].shape,
            params["knots_v"].shape,
            "weights" in params,
        )
    return (surface_type,)


def _dispatch_analytical_group(
    surface_type: str,
    faces_params: list[dict[str, Any]],
    us_mat: jnp.ndarray,
    vs_mat: jnp.ndarray,
) -> jnp.ndarray:
    out: jnp.ndarray
    if surface_type == "plane":
        origins = jnp.stack([p["origin"] for p in faces_params])
        xdirs = jnp.stack([p["xdir"] for p in faces_params])
        ydirs = jnp.stack([p["ydir"] for p in faces_params])
        out = _plane_batched_vmap(origins, xdirs, ydirs, us_mat, vs_mat)
        return out
    if surface_type == "cylinder":
        centers = jnp.stack([p["center"] for p in faces_params])
        xdirs = jnp.stack([p["xdir"] for p in faces_params])
        ydirs = jnp.stack([p["ydir"] for p in faces_params])
        axes = jnp.stack([p["axis"] for p in faces_params])
        radii = jnp.stack([p["radius"] for p in faces_params])
        out = _cylinder_batched_vmap(centers, xdirs, ydirs, axes, radii, us_mat, vs_mat)
        return out
    if surface_type == "sphere":
        centers = jnp.stack([p["center"] for p in faces_params])
        xdirs = jnp.stack([p["xdir"] for p in faces_params])
        ydirs = jnp.stack([p["ydir"] for p in faces_params])
        axes = jnp.stack([p["axis"] for p in faces_params])
        radii = jnp.stack([p["radius"] for p in faces_params])
        out = _sphere_batched_vmap(centers, xdirs, ydirs, axes, radii, us_mat, vs_mat)
        return out
    if surface_type == "cone":
        locations = jnp.stack([p["location"] for p in faces_params])
        xdirs = jnp.stack([p["xdir"] for p in faces_params])
        ydirs = jnp.stack([p["ydir"] for p in faces_params])
        axes = jnp.stack([p["axis"] for p in faces_params])
        ref_radii = jnp.stack([p["ref_radius"] for p in faces_params])
        semi_angles = jnp.stack([p["semi_angle"] for p in faces_params])
        out = _cone_batched_vmap(
            locations, xdirs, ydirs, axes, ref_radii, semi_angles, us_mat, vs_mat
        )
        return out
    if surface_type == "torus":
        centers = jnp.stack([p["center"] for p in faces_params])
        xdirs = jnp.stack([p["xdir"] for p in faces_params])
        ydirs = jnp.stack([p["ydir"] for p in faces_params])
        axes = jnp.stack([p["axis"] for p in faces_params])
        major_rs = jnp.stack([p["major_radius"] for p in faces_params])
        minor_rs = jnp.stack([p["minor_radius"] for p in faces_params])
        out = _torus_batched_vmap(
            centers, xdirs, ydirs, axes, major_rs, minor_rs, us_mat, vs_mat
        )
        return out
    raise ValueError(f"Unknown analytical surface type: {surface_type}")


def _dispatch_bspline_group(
    faces_params: list[dict[str, Any]],
    us_mat: jnp.ndarray,
    vs_mat: jnp.ndarray,
) -> jnp.ndarray:
    # All faces in the group share the same signature (shapes and degrees).
    first = faces_params[0]
    deg_u = first["deg_u"]
    deg_v = first["deg_v"]
    is_rational = "weights" in first
    fn = _get_bspline_batched_vmap(deg_u, deg_v, is_rational)

    cp_b = jnp.stack([p["control_points"] for p in faces_params])
    ku_b = jnp.stack([p["knots_u"] for p in faces_params])
    kv_b = jnp.stack([p["knots_v"] for p in faces_params])
    out: jnp.ndarray
    if is_rational:
        w_b = jnp.stack([p["weights"] for p in faces_params])
        out = fn(cp_b, ku_b, kv_b, w_b, us_mat, vs_mat)
    else:
        out = fn(cp_b, ku_b, kv_b, us_mat, vs_mat)
    return out


def per_face_geometric_params(shape: TopoDS_Shape) -> list[dict[str, Any]]:
    """Per-face primitive parameters in OCCT face traversal order.

    Walks each Solid's face exporer once and calls
    :func:`_extract_face_geometric_params` for each face.  Returns
    the same per-face dicts that :func:`triangulate_shape` puts in
    its ``params_list`` (with ``surface_type`` included), but without
    triggering ``BRepMesh_IncrementalMesh``, the JAX-side vertex
    evaluation, or the cross-face concatenation that
    :func:`triangulate_shape` performs.

    Suited to face-level metrics that only need the analytical
    primitive parameters (radius, axis, etc.) and not the
    triangulation itself — for example,
    :func:`~brepax.metrics.curvature.mean_curvature_per_face` reads
    only ``surface_type`` and the primitive radii.

    The returned ``params_list`` does not include ``n_triangles``
    because no triangulation is performed.  Callers that need a
    mesh as well should call :func:`triangulate_shape` directly.

    Args:
        shape: An OCCT topological shape.  Faces are iterated in the
            same per-Solid order as :func:`triangulate_shape`.

    Returns:
        List of per-face parameter dicts; each dict has at least a
        ``surface_type`` key and the analytical-primitive parameters
        produced by :func:`_extract_face_geometric_params`.

    Raises:
        NotImplementedError: An evaluated face has an unsupported surface type.
    """
    face_sources: list[Any] = []
    exp_solid = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp_solid.More():
        face_sources.append(TopoDS.Solid_s(exp_solid.Current()))
        exp_solid.Next()
    if not face_sources:
        face_sources.append(shape)

    out: list[dict[str, Any]] = []
    for source in face_sources:
        exp = TopExp_Explorer(source, TopAbs_FACE)
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            adaptor = BRepAdaptor_Surface(face)
            result = _extract_face_geometric_params(adaptor)
            if result is None:
                msg = f"Unsupported surface type: {adaptor.GetType().name}"
                raise NotImplementedError(msg)
            surface_type, params = result
            out.append({"surface_type": surface_type, **params})
            exp.Next()
    return out


def _extract_face_records(
    shape: TopoDS_Shape, deflection: float
) -> list[dict[str, Any]]:
    """Mesh ``shape`` and extract per-face UV/connectivity/params in numpy.

    Shared host-side extraction pass ("Pass 1") for :func:`triangulate_shape`
    and :func:`triangulate_shape_watertight`. Each record additionally keeps
    the originating OCCT face, triangulation handle and location so the
    watertight path can look up per-edge mesh discretizations without
    re-meshing or re-walking the shape; :func:`triangulate_shape` ignores
    those three keys.

    Raises:
        NotImplementedError: An evaluated face has an unsupported surface type.
        ValueError: An evaluated face has no triangulation or no triangles.
    """
    BRepMesh_IncrementalMesh(shape, deflection)

    # Iterate faces per-Solid to exclude orphan faces/shells that
    # are not part of any solid and would break the divergence theorem.
    face_sources: list[Any] = []
    exp_solid = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp_solid.More():
        face_sources.append(TopoDS.Solid_s(exp_solid.Current()))
        exp_solid.Next()
    if not face_sources:
        face_sources.append(shape)

    face_records: list[dict[str, Any]] = []
    for source in face_sources:
        exp = TopExp_Explorer(source, TopAbs_FACE)
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            adaptor = BRepAdaptor_Surface(face)

            result = _extract_face_geometric_params(adaptor)
            if result is None:
                msg = f"Unsupported surface type: {adaptor.GetType().name}"
                raise NotImplementedError(msg)
            surface_type, params = result

            loc = TopLoc_Location()
            poly_tri = BRep_Tool.Triangulation_s(face, loc)
            if poly_tri is None:
                raise ValueError("Face has no triangulation after OCCT meshing")

            n_nodes = poly_tri.NbNodes()
            n_tris = poly_tri.NbTriangles()
            if n_nodes == 0 or n_tris == 0:
                raise ValueError("Face triangulation contains no triangles")

            us_np = np.empty(n_nodes)
            vs_np = np.empty(n_nodes)
            for i in range(1, n_nodes + 1):
                uv = poly_tri.UVNode(i)
                us_np[i - 1] = uv.X()
                vs_np[i - 1] = uv.Y()
            # OCCT can place B-spline mesh nodes slightly outside the knot
            # domain; bspline_basis extends the end pieces, as OCCT does.

            conn = np.empty((n_tris, 3), dtype=np.int32)
            for i in range(1, n_tris + 1):
                n1, n2, n3 = poly_tri.Triangle(i).Get()
                conn[i - 1] = [n1 - 1, n2 - 1, n3 - 1]

            reverse = face.Orientation() != TopAbs_FORWARD
            if reverse:
                conn = conn[:, [0, 2, 1]]

            face_records.append(
                {
                    "surface_type": surface_type,
                    "params": params,
                    "us": us_np,
                    "vs": vs_np,
                    "n_points": n_nodes,
                    "n_triangles": int(n_tris),
                    "conn": conn,
                    "face": face,
                    "poly_tri": poly_tri,
                    "loc": loc,
                }
            )
            exp.Next()

    return face_records


def _evaluate_positions_per_face(
    face_records: list[dict[str, Any]],
) -> list[jnp.ndarray]:
    """Batch-evaluate JAX-native vertex positions for each face ("Pass 2").

    Faces are grouped by surface type (and BSpline signature) so each
    group is evaluated with a single batched ``jax.vmap``, eliminating
    per-face Python dispatch overhead. Returns one ``(n_points, 3)``
    position array per face, in ``face_records`` order.
    """
    groups: dict[tuple[Any, ...], list[int]] = {}
    for idx, rec in enumerate(face_records):
        key = _face_group_key(rec["surface_type"], rec["params"])
        groups.setdefault(key, []).append(idx)

    positions_per_face: list[jnp.ndarray] = [jnp.zeros((0, 3))] * len(face_records)

    for _key, indices in groups.items():
        group_recs = [face_records[i] for i in indices]
        max_pts = max(r["n_points"] for r in group_recs)
        n = len(group_recs)

        us_mat = np.zeros((n, max_pts))
        vs_mat = np.zeros((n, max_pts))
        for i, r in enumerate(group_recs):
            us_mat[i, : r["n_points"]] = r["us"]
            vs_mat[i, : r["n_points"]] = r["vs"]
        us_j = jnp.array(us_mat)
        vs_j = jnp.array(vs_mat)

        faces_params = [r["params"] for r in group_recs]
        surface_type = group_recs[0]["surface_type"]
        if surface_type == "bspline":
            batched_out = _dispatch_bspline_group(faces_params, us_j, vs_j)
        else:
            batched_out = _dispatch_analytical_group(
                surface_type, faces_params, us_j, vs_j
            )
        # batched_out: (N_faces, max_pts, 3). Slice per-face to actual count.
        for i, face_idx in enumerate(indices):
            n_pts = group_recs[i]["n_points"]
            positions_per_face[face_idx] = batched_out[i, :n_pts]

    return positions_per_face


def triangulate_shape(
    shape: TopoDS_Shape,
    *,
    deflection: float = _DEFAULT_DEFLECTION,
) -> tuple[jnp.ndarray, list[dict[str, Any]]]:
    """Triangulate all faces of a shape with JAX-native vertices.

    Uses OCCT BRepMesh for watertight mesh topology, then re-evaluates
    each vertex position using JAX-native parametric surface functions.
    Faces are grouped by surface type (and BSpline signature) so each
    group is evaluated with a single batched ``jax.vmap``, eliminating
    per-face Python dispatch overhead.

    Each triangle corner is evaluated independently from its own face's
    parameters: a vertex shared by two faces gets two numerically close
    but generally distinct copies (see :func:`triangulate_shape_watertight`
    for a mode that welds them to a single, shared vertex). This is the
    original, unchanged contract depended on by existing callers.

    Args:
        shape: An OCCT topological shape.
        deflection: Mesh deflection tolerance for OCCT tessellation.

    Returns:
        Tuple of ``(triangles, params_list)`` where triangles has
        shape ``(n_total, 3, 3)`` and params_list contains one
        parameter dict per face, in traversal order.

    Raises:
        NotImplementedError: An evaluated face has an unsupported surface type.
        ValueError: An evaluated face has no triangulation or no triangles.

    If solids are present, only their faces are evaluated. Otherwise all
    faces are evaluated. A shape with no faces returns empty arrays and metadata.
    """
    face_records = _extract_face_records(shape, deflection)
    if not face_records:
        return jnp.zeros((0, 3, 3)), []

    positions_per_face = _evaluate_positions_per_face(face_records)

    # Remap per-face connectivity to global indices, single indexed fetch.
    offsets = np.cumsum([0] + [r["n_points"] for r in face_records], dtype=np.int64)

    global_conn = np.concatenate(
        [face_records[i]["conn"] + offsets[i] for i in range(len(face_records))],
        axis=0,
    )
    all_positions = jnp.concatenate(positions_per_face, axis=0)
    triangles: jnp.ndarray = all_positions[global_conn]

    # params_list preserves traversal order, with surface_type and the
    # per-face triangle count included so downstream tooling can both
    # distinguish batched groupings and slice ``triangles`` per face
    # for face-level metrics (e.g. ``surface_area_per_face``).
    all_params = [
        {
            "surface_type": rec["surface_type"],
            "n_triangles": rec["n_triangles"],
            **rec["params"],
        }
        for rec in face_records
    ]

    return triangles, all_params


class _UnionFind:
    """Union-find over the flattened (face, local node) index space."""

    __slots__ = ("parent",)

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _polygon_on_edge(rec: dict[str, Any], occ_edge: Any) -> Any:
    """``Poly_PolygonOnTriangulation`` for one face's use of one edge.

    OCCT keys a seam edge's two boundary occurrences by orientation, not
    just by the underlying edge: passing the oriented edge object as it
    appears in the face's own wire (not a canonical, unoriented copy)
    returns the occurrence-specific polygon, including for the two
    occurrences of a periodic face's self-seam.
    """
    return BRep_Tool.PolygonOnTriangulation_s(occ_edge, rec["poly_tri"], rec["loc"])


def _weld_shared_vertices(
    shape: TopoDS_Shape, face_records: list[dict[str, Any]]
) -> tuple[np.ndarray, int]:
    """Union-find over (face, local node) pairs that share an OCCT edge.

    Identifies vertices shared between faces using OCCT's own per-edge mesh
    discretization (``Poly_PolygonOnTriangulation``), which BRepMesh shares
    bit-identically across every face bordering an edge, matched by the
    edge's own curve parameter -- not by a vertex distance tolerance.

    Each face is walked with its own ``TopExp_Explorer`` so a periodic
    face's seam edge naturally yields two distinct, oriented occurrences
    of the same canonical edge (one per side of the seam), handled by the
    same code path as an edge shared by two different faces. A degenerate
    edge (a pole or an apex) collapses every node any of its occurrences
    references into one vertex; degenerate edges that share a topological
    vertex (e.g. a pole touched by several patches) are unioned too.

    Returns:
        ``(group_of, n_vertices)`` where ``group_of[i]`` is the welded
        global vertex index for flattened (face, local node) index ``i``
        (offsets given by cumulative ``n_points`` over ``face_records``),
        and ``n_vertices`` is the number of distinct welded vertices.

    Raises:
        ValueError: An edge's shared discretization cannot be identified
            from OCCT topology: a free boundary or non-manifold edge (not
            exactly two occurrences), a missing per-face discretization,
            or two occurrences whose node count or curve parameters
            disagree beyond floating-point roundoff.
    """
    offsets = np.cumsum([0] + [r["n_points"] for r in face_records], dtype=np.int64)
    n_global = int(offsets[-1])
    uf = _UnionFind(n_global)

    def gid(face_idx: int, local_idx: int) -> int:
        return int(offsets[face_idx]) + local_idx

    edges_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_EDGE, edges_map)

    occurrences: dict[int, list[tuple[int, Any]]] = {}
    for face_idx, rec in enumerate(face_records):
        exp = TopExp_Explorer(rec["face"], TopAbs_EDGE)
        while exp.More():
            occ_edge = TopoDS.Edge_s(exp.Current())
            edge_id = edges_map.FindIndex(occ_edge)
            occurrences.setdefault(edge_id, []).append((face_idx, occ_edge))
            exp.Next()

    vertex_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_VERTEX, vertex_map)
    degenerate_representative_by_vertex: dict[int, list[int]] = {}

    for edge_id, occs in occurrences.items():
        edge = TopoDS.Edge_s(edges_map.FindKey(edge_id))

        if BRep_Tool.Degenerated_s(edge):
            nodes: list[int] = []
            for face_idx, occ_edge in occs:
                poly = _polygon_on_edge(face_records[face_idx], occ_edge)
                if poly is None:
                    continue
                nodes.extend(
                    gid(face_idx, poly.Node(k) - 1)
                    for k in range(1, poly.NbNodes() + 1)
                )
            for a, b in pairwise(nodes):
                uf.union(a, b)
            if nodes:
                vexp = TopExp_Explorer(edge, TopAbs_VERTEX)
                vertex_id = vertex_map.FindIndex(TopoDS.Vertex_s(vexp.Current()))
                degenerate_representative_by_vertex.setdefault(vertex_id, []).append(
                    nodes[0]
                )
            continue

        if len(occs) != 2:
            msg = (
                f"Edge shared by {len(occs)} face occurrences (expected exactly 2 "
                "for a closed manifold solid); cannot build a watertight mesh"
            )
            raise ValueError(msg)

        (face_a, edge_a), (face_b, edge_b) = occs
        poly_a = _polygon_on_edge(face_records[face_a], edge_a)
        poly_b = _polygon_on_edge(face_records[face_b], edge_b)
        if poly_a is None or poly_b is None:
            raise ValueError(
                "Shared edge is missing a per-face mesh discretization "
                "(Poly_PolygonOnTriangulation)"
            )
        na, nb = poly_a.NbNodes(), poly_b.NbNodes()
        if na != nb:
            raise ValueError(
                f"Shared edge discretizations disagree in node count ({na} != {nb})"
            )
        if not (poly_a.HasParameters() and poly_b.HasParameters()):
            raise ValueError(
                "Shared edge discretization has no stored curve parameters; "
                "cannot match nodes between its two faces from topology alone"
            )
        params_a = np.array([poly_a.Parameter(k) for k in range(1, na + 1)])
        params_b = np.array([poly_b.Parameter(k) for k in range(1, nb + 1)])
        order_a = np.argsort(params_a, kind="stable")
        order_b = np.argsort(params_b, kind="stable")
        # The edge's own curve parameter is intrinsic (shared by both faces'
        # discretizations of the same curve), so both sides should agree up
        # to floating-point roundoff -- not an adjustable weld tolerance.
        scale = max(1.0, float(np.ptp(params_a)))
        if not np.allclose(
            params_a[order_a], params_b[order_b], rtol=0.0, atol=1e-9 * scale
        ):
            raise ValueError(
                "Shared edge discretizations disagree in curve parameter "
                "beyond floating-point roundoff"
            )
        for ia, ib in zip(order_a, order_b, strict=True):
            uf.union(
                gid(face_a, poly_a.Node(int(ia) + 1) - 1),
                gid(face_b, poly_b.Node(int(ib) + 1) - 1),
            )

    # Cross-union degenerate edges that share a topological vertex (e.g. a
    # pole touched by several faces, each with its own degenerate edge).
    for representatives in degenerate_representative_by_vertex.values():
        for a, b in pairwise(representatives):
            uf.union(a, b)

    if n_global == 0:
        return np.zeros(0, dtype=np.int64), 0
    roots = np.array([uf.find(i) for i in range(n_global)])
    _, group_of = np.unique(roots, return_inverse=True)
    n_vertices = int(group_of.max()) + 1
    return group_of.astype(np.int64), n_vertices


def _pair_welded_vertices(
    face_records: list[dict[str, Any]], group_of: np.ndarray, n_vertices: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Group flattened (face, local node) indices by (welded vertex, face).

    :func:`evaluate_watertight_mesh` averages each face's own nodes within
    one of these pairs first, then averages those per-face means across a
    vertex's distinct incident faces. Without this intermediate grouping, a
    face that touches a shared vertex through more than one of its own
    nodes -- for example both sides of a periodic face's own seam, right
    where that seam meets a neighboring face -- would be counted once per
    node instead of once per face, silently biasing the average toward
    whichever face happens to supply more nodes there.

    Returns:
        ``(pair_of, pair_vertex, n_pairs)`` where ``pair_of[i]`` is the
        pair index for flattened index ``i``, and ``pair_vertex[p]`` is
        the welded vertex index that pair ``p`` belongs to.
    """
    n_faces = len(face_records)
    if n_faces == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 0
    face_of = np.repeat(np.arange(n_faces), [r["n_points"] for r in face_records])
    keys = group_of.astype(np.int64) * n_faces + face_of
    unique_keys, pair_of = np.unique(keys, return_inverse=True)
    pair_vertex = (unique_keys // n_faces).astype(np.int64)
    return pair_of.astype(np.int64), pair_vertex, len(unique_keys)


def _check_manifold_closed(faces: np.ndarray) -> None:
    """Verify every undirected edge is shared by exactly two consistently
    oriented triangles.

    Operates purely on (already-welded) vertex indices, so it checks the
    union-find connectivity result itself rather than vertex positions.
    This alone does not rule out two otherwise-disjoint fans that merely
    touch at one shared vertex with no shared edge; see
    :func:`_check_vertex_links_manifold` for that case.
    """
    if faces.shape[0] == 0:
        return
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("Watertight mesh has a degenerate (zero-length) edge")
    _, labels, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if np.any(counts != 2):
        raise ValueError(
            "Watertight mesh is not closed: some edge is not shared by "
            "exactly two triangles"
        )
    orientation = np.bincount(labels, weights=np.sign(edges[:, 1] - edges[:, 0]))
    if np.any(orientation != 0):
        raise ValueError(
            "Watertight mesh is not consistently oriented across a shared edge"
        )


def _check_vertex_links_manifold(faces: np.ndarray) -> None:
    """Verify every vertex's incident triangles form a single closed fan.

    The edge-pairing check in :func:`_check_manifold_closed` accepts two
    otherwise-disjoint fans (e.g. two tetrahedra) that are pinched together
    at one shared vertex with no shared edge between them -- every edge is
    still paired exactly twice, just not across the two fans. For every
    vertex, its "link" (the opposite edge of each incident triangle) must
    instead form exactly one simple cycle over that vertex's distinct
    neighbors: every neighbor has degree exactly two in that link graph,
    and the link graph is connected (one cycle, not several disjoint ones).

    Raises:
        ValueError: Some vertex's incident triangles do not form a single
            closed fan.
    """
    if faces.shape[0] == 0:
        return
    corner = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    far_a = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    far_b = np.concatenate([faces[:, 2], faces[:, 0], faces[:, 1]])

    order = np.argsort(corner, kind="stable")
    corner, far_a, far_b = corner[order], far_a[order], far_b[order]
    _, start, counts = np.unique(corner, return_index=True, return_counts=True)

    msg = (
        "Watertight mesh has a non-manifold vertex: its incident triangles "
        "do not form a single closed fan"
    )
    for start_idx, count in zip(start, counts, strict=True):
        a = far_a[start_idx : start_idx + count]
        b = far_b[start_idx : start_idx + count]
        neighbors, local = np.unique(np.concatenate([a, b]), return_inverse=True)
        n = len(neighbors)
        local_a, local_b = local[:count], local[count:]
        degree = np.bincount(local_a, minlength=n) + np.bincount(local_b, minlength=n)
        if np.any(degree != 2):
            raise ValueError(msg)

        parent = list(range(n))

        def find(x: int, parent: list[int] = parent) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for ia, ib in zip(local_a, local_b, strict=True):
            ra, rb = find(int(ia)), find(int(ib))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        if len({find(i) for i in range(n)}) != 1:
            raise ValueError(msg)


class WatertightMesh(eqx.Module):
    """Welded, consistently oriented mesh from :func:`evaluate_watertight_mesh`.

    A registered JAX pytree (an :class:`equinox.Module`) of arrays only, so
    it can be returned from -- not just passed through -- ``jax.jit`` and
    ``jax.vmap``. Host metadata (``surface_type``, per-face triangle counts)
    lives on :class:`WatertightTopology` instead, since it is not itself
    array data; correlate it with ``faces`` via
    ``WatertightTopology.n_triangles_per_face``.

    ``vertices[faces]`` reproduces the ``(n, 3, 3)`` triangle-corner layout
    :func:`triangulate_shape` returns, so it can be passed directly to
    :func:`divergence_volume`, :func:`mesh_surface_area`, and the other
    mesh metrics in this module.
    """

    vertices: Float[Array, "n_vertices 3"]
    """Welded vertex positions. Each row is the mean, over every incident
    face, of that face's own mean of its (one or more) nodes at this vertex
    -- see :func:`evaluate_watertight_mesh` -- so ``jax.grad`` distributes
    symmetrically across incident faces, weighted by face, not by node."""

    faces: Int[Array, "n_triangles 3"]
    """Triangle connectivity into ``vertices``."""

    max_gap: Float[Array, ""]
    """An upper bound on the largest disagreement between any two of a
    shared vertex's incident *nodes* (not the per-face means used for
    ``vertices``): twice the largest distance from any individual node's
    own independent evaluation to that vertex's final welded position. By
    the triangle inequality this bounds the true largest pairwise gap
    between any two incident nodes' evaluations, including a disagreement
    between two nodes on the *same* face (e.g. both sides of a periodic
    face's own seam), which the face-weighted mean used for ``vertices``
    does not by itself expose. A large value indicates surface
    reconstructions that disagree at a shared boundary; see the
    ``tolerance`` argument of :func:`triangulate_shape_watertight`."""


# Identity equality and hashing: the fields hold lists and arrays, and a
# topology is built once per shape and passed to jit as a static argument.
@dataclass(frozen=True, eq=False)
class WatertightTopology:
    """Frozen host-side topology feeding :func:`evaluate_watertight_mesh`.

    Produced once per shape by :func:`extract_watertight_topology` (OCCT
    meshing, edge-topology welding, and manifold validation -- all outside
    ``jit``/``vmap``/differentiation), then reused for repeated
    differentiable re-evaluation with different per-face surface
    parameters, analogous to :func:`extract_mesh_topology` and
    :func:`evaluate_mesh` for the non-welded mesh. Not itself a registered
    pytree: it is meant to be closed over or passed as a static argument,
    never returned from or traced by ``jit``/``vmap``.
    """

    face_records: list[dict[str, Any]]
    """Per-face host data: ``surface_type``, ``us``, ``vs``, ``n_points``,
    and -- for a ``"bspline"`` face -- the B-spline degrees ``deg_u``,
    ``deg_v``. Degrees are shape-determining metadata (they select which
    batched evaluator to dispatch to and are used as dict/cache keys), so
    they live here, not in ``params_list``, and stay concrete Python ints
    even when ``evaluate_watertight_mesh`` is called under ``jax.jit``."""

    pair_of: np.ndarray
    """Flattened (face, local node) index -> (welded vertex, face) pair index."""

    pair_vertex: np.ndarray
    """(welded vertex, face) pair index -> welded vertex index."""

    node_vertex: np.ndarray
    """Flattened (face, local node) index -> welded vertex index directly
    (``pair_vertex[pair_of]``), used for the raw per-node gap in
    ``WatertightMesh.max_gap``."""

    n_vertices: int
    n_pairs: int

    faces: jnp.ndarray
    """Triangle connectivity into the welded vertices, ``(n_triangles, 3)``,
    ``int32``; already filtered to drop zero-area triangles (a full-revolution
    pole or apex can leave OCCT's own mesh with one spanning two now-welded
    copies of that point, which contributed exactly zero area and volume
    before welding)."""

    params_list: list[dict[str, Any]]
    """Default per-face surface parameters extracted from OCCT, in
    ``face_records`` order, with B-spline degrees excluded (see
    ``face_records``). Pass to :func:`evaluate_watertight_mesh`, or a
    modified copy of it to differentiate with respect to specific per-face
    parameters."""

    n_triangles_per_face: list[int]
    """Post-filtering triangle count for each face, in ``face_records``
    order. Can differ from :func:`triangulate_shape`'s count for the same
    face because of the zero-area triangles ``faces`` drops."""


def extract_watertight_topology(
    shape: TopoDS_Shape, *, deflection: float = _DEFAULT_DEFLECTION
) -> WatertightTopology:
    """Host-side extraction and validation for a welded watertight mesh.

    Meshes ``shape``, identifies shared vertices from OCCT edge topology
    (see :func:`_weld_shared_vertices`), and validates that the welded
    result is a closed, manifold, consistently oriented mesh whose every
    vertex's incident triangles form a single fan (see
    :func:`_check_manifold_closed` and :func:`_check_vertex_links_manifold`).
    All of this runs once, on the host, outside any JAX transformation;
    :func:`evaluate_watertight_mesh` does the repeated, differentiable part.

    Args:
        shape: An OCCT topological shape.
        deflection: Mesh deflection tolerance for OCCT tessellation.

    Returns:
        A :class:`WatertightTopology`.

    Raises:
        NotImplementedError: An evaluated face has an unsupported surface type.
        ValueError: A face has no triangulation, or the welded result is
            not a closed, manifold, consistently oriented mesh.

    Examples:
        >>> topology = extract_watertight_topology(shape)
        >>> mesh = evaluate_watertight_mesh(topology, topology.params_list)
        >>> vol = divergence_volume(mesh.vertices[mesh.faces])
    """
    face_records = _extract_face_records(shape, deflection)
    if not face_records:
        return WatertightTopology(
            face_records=[],
            pair_of=np.zeros(0, dtype=np.int64),
            pair_vertex=np.zeros(0, dtype=np.int64),
            node_vertex=np.zeros(0, dtype=np.int64),
            n_vertices=0,
            n_pairs=0,
            faces=jnp.zeros((0, 3), dtype=jnp.int32),
            params_list=[],
            n_triangles_per_face=[],
        )

    group_of, n_vertices = _weld_shared_vertices(shape, face_records)
    pair_of, pair_vertex, n_pairs = _pair_welded_vertices(
        face_records, group_of, n_vertices
    )
    # Equal to pair_vertex[pair_of]: the welded vertex for every flattened
    # (face, local node) index directly, independent of pairing by face.
    node_vertex = group_of

    offsets = np.cumsum([0] + [r["n_points"] for r in face_records], dtype=np.int64)
    per_face_faces: list[np.ndarray] = []
    n_triangles_per_face: list[int] = []
    for i, rec in enumerate(face_records):
        local_group = group_of[offsets[i] : offsets[i] + rec["n_points"]]
        face_tris = local_group[rec["conn"]]
        # A full-revolution pole/apex can leave OCCT's own mesh with a
        # zero-area triangle spanning both of the degenerate edge's node
        # copies (e.g. a seam-closing triangle at a sphere pole); welding
        # those copies to one vertex turns it into a repeated-corner row.
        # It already contributed exactly zero area/volume before welding,
        # so dropping it changes nothing but the bookkeeping.
        degenerate = (
            (face_tris[:, 0] == face_tris[:, 1])
            | (face_tris[:, 1] == face_tris[:, 2])
            | (face_tris[:, 0] == face_tris[:, 2])
        )
        if np.any(degenerate):
            face_tris = face_tris[~degenerate]
        per_face_faces.append(face_tris)
        n_triangles_per_face.append(int(face_tris.shape[0]))
    global_faces = np.concatenate(per_face_faces, axis=0)
    _check_manifold_closed(global_faces)
    _check_vertex_links_manifold(global_faces)

    # Degrees are shape-determining dispatch/cache keys (_face_group_key,
    # _get_bspline_batched_vmap), not differentiable data: keep them as
    # concrete host ints in face_records, never inside params_list, so
    # evaluate_watertight_mesh stays safe to call with params_list traced
    # under jax.jit (a traced int cannot be hashed as a dict key).
    light_face_records = [
        {
            "surface_type": rec["surface_type"],
            "us": rec["us"],
            "vs": rec["vs"],
            "n_points": rec["n_points"],
            **(
                {"deg_u": rec["params"]["deg_u"], "deg_v": rec["params"]["deg_v"]}
                if rec["surface_type"] == "bspline"
                else {}
            ),
        }
        for rec in face_records
    ]
    params_list = [
        {k: v for k, v in rec["params"].items() if k not in ("deg_u", "deg_v")}
        for rec in face_records
    ]

    return WatertightTopology(
        face_records=light_face_records,
        pair_of=pair_of,
        pair_vertex=pair_vertex,
        node_vertex=node_vertex,
        n_vertices=n_vertices,
        n_pairs=n_pairs,
        faces=jnp.asarray(global_faces, dtype=jnp.int32),
        params_list=params_list,
        n_triangles_per_face=n_triangles_per_face,
    )


def evaluate_watertight_mesh(
    topology: WatertightTopology, params_list: list[dict[str, Any]]
) -> WatertightMesh:
    """Re-evaluate a welded mesh's vertices from explicit per-face parameters.

    For each face, evaluates its own surface function at its stored UV
    coordinates using ``params_list`` (JAX-native, so ``jax.grad`` flows
    from the result through these parameters). A shared vertex's position
    is a two-level mean: first the mean of each incident face's own node
    evaluations at that vertex (collapsing, for example, both sides of a
    periodic face's own seam to one value per face), then the mean of
    those per-face means across the vertex's distinct incident faces. The
    second level is what keeps the result -- and its gradient -- weighted
    equally per incident *face* regardless of how many of that face's own
    nodes happen to coincide there. ``WatertightMesh.max_gap`` is computed
    separately, directly from the raw per-node evaluations (before either
    level of averaging), so it still exposes a disagreement the position
    averaging would otherwise hide, including between two nodes of the
    *same* face.

    ``params_list`` entries hold only differentiable arrays (for a
    B-spline face, no ``deg_u``/``deg_v``: see ``WatertightTopology.
    face_records``), so this is a pure function of ``params_list`` with
    ``topology`` held static, and it supports ``jax.jit``, ``jax.vmap``,
    and differentiation the same way :func:`evaluate_mesh` does for the
    non-welded mesh -- including returning its ``WatertightMesh`` result
    directly from a jitted or vmapped function.

    Args:
        topology: Frozen topology from :func:`extract_watertight_topology`.
        params_list: One parameter dict per face, in ``topology.face_records``
            order; typically ``topology.params_list`` or a modified copy
            of it (e.g. with one face's control points perturbed).

    Returns:
        A :class:`WatertightMesh`.

    Examples:
        >>> topology = extract_watertight_topology(shape)
        >>> def volume_of(radius, face_index):
        ...     params = [dict(p) for p in topology.params_list]
        ...     params[face_index] = {**params[face_index], "radius": radius}
        ...     mesh = evaluate_watertight_mesh(topology, params)
        ...     return divergence_volume(mesh.vertices[mesh.faces])
        >>> grad = jax.grad(volume_of)(jnp.array(5.0), 0)
    """
    if topology.n_vertices == 0:
        return WatertightMesh(
            vertices=jnp.zeros((0, 3)),
            faces=jnp.zeros((0, 3), dtype=jnp.int32),
            max_gap=jnp.zeros(()),
        )

    # Degrees are reattached from static host metadata, never read from the
    # (possibly traced) params_list argument: see WatertightTopology.face_records.
    face_records = [
        {
            **rec,
            "params": (
                {**params, "deg_u": rec["deg_u"], "deg_v": rec["deg_v"]}
                if rec["surface_type"] == "bspline"
                else params
            ),
        }
        for rec, params in zip(topology.face_records, params_list, strict=True)
    ]
    positions_per_face = _evaluate_positions_per_face(face_records)
    all_positions = jnp.concatenate(positions_per_face, axis=0)

    pair_of = jnp.asarray(topology.pair_of)
    pair_vertex = jnp.asarray(topology.pair_vertex)
    pair_counts = np.bincount(topology.pair_of, minlength=topology.n_pairs).astype(
        all_positions.dtype
    )
    vertex_counts = np.bincount(
        topology.pair_vertex, minlength=topology.n_vertices
    ).astype(all_positions.dtype)

    # First level: each face's own mean at this vertex (collapses a face's
    # own seam-duplicated nodes here to one value per face).
    pair_sum = (
        jnp.zeros((topology.n_pairs, 3), dtype=all_positions.dtype)
        .at[pair_of]
        .add(all_positions)
    )
    pair_mean = pair_sum / pair_counts[:, None]

    # Second level: mean of those per-face means across distinct incident
    # faces, weighted equally per face regardless of node multiplicity.
    vertex_sum = (
        jnp.zeros((topology.n_vertices, 3), dtype=all_positions.dtype)
        .at[pair_vertex]
        .add(pair_mean)
    )
    vertices = vertex_sum / vertex_counts[:, None]

    # The gap diagnostic looks at every raw node independently (not the
    # per-face means above), so it still catches two of a single face's
    # own nodes disagreeing at a vertex -- the per-face mean used for
    # ``vertices`` would otherwise silently absorb exactly that case.
    node_vertex = jnp.asarray(topology.node_vertex)
    deviation = jnp.linalg.norm(all_positions - vertices[node_vertex], axis=-1)
    max_gap = 2.0 * jnp.max(deviation) if deviation.shape[0] else jnp.zeros(())

    return WatertightMesh(
        vertices=vertices,
        faces=topology.faces,
        max_gap=max_gap,
    )


def triangulate_shape_watertight(
    shape: TopoDS_Shape,
    *,
    deflection: float = _DEFAULT_DEFLECTION,
    tolerance: float | None = None,
) -> WatertightMesh:
    """Triangulate a shape into a single welded, watertight mesh.

    Unlike :func:`triangulate_shape`, which gives each triangle corner an
    independent evaluation from its own face (so a shared edge gets two
    numerically close but distinct copies), this welds shared vertices
    identified from OCCT's own per-edge mesh discretization (topology and
    the edge's own curve parameter, not a vertex distance tolerance) and
    averages every incident face's own evaluation there, producing one
    vertex per physical point with gradients that flow symmetrically to
    every incident face's parameters. Degenerate edges (poles, apices) and
    a periodic face's seam are handled explicitly (see
    :func:`_weld_shared_vertices`).

    A convenience composition of :func:`extract_watertight_topology` (host
    extraction and validation) and :func:`evaluate_watertight_mesh`
    (differentiable re-evaluation). Call those directly to re-evaluate the
    same topology under ``jax.jit``, ``jax.vmap``, or ``jax.grad`` against
    specific per-face parameters without repeating the OCCT work.

    Args:
        shape: An OCCT topological shape.
        deflection: Mesh deflection tolerance for OCCT tessellation.
        tolerance: If given, raise when ``WatertightMesh.max_gap`` (an
            upper bound on the largest gap between any two nodes'
            independent evaluations at a shared vertex) exceeds it. Pass the shape's
            own CAD tolerance (e.g. :func:`brepax.experimental.applications.
            _hole_geometry.shape_tolerance`) to reject faces whose
            reconstructions disagree by more than the model already
            tolerates. ``None`` (default) never raises on the gap; the
            value is still returned in ``max_gap`` for the caller to
            inspect.

    Returns:
        A :class:`WatertightMesh`.

    Raises:
        NotImplementedError: An evaluated face has an unsupported surface type.
        ValueError: A face has no triangulation; a shared edge cannot be
            identified from OCCT topology (see :func:`_weld_shared_vertices`);
            the welded result is not a closed, manifold, consistently
            oriented mesh; or ``tolerance`` is given and exceeded.

    Examples:
        >>> shape = read_step("model.step")
        >>> mesh = triangulate_shape_watertight(shape, tolerance=1e-4)
        >>> vol = divergence_volume(mesh.vertices[mesh.faces])
        >>> grad = jax.grad(
        ...     lambda v: divergence_volume(v[mesh.faces])
        ... )(mesh.vertices)
    """
    topology = extract_watertight_topology(shape, deflection=deflection)
    mesh = evaluate_watertight_mesh(topology, topology.params_list)
    if tolerance is not None and float(mesh.max_gap) > tolerance:
        raise ValueError(
            f"Upper bound on the largest shared-vertex evaluation gap "
            f"{float(mesh.max_gap):.6g} exceeds tolerance {tolerance:.6g}"
        )
    return mesh


def divergence_volume(triangles: jnp.ndarray) -> jnp.ndarray:
    """Compute volume of a closed triangle mesh via the divergence theorem.

    Uses the signed-tetrahedra formula in a local frame near the mesh:
    each triangle and the local origin form a tetrahedron whose signed
    volume is ``v0 . (v1 x v2) / 6``.
    The sum over all triangles gives the enclosed volume for a
    watertight mesh with outward-facing normals.

    Local coordinates reduce cancellation for solids far from the world
    origin. Units are length cubed. Empty meshes return zero.

    ``jax.grad`` differentiates the frozen mesh. Geometric shape derivatives
    require shared boundaries to move consistently and preserve closure;
    independently moving duplicated triangle corners can open the mesh.

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
    if triangles.shape[0] == 0:
        return jnp.zeros((), dtype=triangles.dtype)

    local = triangles - jnp.mean(triangles, axis=(0, 1))
    v0, v1, v2 = local[:, 0], local[:, 1], local[:, 2]
    return jnp.sum(v0 * jnp.cross(v1, v2)) / 6.0


def mesh_surface_area(triangles: jnp.ndarray) -> jnp.ndarray:
    """Compute surface area of a triangle mesh.

    Sums the area of each triangle: ``(1/2) ||cross(e1, e2)||``.
    Units are length squared. No absolute area floor is applied. Collapsed
    or collinear triangles have zero area with a chosen zero subgradient
    at the nondifferentiable point. Empty meshes return zero.

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
    squared_norm = jnp.sum(cross**2, axis=-1)
    nonzero = squared_norm > 0.0
    # Keep sqrt's derivative finite even in the unselected zero-area branch.
    safe_norm = jnp.sqrt(jnp.where(nonzero, squared_norm, 1.0))
    return jnp.sum(jnp.where(nonzero, safe_norm, 0.0)) / 2.0


def mesh_center_of_mass(triangles: jnp.ndarray) -> jnp.ndarray:
    """Compute center of mass of a closed triangle mesh.

    Integrates signed tetrahedron first moments in a local frame, then
    divides by signed volume. This avoids cancelling world-origin moments
    and introducing a dimensionful epsilon into the denominator.

    Assumes a watertight mesh with nonzero volume. Units are length.
    Empty meshes return zero. For nonempty zero-volume inputs the center
    is undefined; nearly degenerate solids are ill-conditioned. Shape
    derivatives require closure-preserving shared-boundary motion.

    Args:
        triangles: Triangle vertices, shape ``(n, 3, 3)``.

    Returns:
        Center of mass, shape ``(3,)``.

    Examples:
        >>> tris, _ = triangulate_shape(shape)
        >>> com = mesh_center_of_mass(tris)
    """
    if triangles.shape[0] == 0:
        return jnp.zeros((3,), dtype=triangles.dtype)

    reference = jnp.mean(triangles, axis=(0, 1))
    local = triangles - reference
    v0, v1, v2 = local[:, 0], local[:, 1], local[:, 2]
    signed_volumes = jnp.sum(v0 * jnp.cross(v1, v2), axis=-1) / 6.0
    moments = jnp.sum(signed_volumes[:, None] * jnp.sum(local, axis=1), axis=0) / 4.0
    return reference + moments / jnp.sum(signed_volumes)


def mesh_inertia_tensor(triangles: jnp.ndarray) -> jnp.ndarray:
    """Compute the inertia tensor of a closed triangle mesh about its center of mass.

    Integrates signed tetrahedron moments (Tonon 2005), then applies
    the parallel axis theorem in a local frame near the center of mass.
    See https://doi.org/10.3844/jmssp.2005.8.11 for tetrahedron moments.

    Assumes a watertight, outward-oriented mesh with nonzero volume and
    uniform density ``rho = 1``. Empty meshes return zero. Length units
    ``L`` give inertia units ``L**5``; uniform scaling by ``s`` multiplies
    the tensor by ``s**5``. Nearly degenerate solids are ill-conditioned.

    Args:
        triangles: Triangle vertices, shape ``(n, 3, 3)``.

    Returns:
        Symmetric 3x3 inertia tensor about center of mass.

    Examples:
        >>> tris, _ = triangulate_shape(shape)
        >>> I = mesh_inertia_tensor(tris)
    """
    if triangles.shape[0] == 0:
        return jnp.zeros((3, 3), dtype=triangles.dtype)

    # Local coordinates avoid cancelling large world-origin moments.
    local = triangles - jnp.mean(triangles, axis=(0, 1))
    a, b, c = local[:, 0], local[:, 1], local[:, 2]
    signed_volumes = jnp.sum(a * jnp.cross(b, c), axis=-1) / 6.0
    volume = jnp.sum(signed_volumes)
    vertex_sums = jnp.sum(local, axis=1)
    centroid = jnp.einsum("n,ni->i", signed_volumes, vertex_sums) / (4.0 * volume)

    # For vertices (0,a,b,c), integral(rr^T) = V * (sum(vv^T) + ss^T) / 20.
    second_moment = (
        jnp.einsum("n,nvi,nvj->ij", signed_volumes, local, local)
        + jnp.einsum("n,ni,nj->ij", signed_volumes, vertex_sums, vertex_sums)
    ) / 20.0
    central_moment = second_moment - volume * jnp.outer(centroid, centroid)
    return (
        jnp.trace(central_moment) * jnp.eye(3, dtype=central_moment.dtype)
        - central_moment
    )


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
        result = _extract_face_geometric_params(adaptor)
        assert result is not None  # BSpline is supported
        _stype, bspline_params = result
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

    Raises:
        NotImplementedError: An evaluated face has an unsupported surface type.
        ValueError: An evaluated face has no triangulation or no triangles.

    Face selection and empty-shape behavior follow :func:`triangulate_shape`.

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
                msg = f"Unsupported surface type: {adaptor.GetType().name}"
                raise NotImplementedError(msg)

            loc = TopLoc_Location()
            poly_tri = BRep_Tool.Triangulation_s(face, loc)
            if poly_tri is None:
                raise ValueError("Face has no triangulation after OCCT meshing")

            n_nodes = poly_tri.NbNodes()
            n_tris = poly_tri.NbTriangles()
            if n_nodes == 0 or n_tris == 0:
                raise ValueError("Face triangulation contains no triangles")

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
    "WatertightMesh",
    "WatertightTopology",
    "divergence_volume",
    "evaluate_mesh",
    "evaluate_watertight_mesh",
    "extract_mesh_topology",
    "extract_watertight_topology",
    "mesh_center_of_mass",
    "mesh_inertia_tensor",
    "mesh_surface_area",
    "per_face_geometric_params",
    "triangulate_shape",
    "triangulate_shape_watertight",
]
