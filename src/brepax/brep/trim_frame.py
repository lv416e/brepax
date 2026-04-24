"""Extract Marschner-composition inputs from an OCCT face and compose.

For each analytical primitive type, this module produces the trim-aware
SDF inputs (frame, polygon in UV, polyline in 3D, validity mask) that
``trim_sdf.trim_aware_sdf`` consumes, on top of the primitive-level
reconstruction already done in ``convert.py``.  It also exposes the
per-surface composition wrappers that combine the extracted frame with
the JAX composition primitive, so callers can go from an OCCT face to a
trim-aware signed distance without re-implementing the frame-to-SDF
plumbing on every call.

Plane is supported first; sphere, cylinder, cone, torus, and BSpline
are follow-ups that reuse the same ``NamedTuple``-style return
convention and the same outer-wire sampling path so caller code stays
uniform.

Extraction is Python-side (OCCT traversal is inherently so) and runs
once per face.  The composition function takes the extracted frame
plus a query point and is pure JAX, so a typical usage pattern is:
extract once, then JIT a grid/batch evaluation of the composition.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from brepax._occt.backend import (
    BRepAdaptor_Curve2d,
    BRepAdaptor_Surface,
    BRepTools,
    BRepTools_WireExplorer,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    TopAbs_FORWARD,
)
from brepax._occt.types import TopoDS_Face
from brepax.brep.trim_sdf import trim_aware_sdf

_SAMPLES_PER_EDGE = 8


def _sample_outer_wire_uv(
    face: TopoDS_Face, max_vertices: int
) -> tuple[jnp.ndarray, jnp.ndarray] | None:
    """Sample a face's outer wire into a padded ``(u, v)`` polygon.

    Returns ``(polygon_uv, mask)`` with ``polygon_uv`` shape
    ``(max_vertices, 2)`` and ``mask`` shape ``(max_vertices,)``.
    Returns ``None`` when the face has no outer wire or produces fewer
    than 3 samples (degenerate).  Raises ``ValueError`` when the sample
    count exceeds ``max_vertices``.

    The same 8-samples-per-edge convention is used by every
    analytical primitive extractor so their polygons stay comparable.
    """
    wire = BRepTools.OuterWire_s(face)
    if wire.IsNull():
        return None

    raw_uv: list[tuple[float, float]] = []
    we = BRepTools_WireExplorer(wire, face)
    while we.More():
        edge = we.Current()
        curve2d = BRepAdaptor_Curve2d(edge, face)
        t0 = curve2d.FirstParameter()
        t1 = curve2d.LastParameter()
        is_forward = edge.Orientation() == TopAbs_FORWARD

        # frac spans [0, 1) so t1 is deliberately excluded: adjacent
        # edges of a closed wire share endpoints, and the exclusion
        # avoids duplicating the shared vertex on each seam.
        for i in range(_SAMPLES_PER_EDGE):
            frac = i / _SAMPLES_PER_EDGE
            t = t0 + (t1 - t0) * frac if is_forward else t1 - (t1 - t0) * frac
            pt = curve2d.Value(t)
            raw_uv.append((pt.X(), pt.Y()))

        we.Next()

    if len(raw_uv) < 3:
        return None

    n_valid = len(raw_uv)
    if n_valid > max_vertices:
        raise ValueError(
            f"trim polygon has {n_valid} vertices, exceeds max_vertices={max_vertices}"
        )

    polygon_np = np.zeros((max_vertices, 2), dtype=np.float64)
    mask_np = np.zeros(max_vertices, dtype=np.float64)
    polygon_np[:n_valid] = raw_uv
    mask_np[:n_valid] = 1.0
    return jnp.asarray(polygon_np), jnp.asarray(mask_np)


class PlaneTrimFrame(NamedTuple):
    """Marschner-composition inputs for a plane face.

    Caller combines these with ``query``-specific ``d_s`` (the plane's
    signed distance at the query point, e.g. ``Plane.sdf(query)``) and
    ``foot_uv`` (the query's foot-of-perpendicular in the plane's 2D
    frame, i.e. ``((foot - origin) . frame_u, (foot - origin) . frame_v)``)
    to call ``trim_aware_sdf``.

    ``polyline_3d[i] == origin + polygon_uv[i, 0] * frame_u
    + polygon_uv[i, 1] * frame_v`` holds for every valid slot ``i``
    (``mask[i] == 1.0``).
    """

    normal: Float[Array, 3]
    offset: Float[Array, ""]
    origin: Float[Array, 3]
    frame_u: Float[Array, 3]
    frame_v: Float[Array, 3]
    polygon_uv: Float[Array, "n 2"]
    polyline_3d: Float[Array, "n 3"]
    mask: Float[Array, ...]


class SphereTrimFrame(NamedTuple):
    """Marschner-composition inputs for a sphere face.

    The sphere's parametric surface is
    ``S(u, v) = center
              + radius * (cos(v) * (cos(u) * x_dir + sin(u) * y_dir)
                          + sin(v) * axis)``,
    matching OCCT's Geom_Sphere convention.  ``u`` is longitude around
    the polar ``axis`` and ``v`` is latitude (``-pi/2`` at the south
    pole, ``+pi/2`` at the north).  ``polygon_uv`` stores the trim
    loop in ``(u, v)``; full-revolution faces cover the rectangle
    ``[0, 2*pi] x [-pi/2, pi/2]`` with degenerate edges at the poles.

    ``sign_flip`` is ``+1`` for a ``TopAbs_FORWARD`` face (outward
    normal points radially away from the center, as with the outside
    of a ball) and ``-1`` for a ``TopAbs_REVERSED`` face (outward
    points radially toward the center, as with a spherical hollow).
    The composition wrapper multiplies the primitive's signed
    distance by ``sign_flip`` so phantom elimination stays correct
    for reversed faces, matching the cylinder convention.

    Examples:
        >>> from brepax.brep.trim_frame import extract_sphere_trim_frame
        >>> # tf = extract_sphere_trim_frame(sphere_face)
        >>> # assert tf.polygon_uv.shape == (64, 2)
    """

    center: Float[Array, 3]
    axis: Float[Array, 3]
    x_dir: Float[Array, 3]
    y_dir: Float[Array, 3]
    radius: Float[Array, ""]
    sign_flip: Float[Array, ""]
    polygon_uv: Float[Array, "n 2"]
    polyline_3d: Float[Array, "n 3"]
    mask: Float[Array, ...]


class CylinderTrimFrame(NamedTuple):
    """Marschner-composition inputs for a cylinder face.

    The cylinder's parametric surface is
    ``S(u, v) = origin + v * axis + radius * (cos(u) * x_dir + sin(u) * y_dir)``,
    matching OCCT's Geom_Cylinder convention.  ``polygon_uv`` stores
    the trim loop in ``(u, v)``; OCCT parameterises full-revolution
    faces as rectangles ``[0, 2*pi] x [v_min, v_max]``.  The 3D
    polyline is rebuilt from the frame for every valid slot by the
    same identity.

    ``sign_flip`` captures the face orientation: ``+1`` for a
    ``TopAbs_FORWARD`` face (outward normal points radially away from
    the axis, matching the ``Cylinder`` primitive's signed-distance
    convention) and ``-1`` for a ``TopAbs_REVERSED`` face (outward
    points *toward* the axis, as with the inside of a hole).  The
    composition wrapper multiplies the primitive's signed distance by
    ``sign_flip`` so phantom elimination stays correct for reversed
    faces.

    Examples:
        >>> from brepax.brep.trim_frame import extract_cylinder_trim_frame
        >>> # Assuming ``face`` is an OCCT plane face of a cylinder surface:
        >>> # tf = extract_cylinder_trim_frame(face)
        >>> # tf.polygon_uv.shape   # (64, 2)
        >>> # tf.polyline_3d.shape  # (64, 3)
        >>> # tf.radius             # Array(<scalar>)
    """

    origin: Float[Array, 3]
    axis: Float[Array, 3]
    x_dir: Float[Array, 3]
    y_dir: Float[Array, 3]
    radius: Float[Array, ""]
    sign_flip: Float[Array, ""]
    polygon_uv: Float[Array, "n 2"]
    polyline_3d: Float[Array, "n 3"]
    mask: Float[Array, ...]


def extract_plane_trim_frame(
    face: TopoDS_Face,
    max_vertices: int = 64,
) -> PlaneTrimFrame | None:
    """Build Marschner-composition inputs for a plane face.

    Args:
        face: OCCT face whose underlying surface must be
            ``GeomAbs_Plane``; any other surface type returns ``None``.
        max_vertices: Fixed polygon capacity after padding.  The UV
            polygon is sampled at ``_SAMPLES_PER_EDGE`` points per
            wire edge, and the result is padded to this length so the
            JAX-side consumer sees a static shape.  Raises if the
            actual vertex count exceeds this capacity; callers choose
            a capacity suited to the expected face complexity.

    Returns:
        :class:`PlaneTrimFrame` with the extracted data, or ``None``
        when the face is not a plane or has no outer wire.
    """
    adaptor = BRepAdaptor_Surface(face)
    if adaptor.GetType() != GeomAbs_Plane:
        return None

    gp_plane = adaptor.Plane()
    position = gp_plane.Position()

    loc = position.Location()
    origin = jnp.array([loc.X(), loc.Y(), loc.Z()])

    axis = position.Direction()
    normal = jnp.array([axis.X(), axis.Y(), axis.Z()])
    # OCCT faces carry an orientation independent of the underlying
    # surface normal; REVERSED means the solid's outward direction is
    # opposite the Geom_Plane's axis.  Flip so the frame's normal is
    # always the outward one, matching the convention used by the
    # ``Plane`` primitive's signed distance.
    if face.Orientation() != TopAbs_FORWARD:
        normal = -normal
    offset = jnp.dot(normal, origin)

    xd = position.XDirection()
    frame_u = jnp.array([xd.X(), xd.Y(), xd.Z()])
    yd = position.YDirection()
    frame_v = jnp.array([yd.X(), yd.Y(), yd.Z()])

    sampled = _sample_outer_wire_uv(face, max_vertices)
    if sampled is None:
        return None
    polygon_uv, mask = sampled

    polyline_3d = origin + polygon_uv @ jnp.stack([frame_u, frame_v])

    return PlaneTrimFrame(
        normal=normal,
        offset=offset,
        origin=origin,
        frame_u=frame_u,
        frame_v=frame_v,
        polygon_uv=polygon_uv,
        polyline_3d=polyline_3d,
        mask=mask,
    )


def extract_cylinder_trim_frame(
    face: TopoDS_Face,
    max_vertices: int = 64,
) -> CylinderTrimFrame | None:
    """Build Marschner-composition inputs for a cylinder face.

    Args:
        face: OCCT face whose underlying surface must be
            ``GeomAbs_Cylinder``; any other surface type returns
            ``None``.
        max_vertices: Fixed polygon capacity after padding.  Raises
            when the actual sample count exceeds this capacity.

    Returns:
        :class:`CylinderTrimFrame`, or ``None`` when the face is not a
        cylinder or has no outer wire.

    Examples:
        >>> from brepax.brep.trim_frame import extract_cylinder_trim_frame
        >>> # tf = extract_cylinder_trim_frame(cylinder_face)
        >>> # assert tf is not None
        >>> # assert tf.polygon_uv.shape == (64, 2)
        >>> # assert tf.polyline_3d.shape == (64, 3)
    """
    adaptor = BRepAdaptor_Surface(face)
    if adaptor.GetType() != GeomAbs_Cylinder:
        return None

    gp_cyl = adaptor.Cylinder()
    position = gp_cyl.Position()

    loc = position.Location()
    origin = jnp.array([loc.X(), loc.Y(), loc.Z()])

    ax = position.Direction()
    axis = jnp.array([ax.X(), ax.Y(), ax.Z()])

    xd = position.XDirection()
    x_dir = jnp.array([xd.X(), xd.Y(), xd.Z()])
    yd = position.YDirection()
    y_dir = jnp.array([yd.X(), yd.Y(), yd.Z()])

    radius = jnp.asarray(gp_cyl.Radius())

    # FORWARD cylinder faces (e.g. outside of a boss) have their
    # outward normal radially outward; REVERSED faces (e.g. inside of
    # a hole) have it pointing toward the axis.  ``sign_flip`` is
    # applied to the ``Cylinder`` primitive's radial signed distance
    # in the composition wrapper.
    sign_flip = jnp.asarray(1.0 if face.Orientation() == TopAbs_FORWARD else -1.0)

    sampled = _sample_outer_wire_uv(face, max_vertices)
    if sampled is None:
        return None
    polygon_uv, mask = sampled

    # S(u, v) = origin + v * axis + radius * (cos(u) * x_dir + sin(u) * y_dir)
    us = polygon_uv[:, 0]
    vs = polygon_uv[:, 1]
    polyline_3d = (
        origin
        + vs[:, None] * axis
        + radius * (jnp.cos(us)[:, None] * x_dir + jnp.sin(us)[:, None] * y_dir)
    )

    return CylinderTrimFrame(
        origin=origin,
        axis=axis,
        x_dir=x_dir,
        y_dir=y_dir,
        radius=radius,
        sign_flip=sign_flip,
        polygon_uv=polygon_uv,
        polyline_3d=polyline_3d,
        mask=mask,
    )


def extract_sphere_trim_frame(
    face: TopoDS_Face,
    max_vertices: int = 64,
) -> SphereTrimFrame | None:
    """Build Marschner-composition inputs for a sphere face.

    Args:
        face: OCCT face whose underlying surface must be
            ``GeomAbs_Sphere``; any other surface type returns ``None``.
        max_vertices: Fixed polygon capacity after padding.  Raises
            ``ValueError`` when the actual sample count exceeds this
            capacity.

    Returns:
        :class:`SphereTrimFrame`, or ``None`` when the face is not a
        sphere or has no outer wire.

    Examples:
        >>> from brepax.brep.trim_frame import extract_sphere_trim_frame
        >>> # tf = extract_sphere_trim_frame(sphere_face)
        >>> # assert tf is not None
        >>> # assert tf.polyline_3d.shape == (64, 3)
    """
    adaptor = BRepAdaptor_Surface(face)
    if adaptor.GetType() != GeomAbs_Sphere:
        return None

    gp_sph = adaptor.Sphere()
    position = gp_sph.Position()

    loc = position.Location()
    center = jnp.array([loc.X(), loc.Y(), loc.Z()])

    ax = position.Direction()
    axis = jnp.array([ax.X(), ax.Y(), ax.Z()])

    xd = position.XDirection()
    x_dir = jnp.array([xd.X(), xd.Y(), xd.Z()])
    yd = position.YDirection()
    y_dir = jnp.array([yd.X(), yd.Y(), yd.Z()])

    radius = jnp.asarray(gp_sph.Radius())

    # FORWARD sphere faces (e.g. outside of a ball) have their outward
    # normal radially away from the center; REVERSED faces (e.g. a
    # spherical hollow) have it pointing toward the center.  sign_flip
    # is applied to the Sphere primitive's radial signed distance in
    # the composition wrapper, matching extract_cylinder_trim_frame.
    sign_flip = jnp.asarray(1.0 if face.Orientation() == TopAbs_FORWARD else -1.0)

    sampled = _sample_outer_wire_uv(face, max_vertices)
    if sampled is None:
        return None
    polygon_uv, mask = sampled

    # S(u, v) = center
    #        + r * (cos(v) * (cos(u) * x_dir + sin(u) * y_dir)
    #               + sin(v) * axis)
    us = polygon_uv[:, 0]
    vs = polygon_uv[:, 1]
    cos_v = jnp.cos(vs)[:, None]
    sin_v = jnp.sin(vs)[:, None]
    equator_dir = jnp.cos(us)[:, None] * x_dir + jnp.sin(us)[:, None] * y_dir
    polyline_3d = center + radius * (cos_v * equator_dir + sin_v * axis)

    return SphereTrimFrame(
        center=center,
        axis=axis,
        x_dir=x_dir,
        y_dir=y_dir,
        radius=radius,
        sign_flip=sign_flip,
        polygon_uv=polygon_uv,
        polyline_3d=polyline_3d,
        mask=mask,
    )


def plane_face_sdf_from_frame(
    frame: PlaneTrimFrame,
    query: Float[Array, 3],
    sharpness: float = 200.0,
) -> Float[Array, ""]:
    """Trim-aware signed distance to a plane face, from a pre-extracted frame.

    Pure JAX, jittable.  Combines the signed half-space distance
    ``d_s = normal . query - offset`` with the trim-aware composition
    from ``trim_sdf.trim_aware_sdf`` using the frame's ``polygon_uv``,
    ``polyline_3d``, and ``mask``.  The foot-of-perpendicular UV is the
    projection of ``query`` onto the plane's 2D frame; because
    ``frame_u`` and ``frame_v`` are orthogonal to ``normal``, the
    projection collapses to two dot products against the frame basis.

    Args:
        frame: Extracted plane-face data from
            :func:`extract_plane_trim_frame`.
        query: 3D query point, shape ``(3,)``.
        sharpness: Trim-indicator sigmoid sharpness; forwarded to
            ``trim_aware_sdf``.

    Returns:
        Signed scalar distance.  Negative strictly inside the trimmed
        face's half-space, positive outside (including in the phantom
        region where the query is on the inside of the infinite
        half-space but outside the trim polygon).
    """
    # Compute d_s via delta to keep the subtraction in coordinate space,
    # avoiding catastrophic cancellation when query and origin are both
    # far from the world origin but close to each other.
    delta = query - frame.origin
    d_s = jnp.dot(frame.normal, delta)
    foot_uv = jnp.stack([jnp.dot(delta, frame.frame_u), jnp.dot(delta, frame.frame_v)])
    return trim_aware_sdf(
        query,
        d_s,
        foot_uv,
        frame.polygon_uv,
        frame.mask,
        frame.polyline_3d,
        frame.mask,
        sharpness=sharpness,
    )


def plane_face_sdf(
    face: TopoDS_Face,
    query: Float[Array, 3],
    max_vertices: int = 64,
    sharpness: float = 200.0,
) -> Float[Array, ""] | None:
    """Convenience wrapper: extract the frame and compose in one call.

    Intended for one-off evaluations.  For grids or batches, extract
    the frame once via :func:`extract_plane_trim_frame` and JIT the
    composition over queries via :func:`plane_face_sdf_from_frame`.

    Returns ``None`` when the face is not a plane or has no outer wire.
    """
    frame = extract_plane_trim_frame(face, max_vertices=max_vertices)
    if frame is None:
        return None
    return plane_face_sdf_from_frame(frame, query, sharpness=sharpness)


_EPS_SQ_CYLINDER = 1e-24


def cylinder_face_sdf_from_frame(
    frame: CylinderTrimFrame,
    query: Float[Array, 3],
    sharpness: float = 200.0,
) -> Float[Array, ""]:
    """Trim-aware signed distance to a cylinder face, from a pre-extracted frame.

    Pure JAX, jittable.  Computes the cylinder primitive's signed
    distance, applies ``sign_flip`` to get the outward-aware half-space
    sign, and composes via :func:`trim_aware_sdf`.  The foot-of-
    perpendicular UV is ``(theta, v)`` with ``theta`` wrapped to the
    ``[0, 2*pi]`` range that OCCT uses for its cylinder parameter
    polygon, and ``v`` the axial projection.

    Axis-on queries (``perp == 0``) hit the safe-square pattern: the
    radial norm is evaluated on a shifted squared quantity so both
    the forward value (``perp = 0``) and the VJP stay finite.

    Args:
        frame: Extracted cylinder-face data from
            :func:`extract_cylinder_trim_frame`.
        query: 3D query point, shape ``(3,)``.
        sharpness: Trim-indicator sigmoid sharpness; forwarded to
            ``trim_aware_sdf``.

    Returns:
        Signed scalar distance.  Negative strictly inside the trimmed
        face's half-space (as declared by ``sign_flip``); positive
        outside, including the phantom region where the query is
        axially outside the trim range but the untrimmed cylinder
        classification would have lied.

    Examples:
        >>> from brepax.brep.trim_frame import (
        ...     cylinder_face_sdf_from_frame,
        ...     extract_cylinder_trim_frame,
        ... )
        >>> # tf = extract_cylinder_trim_frame(cylinder_face)
        >>> # d = cylinder_face_sdf_from_frame(tf, jnp.array([0., 0., 5.]))
    """
    delta = query - frame.origin
    axial_len = jnp.dot(delta, frame.axis)
    radial_vec = delta - axial_len * frame.axis

    radial_sq = jnp.dot(radial_vec, radial_vec)
    is_off_axis = radial_sq > _EPS_SQ_CYLINDER
    safe_sq = jnp.where(is_off_axis, radial_sq, 1.0)
    safe_norm = jnp.sqrt(safe_sq)
    perp = jnp.where(is_off_axis, safe_norm, 0.0)

    d_s_raw = perp - frame.radius
    d_s = frame.sign_flip * d_s_raw

    # Double-where guard: arctan2(0, 0) has NaN gradient in its
    # backward pass, and jnp.where evaluates both branches for VJP.
    # Substitute dummy arguments on the on-axis branch so that the
    # untaken arctan2 still receives a non-degenerate input.
    y_comp = jnp.dot(radial_vec, frame.y_dir)
    x_comp = jnp.dot(radial_vec, frame.x_dir)
    theta_raw = jnp.where(
        is_off_axis,
        jnp.arctan2(
            jnp.where(is_off_axis, y_comp, 1.0),
            jnp.where(is_off_axis, x_comp, 1.0),
        ),
        0.0,
    )
    theta = jnp.mod(theta_raw, 2.0 * jnp.pi)
    foot_uv = jnp.stack([theta, axial_len])

    return trim_aware_sdf(
        query,
        d_s,
        foot_uv,
        frame.polygon_uv,
        frame.mask,
        frame.polyline_3d,
        frame.mask,
        sharpness=sharpness,
    )


def cylinder_face_sdf(
    face: TopoDS_Face,
    query: Float[Array, 3],
    max_vertices: int = 64,
    sharpness: float = 200.0,
) -> Float[Array, ""] | None:
    """Convenience wrapper: extract the frame and compose in one call.

    Intended for one-off evaluations.  For grids or batches, extract
    the frame once via :func:`extract_cylinder_trim_frame` and JIT the
    composition over queries via
    :func:`cylinder_face_sdf_from_frame`.

    Returns ``None`` when the face is not a cylinder or has no outer
    wire.

    Examples:
        >>> from brepax.brep.trim_frame import cylinder_face_sdf
        >>> # d = cylinder_face_sdf(cylinder_face, jnp.array([0., 0., 5.]))
    """
    frame = extract_cylinder_trim_frame(face, max_vertices=max_vertices)
    if frame is None:
        return None
    return cylinder_face_sdf_from_frame(frame, query, sharpness=sharpness)


_EPS_SQ_SPHERE = 1e-24


def sphere_face_sdf_from_frame(
    frame: SphereTrimFrame,
    query: Float[Array, 3],
    sharpness: float = 200.0,
) -> Float[Array, ""]:
    """Trim-aware signed distance to a sphere face, from a pre-extracted frame.

    Pure JAX, jittable.  Computes the sphere primitive's signed
    distance ``|delta| - radius``, applies ``sign_flip``, and composes
    via :func:`trim_aware_sdf`.  The foot-of-perpendicular UV is
    ``(u, v)`` where ``u`` is longitude around the polar axis
    (wrapped to ``[0, 2*pi]``) and ``v`` is latitude in
    ``[-pi/2, pi/2]``, matching OCCT's Geom_Sphere parameterisation.

    Two degenerate inputs need safe-square / double-where guards so
    gradients stay finite:

    - ``query == center``: distance is zero and the foot direction is
      undefined.  The safe-square pattern evaluates ``sqrt`` on a
      shifted squared norm and the UV falls back to ``(0, 0)``.
    - ``query on axis`` (radial component zero, axial non-zero): the
      longitude ``u`` is undefined at the pole.  The untaken
      ``arctan2`` branch receives dummy non-degenerate arguments.

    Args:
        frame: Extracted sphere-face data from
            :func:`extract_sphere_trim_frame`.
        query: 3D query point, shape ``(3,)``.
        sharpness: Trim-indicator sigmoid sharpness; forwarded to
            ``trim_aware_sdf``.

    Returns:
        Signed scalar distance.  Negative strictly inside the trimmed
        face's half-space (per ``sign_flip``); positive outside,
        including the phantom region where the query's latitude is
        outside the trim range but the untrimmed sphere classification
        would have lied.

    Examples:
        >>> from brepax.brep.trim_frame import (
        ...     extract_sphere_trim_frame,
        ...     sphere_face_sdf_from_frame,
        ... )
        >>> # tf = extract_sphere_trim_frame(sphere_face)
        >>> # d = sphere_face_sdf_from_frame(tf, jnp.array([0., 4., 3.]))
    """
    delta = query - frame.center

    dist_sq = jnp.dot(delta, delta)
    is_not_at_center = dist_sq > _EPS_SQ_SPHERE
    safe_dist_sq = jnp.where(is_not_at_center, dist_sq, 1.0)
    safe_dist = jnp.sqrt(safe_dist_sq)
    dist = jnp.where(is_not_at_center, safe_dist, 0.0)

    d_s_raw = dist - frame.radius
    d_s = frame.sign_flip * d_s_raw

    axial_component = jnp.dot(delta, frame.axis)
    radial_vec = delta - axial_component * frame.axis
    radial_sq = jnp.dot(radial_vec, radial_vec)
    is_off_axis = radial_sq > _EPS_SQ_SPHERE
    safe_radial_sq = jnp.where(is_off_axis, radial_sq, 1.0)
    safe_radial_norm = jnp.sqrt(safe_radial_sq)
    radial_norm = jnp.where(is_off_axis, safe_radial_norm, 0.0)

    # foot_v = atan2(axial, radial_norm); valid even at center because
    # atan2(0, 0) = 0 — but its backward pass is NaN, so protect with
    # the double-where idiom on the untaken branch.
    foot_v = jnp.where(
        is_not_at_center,
        jnp.arctan2(
            jnp.where(is_not_at_center, axial_component, 0.0),
            jnp.where(is_not_at_center, radial_norm, 1.0),
        ),
        0.0,
    )

    # foot_u = atan2(radial . y_dir, radial . x_dir); undefined on the
    # polar axis where the radial vector vanishes.  Same double-where.
    y_comp = jnp.dot(radial_vec, frame.y_dir)
    x_comp = jnp.dot(radial_vec, frame.x_dir)
    u_raw = jnp.where(
        is_off_axis,
        jnp.arctan2(
            jnp.where(is_off_axis, y_comp, 1.0),
            jnp.where(is_off_axis, x_comp, 1.0),
        ),
        0.0,
    )
    foot_u = jnp.mod(u_raw, 2.0 * jnp.pi)

    foot_uv = jnp.stack([foot_u, foot_v])

    return trim_aware_sdf(
        query,
        d_s,
        foot_uv,
        frame.polygon_uv,
        frame.mask,
        frame.polyline_3d,
        frame.mask,
        sharpness=sharpness,
    )


def sphere_face_sdf(
    face: TopoDS_Face,
    query: Float[Array, 3],
    max_vertices: int = 64,
    sharpness: float = 200.0,
) -> Float[Array, ""] | None:
    """Convenience wrapper: extract the frame and compose in one call.

    Intended for one-off evaluations.  For grids or batches, extract
    the frame once via :func:`extract_sphere_trim_frame` and JIT the
    composition over queries via :func:`sphere_face_sdf_from_frame`.

    Returns ``None`` when the face is not a sphere or has no outer
    wire.

    Examples:
        >>> from brepax.brep.trim_frame import sphere_face_sdf
        >>> # d = sphere_face_sdf(sphere_face, jnp.array([0., 4., 3.]))
    """
    frame = extract_sphere_trim_frame(face, max_vertices=max_vertices)
    if frame is None:
        return None
    return sphere_face_sdf_from_frame(frame, query, sharpness=sharpness)


__all__ = [
    "CylinderTrimFrame",
    "PlaneTrimFrame",
    "SphereTrimFrame",
    "cylinder_face_sdf",
    "cylinder_face_sdf_from_frame",
    "extract_cylinder_trim_frame",
    "extract_plane_trim_frame",
    "extract_sphere_trim_frame",
    "plane_face_sdf",
    "plane_face_sdf_from_frame",
    "sphere_face_sdf",
    "sphere_face_sdf_from_frame",
]
