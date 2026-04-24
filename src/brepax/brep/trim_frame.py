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
    GeomAbs_Plane,
    TopAbs_FORWARD,
)
from brepax._occt.types import TopoDS_Face
from brepax.brep.trim_sdf import trim_aware_sdf

_SAMPLES_PER_EDGE = 8


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
    offset = jnp.dot(normal, origin)

    xd = position.XDirection()
    frame_u = jnp.array([xd.X(), xd.Y(), xd.Z()])
    yd = position.YDirection()
    frame_v = jnp.array([yd.X(), yd.Y(), yd.Z()])

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

    polygon_uv = jnp.asarray(polygon_np)
    mask = jnp.asarray(mask_np)

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
    delta = query - frame.origin
    d_s = jnp.dot(frame.normal, query) - frame.offset
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


__all__ = [
    "PlaneTrimFrame",
    "extract_plane_trim_frame",
    "plane_face_sdf",
    "plane_face_sdf_from_frame",
]
