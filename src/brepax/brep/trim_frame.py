"""Extract Marschner-composition inputs from an OCCT face.

For each analytical primitive type, this module produces the trim-aware
SDF inputs (frame, polygon in UV, polyline in 3D, validity mask) that
``trim_sdf.trim_aware_sdf`` consumes, on top of the primitive-level
reconstruction already done in ``convert.py``.

Plane is supported first; sphere, cylinder, cone, torus, and BSpline
are follow-ups that reuse the same ``NamedTuple``-style return
convention and the same outer-wire sampling path so caller code stays
uniform.

The helper is a pure Python function (no JIT).  Its outputs are JAX
arrays that feed directly into ``trim_aware_sdf``, which *is* JIT-able.
OCCT traversal is inherently Python-side, so this separation keeps the
Python-side work isolated and runs once per face at reconstruction time.
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
    for i, (u, v) in enumerate(raw_uv):
        polygon_np[i, 0] = u
        polygon_np[i, 1] = v
        mask_np[i] = 1.0

    polygon_uv = jnp.asarray(polygon_np)
    mask = jnp.asarray(mask_np)

    polyline_3d = (
        origin
        + polygon_uv[:, 0:1] * frame_u[None, :]
        + polygon_uv[:, 1:2] * frame_v[None, :]
    )

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


__all__ = [
    "PlaneTrimFrame",
    "extract_plane_trim_frame",
]
