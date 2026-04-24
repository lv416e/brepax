"""Trim-aware CSG-Stump composition.

Wires the Marschner signed-blend trim-aware SDF into CSG-Stump DNF
composition for *curved* primitives only: cylinder, sphere, cone,
torus.  Plane primitives continue to use their raw half-space SDF.

The asymmetry is deliberate.  Marschner replaces a primitive's
untrimmed signed distance with a face-surface-distance that is
positive when the query projects outside the trim polygon.  For
curved primitives the untrimmed extension is the source of phantom
material in CSG (infinite cylinders extending past a solid's bounds,
etc.) and the Marschner swap kills it.  For plane primitives the
untrimmed half-space is *already* the correct CSG ingredient — a
closed convex polyhedron is exactly the intersection of its half-
spaces — so swapping in a face-patch distance breaks the DNF: the
x=0 face of a box does not itself bound the solid by a distance of
"how far is the query from the x=0 face patch", and that mis-
classifies queries that are inside the x=0 half-space but project
outside the face patch.

Each position in ``frames`` has a concrete Python type, so the
``isinstance`` dispatch resolves at trace time and the method JITs
cleanly.
"""

from __future__ import annotations

import warnings
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from brepax._occt.backend import (
    BRepAdaptor_Surface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopExp_Explorer,
    TopoDS,
)
from brepax._occt.types import TopoDS_Face, TopoDS_Shape
from brepax.brep.csg_eval import integrate_sdf_volume, make_grid_3d
from brepax.brep.csg_stump import CSGStump, _evaluate_dnf_sdf, _primitives_bounds
from brepax.brep.trim_frame import (
    ConeTrimFrame,
    CylinderTrimFrame,
    PlaneTrimFrame,
    SphereTrimFrame,
    TorusTrimFrame,
    cone_face_sdf_from_frame,
    cylinder_face_sdf_from_frame,
    extract_cone_trim_frame,
    extract_cylinder_trim_frame,
    extract_plane_trim_frame,
    extract_sphere_trim_frame,
    extract_torus_trim_frame,
    sphere_face_sdf_from_frame,
    torus_face_sdf_from_frame,
)

# Union of every supported trim-frame type.
TrimFrame = (
    PlaneTrimFrame
    | CylinderTrimFrame
    | SphereTrimFrame
    | ConeTrimFrame
    | TorusTrimFrame
)


def _dispatch_frame_sdf(
    frame: Any,
    query: Float[Array, 3],
    sharpness: float,
) -> Float[Array, ""]:
    """Per-slot SDF for CSG composition, computed from the frame alone.

    Plane primitives contribute the raw half-space SDF derived from
    the frame's normal/origin; the Marschner blend is deliberately
    NOT applied to planes because it would replace the half-space
    distance with a face-patch distance and break CSG composition
    (see module docstring).

    Curved primitives (cylinder, sphere, cone, torus) route through
    the Marschner trim-aware wrapper — each wrapper reads its
    parameters from the frame, so gradients through frame fields
    (radius, axis, etc.) flow cleanly.

    ``isinstance`` resolves at JAX trace time since each slot has a
    concrete Python type.
    """
    if isinstance(frame, PlaneTrimFrame):
        # Half-space SDF from the frame's outward-pointing normal.
        # Avoid ``dot(normal, query) - offset`` to dodge catastrophic
        # cancellation when both terms are large and close.
        return jnp.dot(frame.normal, query - frame.origin)
    if isinstance(frame, CylinderTrimFrame):
        return cylinder_face_sdf_from_frame(frame, query, sharpness=sharpness)
    if isinstance(frame, SphereTrimFrame):
        return sphere_face_sdf_from_frame(frame, query, sharpness=sharpness)
    if isinstance(frame, ConeTrimFrame):
        return cone_face_sdf_from_frame(frame, query, sharpness=sharpness)
    if isinstance(frame, TorusTrimFrame):
        return torus_face_sdf_from_frame(frame, query, sharpness=sharpness)
    raise TypeError(
        f"Unsupported trim frame type: {type(frame).__name__}. "
        "Expected one of PlaneTrimFrame / CylinderTrimFrame / "
        "SphereTrimFrame / ConeTrimFrame / TorusTrimFrame."
    )


class TrimmedCSGStump(eqx.Module):
    """CSG-Stump composed from trim-aware per-face frames.

    Each slot holds a ``TrimFrame`` (plane / cylinder / sphere / cone
    / torus); the composite SDF dispatches on frame type to produce
    the right per-primitive signed distance, then composes via the
    same DNF (intersection matrix + union mask) as the untrimmed
    stump.  Frames are the single source of truth for differentiable
    parameters — gradients of ``sdf`` / ``volume`` flow through
    frame fields (``radius``, ``axis``, etc.) directly, so a caller
    can update frames to optimise geometry.
    """

    frames: tuple[Any, ...]
    intersection_matrix: np.ndarray = eqx.field(static=True)
    union_mask: np.ndarray = eqx.field(static=True)
    bbox_lo: Float[Array, 3] | None = eqx.field(default=None)
    bbox_hi: Float[Array, 3] | None = eqx.field(default=None)
    sharpness: float = eqx.field(static=True, default=200.0)

    def sdf(self, x: Float[Array, "... 3"]) -> Float[Array, ...]:
        """Composite CSG SDF with per-frame trim-awareness.

        Plane slots contribute the raw half-space SDF derived from
        the frame; curved slots contribute the Marschner signed
        blend.  Composition uses the same DNF as the untrimmed stump.
        """

        def _single(query: Float[Array, 3]) -> Float[Array, ""]:
            sdfs = jnp.stack(
                [_dispatch_frame_sdf(f, query, self.sharpness) for f in self.frames]
            )
            return _evaluate_dnf_sdf(sdfs, self.intersection_matrix, self.union_mask)

        if x.ndim == 1:
            return _single(x)
        flat = x.reshape(-1, 3)
        out = jax.vmap(_single)(flat)
        return out.reshape(x.shape[:-1])

    def volume(
        self,
        *,
        resolution: int = 64,
        lo: Float[Array, 3] | None = None,
        hi: Float[Array, 3] | None = None,
    ) -> Float[Array, ""]:
        """Differentiable volume via grid integration of the trim-aware SDF.

        Falls back on the bounding box captured at construction time
        when ``lo`` or ``hi`` is not supplied; raises if neither the
        argument nor the stored box is available.
        """
        if lo is None:
            lo = self.bbox_lo
        if hi is None:
            hi = self.bbox_hi
        if lo is None or hi is None:
            raise ValueError(
                "bounds must be supplied either via arguments or "
                "``bbox_lo``/``bbox_hi`` set by ``enrich_with_trim_frames``"
            )

        lo = jax.lax.stop_gradient(lo)
        hi = jax.lax.stop_gradient(hi)

        grid = make_grid_3d(lo, hi, resolution)[0]
        sdf = self.sdf(grid)
        return integrate_sdf_volume(sdf, lo, hi, resolution)


def _iter_faces(shape: TopoDS_Shape) -> list[TopoDS_Face]:
    faces: list[TopoDS_Face] = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        faces.append(TopoDS.Face_s(exp.Current()))
        exp.Next()
    return faces


def _extract_frame_for_face(face: TopoDS_Face, max_vertices: int) -> Any:
    surf_type = BRepAdaptor_Surface(face).GetType()
    if surf_type == GeomAbs_Plane:
        return extract_plane_trim_frame(face, max_vertices=max_vertices)
    if surf_type == GeomAbs_Cylinder:
        return extract_cylinder_trim_frame(face, max_vertices=max_vertices)
    if surf_type == GeomAbs_Sphere:
        return extract_sphere_trim_frame(face, max_vertices=max_vertices)
    if surf_type == GeomAbs_Cone:
        return extract_cone_trim_frame(face, max_vertices=max_vertices)
    if surf_type == GeomAbs_Torus:
        return extract_torus_trim_frame(face, max_vertices=max_vertices)
    return None


def enrich_with_trim_frames(
    stump: CSGStump,
    shape: TopoDS_Shape,
    *,
    max_vertices: int = 64,
    sharpness: float = 200.0,
) -> TrimmedCSGStump:
    """Build a :class:`TrimmedCSGStump` from a reconstructed stump.

    For each primitive in ``stump``, extract the trim frame of its
    source face from ``shape`` using :func:`extract_*_trim_frame`.
    The primitive's ``face_ids`` must be a single-face list (as
    produced by :func:`reconstruct_csg_stump` before
    :func:`group_stump_primitives`); grouped primitives from
    :func:`group_stump_primitives` are not supported because a single
    frame cannot represent multiple faces.

    Args:
        stump: An ungrouped CSG-Stump.
        shape: The OCCT shape the stump was reconstructed from.
        max_vertices: Trim-polygon padding capacity per face.
        sharpness: Sigmoid sharpness for every primitive's trim
            indicator.

    Returns:
        A :class:`TrimmedCSGStump` with the same topology as ``stump``.

    Raises:
        ValueError: If any primitive maps to more than one face or
            to a face whose surface type is not yet supported.
    """
    all_faces = _iter_faces(shape)

    frames: list[Any] = []
    reversed_slots: list[int] = []
    for prim_idx, face_ids in enumerate(stump.face_ids):
        if len(face_ids) != 1:
            raise ValueError(
                f"primitive {prim_idx} maps to {len(face_ids)} faces; "
                "trim-aware wiring requires ungrouped single-face primitives"
            )
        face = all_faces[face_ids[0]]
        frame = _extract_frame_for_face(face, max_vertices)
        if frame is None:
            type_name = BRepAdaptor_Surface(face).GetType()
            raise ValueError(
                f"primitive {prim_idx} has no trim-aware SDF for surface "
                f"type {type_name}"
            )
        frames.append(frame)
        # Frames store the outward-pointing normal (planes, flipped
        # for REVERSED faces) or a signed-flipped radial distance
        # (curved surfaces with sign_flip == -1).  Both negate the
        # primitive's raw OCCT signed distance relative to the
        # CSGStump's intersection matrix convention, so the matching
        # column is negated to preserve the DNF semantics.
        if face.Orientation() != TopAbs_FORWARD:
            reversed_slots.append(prim_idx)

    matrix = np.asarray(stump.intersection_matrix, dtype=np.float64).copy()
    for slot in reversed_slots:
        matrix[:, slot] = -matrix[:, slot]

    # Prefer the bounding box already carried by the stump (OCCT
    # metadata derived from the shape), since ``_primitives_bounds``
    # issues warnings for infinite primitives such as planes.  Fall
    # back on the primitive-derived bounds if the stump has no box.
    if stump.bbox_lo is not None and stump.bbox_hi is not None:
        bbox_lo_arr = jnp.asarray(stump.bbox_lo)
        bbox_hi_arr = jnp.asarray(stump.bbox_hi)
    else:
        bbox_lo_arr, bbox_hi_arr = _primitives_bounds(stump.primitives)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="A JAX array is being set as static")
        return TrimmedCSGStump(
            frames=tuple(frames),
            intersection_matrix=matrix,
            union_mask=np.asarray(stump.union_mask),
            bbox_lo=bbox_lo_arr,
            bbox_hi=bbox_hi_arr,
            sharpness=sharpness,
        )


__all__ = [
    "TrimFrame",
    "TrimmedCSGStump",
    "enrich_with_trim_frames",
]
