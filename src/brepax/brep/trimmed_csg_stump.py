"""Trim-aware CSG-Stump composition.

Wires CSG-Stump composition for primitives reconstructed from
trimmed B-Rep faces.

Per ADR-0019, every *analytical* primitive (plane, cylinder, sphere,
cone, torus) contributes its raw untrimmed signed distance to the DNF.
The untrimmed half-space *is* the correct CSG ingredient: a closed
solid is exactly the intersection of half-spaces, and substituting in
a trimmed face-patch SDF (which collapses to a non-negative boundary
distance outside the trim parameter range) breaks that composition.
This holds uniformly for every analytical surface type — plane was not
a special case but the only case the original wiring implemented
correctly by accident.

BSpline primitives are different.  BSpline patches are finite in
parameter space, so the untrimmed signed distance is the source of
phantom material outside the trim region (ADR-0016, Linkrods +219%
measurement on raw ``abs(BSpline SDF)``).  For BSpline slots the
composite SDF routes through the Marschner trim-aware blend
(ADR-0018) by combining the primitive's projection-based signed
distance with the unsigned distance to the 3D trim polyline; the
result classifies queries outside the patch as outside the primitive
and leaves the inside of the trimmed region unchanged.

The Marschner trim-aware blend is also exposed for standalone
trimmed-face distance queries via
``brep/trim_frame.py``'s ``*_face_sdf_from_frame`` wrappers — those
are unaffected by the CSG-Stump integration here.

Analytical primitives only need ``primitive.sdf(query)`` — the same
SDF that :class:`DifferentiableCSGStump` consumes.  BSpline slots
additionally consume the per-slot :class:`BSplineTrimFrame`.
"""

from __future__ import annotations

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from brepax._occt.backend import (
    BRepAdaptor_Surface,
    GeomAbs_BSplineSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    TopAbs_FACE,
    TopExp_Explorer,
    TopoDS,
)
from brepax._occt.types import TopoDS_Face, TopoDS_Shape
from brepax.brep.csg_eval import integrate_sdf_volume, make_grid_3d
from brepax.brep.csg_stump import CSGStump, _evaluate_dnf_sdf, _primitives_bounds
from brepax.brep.trim_frame import (
    BSplineTrimFrame,
    ConeTrimFrame,
    CylinderTrimFrame,
    PlaneTrimFrame,
    SphereTrimFrame,
    TorusTrimFrame,
    bspline_face_sdf_from_frame,
    extract_bspline_trim_frame,
    extract_cone_trim_frame,
    extract_cylinder_trim_frame,
    extract_plane_trim_frame,
    extract_sphere_trim_frame,
    extract_torus_trim_frame,
)
from brepax.primitives._base import Primitive
from brepax.primitives.bspline_surface import BSplineSurface

# Union of every supported trim-frame type.
TrimFrame = (
    PlaneTrimFrame
    | CylinderTrimFrame
    | SphereTrimFrame
    | ConeTrimFrame
    | TorusTrimFrame
    | BSplineTrimFrame
)

# Sigmoid sharpness for the Marschner trim indicator (ADR-0018);
# consumed by the future BSpline-patch dispatch and carried on the
# stump for that integration.
DEFAULT_TRIM_SHARPNESS: float = 200.0


class TrimmedCSGStump(eqx.Module):
    """CSG-Stump enriched with per-face trim metadata.

    Each slot pairs a primitive (``Plane`` / ``Sphere`` / ``Cylinder``
    / ``Cone`` / ``Torus`` / ``BSplineSurface``) with the trim frame
    extracted from its source OCCT face.  The composite SDF dispatches
    on primitive type:

    - **Analytical primitives**: return ``primitive.sdf(query)`` (raw
      untrimmed half-space SDF), per ADR-0019.  Bit-equivalent to
      :class:`DifferentiableCSGStump` for analytical-only models.
    - **BSpline primitives**: return the Marschner trim-aware blend
      via :func:`bspline_face_sdf_from_frame`, eliminating the phantom
      that the untrimmed BSpline SDF carries past the patch boundary.

    The DNF composition (intersection matrix + union mask) is shared
    with :class:`DifferentiableCSGStump`; only the per-slot SDF
    differs.

    Gradients through ``sdf`` and ``volume`` flow through the
    primitives' differentiable parameters (``radius``, ``axis``,
    plane ``normal`` / ``offset``, BSpline ``control_points`` /
    ``weights``, etc.).

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.brep.csg_stump import reconstruct_csg_stump
        >>> from brepax.brep.trimmed_csg_stump import enrich_with_trim_frames
        >>> # Assuming ``shape`` is an OCCT TopoDS_Shape with analytical faces:
        >>> # stump = reconstruct_csg_stump(shape)
        >>> # trimmed = enrich_with_trim_frames(stump, shape)
        >>> # d = trimmed.sdf(jnp.array([0.0, 0.0, 0.0]))   # scalar SDF at a point
        >>> # v = trimmed.volume(resolution=64)             # grid-integrated volume
    """

    primitives: tuple[Primitive, ...]
    frames: tuple[TrimFrame, ...]
    intersection_matrix: np.ndarray = eqx.field(static=True)
    union_mask: np.ndarray = eqx.field(static=True)
    bbox_lo: Float[Array, 3] | None = eqx.field(default=None)
    bbox_hi: Float[Array, 3] | None = eqx.field(default=None)
    sharpness: float = eqx.field(static=True, default=DEFAULT_TRIM_SHARPNESS)

    def sdf(self, x: Float[Array, "... 3"]) -> Float[Array, ...]:
        """Composite CSG SDF.

        For analytical slots, returns ``primitive.sdf(query)``.  For
        BSpline slots, returns the Marschner trim-aware blend via
        :func:`bspline_face_sdf_from_frame`.  Composes via the stump's
        DNF.  ``isinstance`` resolves at JAX trace time since each
        slot has a concrete Python type.
        """

        def _per_slot_sdf(
            prim: Primitive, frame: TrimFrame, query: Float[Array, 3]
        ) -> Float[Array, ""]:
            if isinstance(prim, BSplineSurface) and isinstance(frame, BSplineTrimFrame):
                return bspline_face_sdf_from_frame(
                    frame, prim, query, sharpness=self.sharpness
                )
            return prim.sdf(query)

        def _single(query: Float[Array, 3]) -> Float[Array, ""]:
            sdfs = jnp.stack(
                [
                    _per_slot_sdf(prim, frame, query)
                    for prim, frame in zip(self.primitives, self.frames, strict=True)
                ]
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
        """Differentiable volume via grid integration of the composite SDF.

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


def _extract_frame_for_face(face: TopoDS_Face, max_vertices: int) -> TrimFrame | None:
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
    if surf_type == GeomAbs_BSplineSurface:
        return extract_bspline_trim_frame(face, max_vertices=max_vertices)
    return None


def enrich_with_trim_frames(
    stump: CSGStump,
    shape: TopoDS_Shape,
    *,
    max_vertices: int = 64,
    sharpness: float = DEFAULT_TRIM_SHARPNESS,
) -> TrimmedCSGStump:
    """Build a :class:`TrimmedCSGStump` from a reconstructed stump.

    For each primitive in ``stump``, extract the trim frame of its
    source face from ``shape`` so the BSpline-patch path described in
    ADR-0018 / ADR-0019 has its per-slot frame ready.  The primitive's
    ``face_ids`` must be a single-face list (as produced by
    :func:`reconstruct_csg_stump` before
    :func:`group_stump_primitives`); grouped primitives from
    :func:`group_stump_primitives` are not supported because a single
    frame cannot represent multiple faces.

    Args:
        stump: An ungrouped CSG-Stump.
        shape: The OCCT shape the stump was reconstructed from.
        max_vertices: Trim-polygon padding capacity per face.
        sharpness: Sigmoid sharpness for the trim indicator; used by
            future Marschner-based dispatches (BSpline) and stored
            on the stump for that purpose.

    Returns:
        A :class:`TrimmedCSGStump` with the same topology as ``stump``.

    Raises:
        ValueError: If any primitive maps to more than one face or
            to a face whose surface type is not yet supported.

    Examples:
        >>> from brepax.brep.csg_stump import reconstruct_csg_stump
        >>> from brepax.brep.trimmed_csg_stump import enrich_with_trim_frames
        >>> from brepax.io.step import read_step
        >>> # shape = read_step("part.step")
        >>> # stump = reconstruct_csg_stump(shape)
        >>> # trimmed = enrich_with_trim_frames(stump, shape, max_vertices=64)
        >>> # assert len(trimmed.frames) == len(stump.primitives)
    """
    all_faces = _iter_faces(shape)

    frames: list[TrimFrame] = []
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

    matrix = np.asarray(stump.intersection_matrix, dtype=np.float64).copy()

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
            primitives=tuple(stump.primitives),
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
