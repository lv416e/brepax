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
from brepax.brep.csg_stump import CSGStump, _evaluate_dnf_sdf
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
from brepax.primitives._base import Primitive

# Union of every supported trim-frame type.
TrimFrame = (
    PlaneTrimFrame
    | CylinderTrimFrame
    | SphereTrimFrame
    | ConeTrimFrame
    | TorusTrimFrame
)


def _dispatch_primitive_sdf(
    primitive: Primitive,
    frame: Any,
    query: Float[Array, 3],
    sharpness: float,
) -> Float[Array, ""]:
    """Per-slot SDF for CSG composition.

    Plane primitives use their untrimmed half-space SDF; curved
    primitives route through the Marschner trim-aware wrapper.
    ``isinstance`` resolves at JAX trace time since each slot has a
    concrete Python type.
    """
    if isinstance(frame, PlaneTrimFrame):
        return primitive.sdf(query)
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
    """CSG-Stump wrapped with per-primitive trim-aware SDFs.

    Each primitive's untrimmed ``.sdf()`` is replaced by its
    Marschner signed-blend composition; the DNF composition is
    unchanged.  Phantom material from untrimmed half-space extensions
    is eliminated at the primitive level, so the existing
    intersection-matrix + union-mask logic carries correct signs.
    """

    primitives: tuple[Primitive, ...]
    frames: tuple[Any, ...]
    intersection_matrix: np.ndarray = eqx.field(static=True)
    union_mask: np.ndarray = eqx.field(static=True)
    sharpness: float = eqx.field(static=True, default=200.0)

    def sdf(self, x: Float[Array, "... 3"]) -> Float[Array, ...]:
        """Composite CSG SDF with per-primitive trim-awareness.

        Plane primitives contribute their untrimmed half-space SDF;
        curved primitives contribute the Marschner signed blend.
        Composition across primitives uses the same DNF
        (intersection-matrix + union-mask) as the untrimmed stump.
        """

        def _single(query: Float[Array, 3]) -> Float[Array, ""]:
            sdfs = jnp.stack(
                [
                    _dispatch_primitive_sdf(p, f, query, self.sharpness)
                    for p, f in zip(self.primitives, self.frames, strict=True)
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
        """Differentiable volume via grid integration of the trim-aware SDF."""
        if lo is None or hi is None:
            # The frames' primitive parameters are captured inside the
            # frames themselves; fall back on the caller to provide
            # bounds if the stump has no primitives list.  The common
            # case is that TrimmedCSGStump is built from an existing
            # CSGStump whose bbox is known.
            raise ValueError("lo and hi must be provided")

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
    curved_reversed_slots: list[int] = []
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
        # Curved primitives route through the trim-aware wrapper,
        # which bakes ``sign_flip`` into the returned distance.  For a
        # REVERSED curved face, this negates the primitive's raw
        # signed distance, so the matching column in the intersection
        # matrix must flip sign.  Plane primitives keep the raw SDF
        # (see module docstring) and need no matrix adjustment.
        if (
            not isinstance(frame, PlaneTrimFrame)
            and face.Orientation() != TopAbs_FORWARD
        ):
            curved_reversed_slots.append(prim_idx)

    matrix = np.asarray(stump.intersection_matrix, dtype=np.float64).copy()
    for slot in curved_reversed_slots:
        matrix[:, slot] = -matrix[:, slot]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="A JAX array is being set as static")
        return TrimmedCSGStump(
            primitives=tuple(stump.primitives),
            frames=tuple(frames),
            intersection_matrix=matrix,
            union_mask=np.asarray(stump.union_mask),
            sharpness=sharpness,
        )


__all__ = [
    "TrimFrame",
    "TrimmedCSGStump",
    "enrich_with_trim_frames",
]
