"""Offset planar faces of an imported solid along their outward normals.

A recognized face moves as a rigid plane between conservative event bounds.
Within those bounds the volume and first moment are exact polynomials in the
offsets, and each edit can be rebuilt and validated in OCCT. Topology
changes, drafted or filleted neighbors, and faces with inner loops are out of
scope.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from brepax._occt.types import TopoDS_Shape
from brepax.experimental.applications._edit_validation import volume_and_moment
from brepax.experimental.applications._face_geometry import (
    PlanarFace,
    RejectedFace,
    interacting,
    rebuild_faces,
    recognize_faces,
)
from brepax.experimental.applications._hole_geometry import (
    normalize_shape,
    unify_same_domain,
)
from brepax.experimental.applications.cad_frame import CadFrame


class FaceOffsetModel(eqx.Module):
    """Exact volume and first-moment laws of independent planar face offsets.

    Each method takes ``offsets`` in input length units, one per selected
    face, positive outward. Within ``lower <= offsets <= upper`` the volume
    is ``V0 + sum(area * offset)`` and the first moment adds
    ``area * offset * (centroid + normal * offset / 2)`` per face; these hold
    exactly for the represented faces, up to the CAD tolerance of the
    prepared solid. Outside the bounds the methods return NaN. Methods
    support ``jit``, ``vmap`` and differentiation; the validity predicate
    supports ``jit`` and ``vmap``. Under positive unit scaling ``s``, volume
    scales by ``s**3`` and its offset gradient by ``s**2``.
    """

    reference_volume: Float[Array, ""]
    reference_moment: Float[Array, 3]
    areas: Float[Array, " f"]
    centroids: Float[Array, "f 3"]
    normals: Float[Array, "f 3"]
    lower: Float[Array, " f"]
    upper: Float[Array, " f"]
    scale: Float[Array, ""]
    origin: Float[Array, 3]

    def is_valid(self, offsets: Float[Array, " f"]) -> Bool[Array, ""]:
        """Check that every offset is finite and inside its interval."""
        if offsets.shape != self.areas.shape:
            raise ValueError("Expected one offset per selected face")
        return (
            jnp.all(jnp.isfinite(offsets))
            & jnp.all(offsets >= self.lower)
            & jnp.all(offsets <= self.upper)
        )

    def _terms(
        self, offsets: Float[Array, " f"]
    ) -> tuple[Float[Array, ""], Float[Array, 3]]:
        valid = self.is_valid(offsets)
        travel = jnp.where(valid, offsets, jnp.nan) / self.scale
        volume = self.reference_volume + jnp.sum(self.areas * travel)
        moment = self.reference_moment + jnp.sum(
            (self.areas * travel)[:, None]
            * (self.centroids + self.normals * travel[:, None] / 2),
            axis=0,
        )
        return volume, moment

    def volume(self, offsets: Float[Array, " f"]) -> Float[Array, ""]:
        """Solid volume in input length units cubed."""
        return self._terms(offsets)[0] * self.scale**3

    def center_of_mass(self, offsets: Float[Array, " f"]) -> Float[Array, 3]:
        """Uniform-density center of mass in external coordinates."""
        volume, moment = self._terms(offsets)
        return moment / volume * self.scale + self.origin


@dataclass(frozen=True)
class PreparedFaceOffset:
    """Host result with the selected faces, the frame and the JAX model.

    ``faces`` are ordered like the offset vector and use input units.
    """

    model: FaceOffsetModel
    faces: tuple[PlanarFace, ...]
    frame: CadFrame
    _shape: TopoDS_Shape = field(repr=False)
    _working_faces: tuple[PlanarFace, ...] = field(repr=False)

    def rebuild(self, offsets: Float[Array, " f"]) -> TopoDS_Shape:
        """Rebuild the offsets in OCCT and restore external coordinates.

        This host operation is not differentiable. Raises ValueError for an
        invalid offset vector and RuntimeError when OCCT fails or the result
        disagrees with the slab law. Same-domain faces of the result are
        unified, so face IDs of the rebuilt solid differ from the input's.
        """
        values = jnp.asarray(offsets)
        if values.shape != self.model.areas.shape or not bool(
            self.model.is_valid(values)
        ):
            raise ValueError("Invalid offsets: expected finite values within bounds")
        working = rebuild_faces(
            self._shape,
            self._working_faces,
            np.asarray(values, dtype=np.float64) / self.frame.length_scale,
        )
        return self.frame.restore_shape(working)


def _external(face: PlanarFace, frame: CadFrame) -> PlanarFace:
    scale = frame.length_scale
    centroid = np.asarray(face.centroid) * scale + np.asarray(frame.origin)
    return PlanarFace(
        face.face_id,
        face.normal,
        (float(centroid[0]), float(centroid[1]), float(centroid[2])),
        face.area * scale**2,
        face.perimeter * scale,
        face.lower * scale,
        face.upper * scale,
    )


def recognize_face_offsets(
    shape: TopoDS_Shape,
    *,
    frame: CadFrame | None = None,
    merge_same_domain: bool = False,
) -> tuple[tuple[PlanarFace, ...], tuple[RejectedFace, ...]]:
    """Find the planar faces of one solid that can be offset.

    Returns recognized faces in input units and rejected planar faces with a
    reason; non-planar faces are reported as ``"not planar"``. Recognition
    sweeps and probes each candidate in OCCT and can take seconds per face
    on large parts. ``frame`` and ``merge_same_domain`` behave as in
    :func:`prepare_face_offset`, and face IDs refer to the same solid.
    """
    working, frame = normalize_shape(shape, frame)
    if merge_same_domain:
        working = unify_same_domain(working)
    found, rejected = recognize_faces(working)
    return tuple(_external(face, frame) for face in found), rejected


def prepare_face_offset(
    shape: TopoDS_Shape,
    face_ids: Sequence[int],
    *,
    frame: CadFrame | None = None,
    merge_same_domain: bool = False,
) -> PreparedFaceOffset:
    """Prepare offsets of selected planar faces of one solid.

    Accepts a B-Rep returned by ``read_step``. Each selected face must be
    planar with one boundary loop, and every neighbor must be a
    perpendicular plane or a cylinder whose axis runs along the face normal.
    Each face gets a closed offset interval that ends before the first
    event: the swept face meeting other material, leaving the solid, or
    running past the end of a neighbor. Intervals are derived conservatively
    from bounding boxes of OCCT Boolean results, keep a margin of 10 CAD
    tolerances, and are capped at twice the part size. A face whose Boolean
    already disagrees with the law at a tiny offset is rejected.

    Selected faces must be pairwise independent: not adjacent, and with
    disjoint boxes around their full swept ranges. Offsets of such faces add
    up exactly; coupled moves are out of scope.

    ``frame`` and ``merge_same_domain`` behave as in ``prepare_hole_edit``:
    with ``merge_same_domain=True``, face IDs refer to the unified working
    solid, so take them from :func:`recognize_face_offsets` with the same
    setting. Requires JAX x64.

    Raises ValueError for invalid CAD, duplicate or unsupported selections,
    or interacting faces.

    Example:
        >>> import jax
        >>> from brepax.io.step import read_step
        >>> jax.config.update("jax_enable_x64", True)
        >>> shape = read_step("bracket.step")
        >>> faces, _ = recognize_face_offsets(shape)
        >>> edit = prepare_face_offset(shape, [faces[0].face_id])
        >>> offsets = jax.numpy.array([faces[0].upper / 2])
        >>> volume, gradient = jax.value_and_grad(edit.model.volume)(offsets)
        >>> edited_cad = edit.rebuild(offsets)
    """
    if jnp.asarray(0.0).dtype != np.dtype("float64"):
        raise ValueError("Face offsets require jax_enable_x64=True")
    ids = tuple(int(i) for i in face_ids)
    if not ids:
        raise ValueError("Select at least one face")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate face IDs")
    working, frame = normalize_shape(shape, frame)
    if merge_same_domain:
        working = unify_same_domain(working)
    found, rejected = recognize_faces(working, ids)
    if rejected:
        raise ValueError(f"Selected faces cannot be offset: {rejected}")
    by_id = {face.face_id: face for face in found}
    selected = tuple(by_id[i] for i in ids)
    for first in range(len(selected)):
        for second in range(first + 1, len(selected)):
            if interacting(selected[first], selected[second], working):
                raise ValueError(
                    f"Faces {selected[first].face_id} and {selected[second].face_id} "
                    "may interact; select independent faces"
                )
    volume, moment = volume_and_moment(working)
    external = tuple(_external(face, frame) for face in selected)
    model = FaceOffsetModel(
        reference_volume=jnp.asarray(volume),
        reference_moment=jnp.asarray(moment),
        areas=jnp.array([face.area for face in selected]),
        centroids=jnp.array([face.centroid for face in selected]),
        normals=jnp.array([face.normal for face in selected]),
        lower=jnp.array([face.lower for face in external]),
        upper=jnp.array([face.upper for face in external]),
        scale=jnp.asarray(frame.length_scale),
        origin=jnp.asarray(frame.origin),
    )
    return PreparedFaceOffset(model, external, frame, working, selected)


__all__ = [
    "FaceOffsetModel",
    "PlanarFace",
    "PreparedFaceOffset",
    "RejectedFace",
    "prepare_face_offset",
    "recognize_face_offsets",
]
