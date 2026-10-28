"""Experimental fixed-topology radius and center editing of STEP through-holes.

Host preparation/reconstruction uses OCCT. The array model couples cylindrical
walls and planar end boundaries on one frozen, welded mesh. It does not trace
through feature recognition, Boolean rebuilding, meshing, or topology changes.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from brepax._occt.types import TopoDS_Shape
from brepax.brep.triangulate import (
    divergence_volume,
    mesh_center_of_mass,
    mesh_surface_area,
    triangulate_shape,
)
from brepax.experimental.applications._hole_geometry import (
    RejectedHole,
    ThroughHole,
    external_hole,
    normalize_shape,
    rebuild_holes,
    recognize_holes,
)
from brepax.experimental.applications._hole_mesh import closed_mesh, displacement_fields
from brepax.experimental.applications.cad_frame import CadFrame


class HoleEditModel(eqx.Module):
    """Pure array model; differentiate radii and shifts with preparation frozen.

    Each method accepts external-unit ``radii`` and optional ``center_shifts``
    with shape (selected holes, 3). Shifts are world-coordinate vectors from
    the original centers, perpendicular to the fixed hole axes. Omission means
    zero shift. Axial components exceeding 64*x64_epsilon*reference_radius are
    invalid; roundoff-sized components are projected out. Shift Jacobians are
    tangent derivatives, with zero axial component, not axial edit support.

    Geometry and metric methods support ``jit``, ``vmap`` and parameter
    differentiation; the Boolean validity predicate supports ``jit``/``vmap``.
    Lengths are in the input CAD's units. Use the metric methods for stable
    evaluation before restoring a potentially large world origin. Scalar
    measures and their radius derivatives are invariant under rigid placement.
    Rotate shifts along with the CAD: scalar shift gradients rotate as vectors,
    while the center-of-mass shift Jacobian transforms as R*J*R.T. Under positive
    unit scaling s, volume/area scale by s**3/s**2 and length gradients by
    s**2/s. Center-of-mass length Jacobians do not scale. Points and centers
    transform as points. These laws hold
    up to CAD tolerance and discretization error after fresh preparation.

    ``lower_radii``/``upper_radii`` bound radii. Shift magnitude plus positive
    radius growth must not exceed ``upper_radii - reference_radii``. Both use
    the same clearance budget, including when shrinking the hole. ``is_valid``
    additionally rejects collapsed or locally reversed triangles. Invalid edits
    produce NaN geometry/measures; derivatives there are undefined. The validity test
    is not a general mesh self-intersection or CAD certification algorithm.
    """

    vertices: Float[Array, "v 3"]
    faces: Int[Array, "t 3"]
    fields: Float[Array, "h v 3"]
    translation_weights: Float[Array, "h v"]
    axes: Float[Array, "h 3"]
    reference_radii: Float[Array, " h"]
    lower_radii: Float[Array, " h"]
    upper_radii: Float[Array, " h"]
    scale: Float[Array, ""]
    origin: Float[Array, "3"]

    def _shifts(self, center_shifts: Float[Array, "h 3"] | None) -> Float[Array, "h 3"]:
        if center_shifts is None:
            return jnp.zeros_like(self.axes)
        if center_shifts.shape != self.axes.shape:
            raise ValueError("Expected one 3D center shift per selected hole")
        return center_shifts

    def _transverse(self, shifts: Float[Array, "h 3"]) -> Float[Array, "h 3"]:
        return shifts - jnp.sum(shifts * self.axes, axis=-1)[:, None] * self.axes

    def _triangles(
        self, radii: Float[Array, " h"], shifts: Float[Array, "h 3"]
    ) -> Float[Array, "t 3 3"]:
        if radii.shape != self.reference_radii.shape:
            raise ValueError("Expected one radius per selected hole")
        delta = (radii - self.reference_radii) / self.scale
        vertices = self.vertices + jnp.einsum("h,hvi->vi", delta, self.fields)
        vertices = vertices + jnp.einsum(
            "hv,hi->vi", self.translation_weights, self._transverse(shifts) / self.scale
        )
        return vertices[self.faces]

    def _valid(
        self,
        radii: Float[Array, " h"],
        shifts: Float[Array, "h 3"],
        triangles: Float[Array, "t 3 3"],
    ) -> Bool[Array, ""]:
        original = self.vertices[self.faces]
        normals = jnp.cross(
            original[:, 1] - original[:, 0], original[:, 2] - original[:, 0]
        )
        changed = jnp.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        # Relative oriented-area margin, independent of length units.
        orientation = jnp.sum(normals * changed, axis=-1)
        area_squared = jnp.sum(normals**2, axis=-1)
        # Translation and positive radial growth share the same clearance.
        # Squared norms avoid a singular norm derivative at the zero-shift default.
        slack = 64 * np.finfo(np.float64).eps * self.reference_radii
        remaining = self.upper_radii - jnp.maximum(radii, self.reference_radii)
        return (
            jnp.all(jnp.isfinite(radii))
            & jnp.all(jnp.isfinite(shifts))
            & jnp.all(jnp.abs(jnp.sum(shifts * self.axes, axis=-1)) <= slack)
            & jnp.all(
                jnp.sum(self._transverse(shifts) ** 2, axis=-1)
                <= (remaining + slack) ** 2
            )
            & jnp.all(radii >= self.lower_radii)
            & jnp.all(radii <= self.upper_radii)
            & jnp.all(orientation > 1e-8 * area_squared)
        )

    def is_valid(
        self,
        radii: Float[Array, " h"],
        center_shifts: Float[Array, "h 3"] | None = None,
    ) -> Bool[Array, ""]:
        """Check radius/movement bounds, transverse shifts and mesh orientation."""
        shifts = self._shifts(center_shifts)
        return self._valid(radii, shifts, self._triangles(radii, shifts))

    def working_triangles(
        self,
        radii: Float[Array, " h"],
        center_shifts: Float[Array, "h 3"] | None = None,
    ) -> Float[Array, "t 3 3"]:
        """Return normalized triangles; both parameters still use input units."""
        shifts = self._shifts(center_shifts)
        triangles = self._triangles(radii, shifts)
        return jnp.where(self._valid(radii, shifts, triangles), triangles, jnp.nan)

    def triangles(
        self,
        radii: Float[Array, " h"],
        center_shifts: Float[Array, "h 3"] | None = None,
    ) -> Float[Array, "t 3 3"]:
        """Return external-coordinate triangles, or NaNs for invalid edits."""
        return self.working_triangles(radii, center_shifts) * self.scale + self.origin

    def volume(
        self,
        radii: Float[Array, " h"],
        center_shifts: Float[Array, "h 3"] | None = None,
    ) -> Float[Array, ""]:
        """Signed mesh volume in input length units cubed."""
        return (
            divergence_volume(self.working_triangles(radii, center_shifts))
            * self.scale**3
        )

    def surface_area(
        self,
        radii: Float[Array, " h"],
        center_shifts: Float[Array, "h 3"] | None = None,
    ) -> Float[Array, ""]:
        """Mesh surface area in input length units squared."""
        return (
            mesh_surface_area(self.working_triangles(radii, center_shifts))
            * self.scale**2
        )

    def center_of_mass(
        self,
        radii: Float[Array, " h"],
        center_shifts: Float[Array, "h 3"] | None = None,
    ) -> Float[Array, "3"]:
        """Uniform-density mesh center of mass in external coordinates."""
        return (
            mesh_center_of_mass(self.working_triangles(radii, center_shifts))
            * self.scale
            + self.origin
        )


@dataclass(frozen=True)
class PreparedHoleEdit:
    """Host result with selected holes, rejected candidates, frame and JAX model.

    ``holes`` are ordered like the radius vector; explicit face selections
    preserve caller order, otherwise original face order is used. Unselected
    features remain fixed. Host CAD objects never enter the JAX model.
    """

    model: HoleEditModel
    holes: tuple[ThroughHole, ...]
    rejected: tuple[RejectedHole, ...]
    frame: CadFrame
    _shape: TopoDS_Shape = field(repr=False)
    _working_holes: tuple[ThroughHole, ...] = field(repr=False)

    def rebuild(
        self,
        radii: Float[Array, " h"],
        center_shifts: Float[Array, "h 3"] | None = None,
    ) -> TopoDS_Shape:
        """Rebuild isolated holes in OCCT and restore external coordinates.

        This host operation is not differentiable. Raises ValueError for an
        invalid radius/shift vector and RuntimeError for reconstruction failure
        or disagreement with isolated-hole volume, area or center of mass. Input CAD
        remains owned by the caller; each reconstruction uses an owned copy.
        Same-domain faces/edges are unified, so unselected geometry is preserved
        but original face partitions and IDs are not.
        """
        values = jnp.asarray(radii)
        shifts = self.model._shifts(
            None if center_shifts is None else jnp.asarray(center_shifts)
        )
        if values.shape != self.model.reference_radii.shape or not bool(
            self.model.is_valid(values, shifts)
        ):
            raise ValueError(
                "Invalid radii or center shifts: expected finite bounded values, "
                "transverse shifts and valid mesh orientation"
            )
        working = rebuild_holes(
            self._shape,
            self._working_holes,
            np.asarray(values, dtype=np.float64) / self.frame.length_scale,
            np.asarray(self.model._transverse(shifts)) / self.frame.length_scale,
        )
        return self.frame.restore_shape(working)


def prepare_hole_edit(
    shape: TopoDS_Shape,
    face_ids: Sequence[int] | None = None,
    *,
    frame: CadFrame | None = None,
    deflection: float = 0.001,
) -> PreparedHoleEdit:
    """Prepare bounded edits of full cylindrical through-holes in one solid.

    Accepts a B-Rep returned by ``read_step``; does not assume a stock primitive
    or a construction history. Each selected inward cylinder must be complete,
    with two perpendicular planar ends and sufficient clearance. Blind, split,
    oblique, intersecting and near-boundary holes are unsupported. Rejected
    inward cylinders are reported; an explicit selection of one raises.

    ``frame`` maps input units to working coordinates; by default the bounding
    box is centered and its largest extent becomes 40 working units. It is
    frozen during editing. ``deflection`` is an absolute WORKING-unit meshing
    tolerance, finite and positive. Preparation requires JAX x64; it never
    changes the global JAX precision setting. Input CAD is copied.

    Radius bounds are [0.5*r0, r0 + 0.2*clearance]. Optional transverse center
    shifts share the growth budget: norm(shift) + max(radius-r0, 0) must not
    exceed 0.2*clearance. Moving cylinder samples and
    neighboring planar samples together preserves the welded connectivity.
    Geometry outside the radial support stays fixed. These are local edits,
    not a topology-changing CAD kernel. Mesh orientation may restrict the
    usable interval further; callers must check ``model.is_valid``.

    Raises ValueError for invalid CAD, unsupported selections, no supported
    holes, or a mesh that cannot be welded into a closed oriented surface.
    Unsupported surface evaluators raise NotImplementedError. OCCT extraction
    failures propagate. Meshing accuracy is not certified by preparation.

    Example:
        >>> import jax
        >>> from brepax.io.step import read_step
        >>> jax.config.update("jax_enable_x64", True)
        >>> edit = prepare_hole_edit(read_step("plate_with_holes.step"))
        >>> radii = (edit.model.reference_radii + edit.model.upper_radii) / 2
        >>> assert bool(edit.model.is_valid(radii))
        >>> volume, gradient = jax.jit(jax.value_and_grad(edit.model.volume))(radii)
        >>> edited_cad = edit.rebuild(radii)

    To move a hole, pass ``center_shifts`` in external units to the model and
    ``rebuild``. For a selected Z-axis hole, ``jnp.array([[0.1, 0.0, 0.0]])``
    moves its center in X; check the combined budget with ``is_valid`` first.
    """
    if jnp.asarray(0.0).dtype != np.dtype("float64"):
        raise ValueError("Hole editing requires jax_enable_x64=True")
    if not np.isfinite(deflection) or deflection <= 0:
        raise ValueError("deflection must be finite and positive in working units")
    working, frame = normalize_shape(shape, frame)
    found, rejected, tolerance = recognize_holes(working)
    if face_ids is None:
        selected = found
    else:
        ids = tuple(face_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate hole face IDs")
        by_id = {hole.face_id: hole for hole in found}
        if any(i not in by_id for i in ids):
            raise ValueError(
                f"Selected face is not a supported through-hole; rejected={rejected}"
            )
        selected = tuple(by_id[i] for i in ids)
    if not selected:
        raise ValueError(f"No supported through-holes selected; rejected={rejected}")
    triangles, _ = triangulate_shape(working, deflection=deflection)
    vertices, faces = closed_mesh(np.asarray(triangles), tolerance * 0.1)
    holes = tuple(external_hole(hole, frame) for hole in selected)
    reference = jnp.array([hole.radius for hole in holes])
    fields = displacement_fields(vertices, selected, tolerance)
    model = HoleEditModel(
        vertices=jnp.asarray(vertices),
        faces=jnp.asarray(faces),
        fields=jnp.asarray(fields.radial),
        translation_weights=jnp.asarray(fields.translation),
        axes=jnp.array([hole.axis for hole in holes]),
        reference_radii=reference,
        lower_radii=reference * 0.5,
        upper_radii=reference + 0.2 * jnp.array([hole.clearance for hole in holes]),
        scale=jnp.asarray(frame.length_scale),
        origin=jnp.asarray(frame.origin),
    )
    if not bool(model.is_valid(reference)) or float(model.volume(reference)) <= 0:
        raise ValueError("Reference mesh contains degenerate or inverted geometry")
    return PreparedHoleEdit(model, holes, rejected, frame, working, selected)


__all__ = [
    "HoleEditModel",
    "PreparedHoleEdit",
    "RejectedHole",
    "ThroughHole",
    "prepare_hole_edit",
]
