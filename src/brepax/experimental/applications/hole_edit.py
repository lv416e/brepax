"""Experimental fixed-topology radius and center editing of STEP through-holes.

Host preparation and reconstruction use OCCT. The array model holds the exact
volume, surface-area and first-moment laws of the represented holes, so it
needs no mesh; responses without a law belong to the optional frozen mesh of
:mod:`brepax.experimental.applications.hole_mesh`. Nothing traces through
feature recognition, Boolean rebuilding or topology changes.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from brepax._occt.backend import BRepGProp, GProp_GProps
from brepax._occt.types import TopoDS_Shape
from brepax.experimental.applications._edit_validation import volume_and_moment
from brepax.experimental.applications._hole_geometry import (
    RejectedHole,
    ThroughHole,
    external_hole,
    normalize_shape,
    rebuild_holes,
    recognize_holes,
    unify_same_domain,
)
from brepax.experimental.applications.cad_frame import CadFrame


class HoleEditModel(eqx.Module):
    """Exact laws of isolated holes; differentiate radii and shifts directly.

    Each method accepts external-unit ``radii`` and optional ``center_shifts``
    with shape (selected holes, 3). Shifts are world-coordinate vectors from
    the original centers, perpendicular to the fixed hole axes. Omission means
    zero shift. Axial components exceeding 64*x64_epsilon*reference_radius are
    invalid; roundoff-sized components are projected out. Shift Jacobians are
    tangent derivatives, with zero axial component, not axial edit support.

    Within the bounds, with ``t = r - r0``, a hole of height ``h`` changes the
    volume by ``-pi*h*t*(2*r0 + t)``, the surface area by
    ``2*pi*h*t - 2*pi*t*(2*r0 + t)``, and the first moment by
    ``-pi*h*(t*(2*r0 + t)*c + r**2*s)`` for center ``c`` and transverse shift
    ``s``. The laws hold exactly for holes recognized as complete cylinders
    with planar ends, up to the CAD tolerance of the prepared solid. The
    reference measures come from adaptive OCCT integration to 1e-9 relative.

    Geometry and metric methods support ``jit``, ``vmap`` and parameter
    differentiation; the Boolean validity predicate supports ``jit``/``vmap``.
    Lengths are in the input CAD's units. Scalar measures and their radius
    derivatives are invariant under rigid placement. Rotate shifts along with
    the CAD: scalar shift gradients rotate as vectors, while the
    center-of-mass shift Jacobian transforms as R*J*R.T. Under positive unit
    scaling s, volume/area scale by s**3/s**2 and length gradients by s**2/s.
    Center-of-mass length Jacobians do not scale. Centers transform as points.

    ``lower_radii``/``upper_radii`` bound radii one hole at a time: an upper
    radius stops ``contact_margin`` short of the nearest feature, with the other
    holes unedited. Each hole's reach, its shift magnitude plus positive radius
    growth, must not exceed ``upper_radii - reference_radii``, including when
    shrinking the hole. Two selected holes must also keep ``contact_margin``
    between them, so their reaches must sum to at most their distance less that
    margin, and all holes at their upper radii are generally invalid together.
    Invalid edits produce NaN measures; derivatives there are undefined.

    This range only keeps the laws true. A wall this thin is rarely a design
    you want: state the distance to keep with ``ligament_bounds``, or with
    :func:`~brepax.experimental.applications.dimension_query.hole_ligament_constraints`.

    ``fixed_clearance`` is each hole's distance to every boundary face other
    than its end faces and the selected holes, ``hole_clearance`` its distance
    to each selected hole (infinite on the diagonal). Both are in input units;
    ``ligament_bounds`` uses them.
    """

    reference_volume: Float[Array, ""]
    reference_area: Float[Array, ""]
    reference_moment: Float[Array, "3"]
    centers: Float[Array, "h 3"]
    axes: Float[Array, "h 3"]
    heights: Float[Array, " h"]
    working_radii: Float[Array, " h"]
    reference_radii: Float[Array, " h"]
    lower_radii: Float[Array, " h"]
    upper_radii: Float[Array, " h"]
    fixed_clearance: Float[Array, " h"]
    hole_clearance: Float[Array, "h h"]
    contact_margin: Float[Array, ""]
    scale: Float[Array, ""]
    origin: Float[Array, "3"]

    def shifts_or_zero(
        self, center_shifts: Float[Array, "h 3"] | None
    ) -> Float[Array, "h 3"]:
        """Return validated shifts, with zeros when they are omitted."""
        if center_shifts is None:
            return jnp.zeros_like(self.axes)
        if center_shifts.shape != self.axes.shape:
            raise ValueError("Expected one 3D center shift per selected hole")
        return center_shifts

    def transverse(self, shifts: Float[Array, "h 3"]) -> Float[Array, "h 3"]:
        """Project shifts onto the planes perpendicular to the hole axes."""
        return shifts - jnp.sum(shifts * self.axes, axis=-1)[:, None] * self.axes

    def shift_lengths(self, shifts: Float[Array, "h 3"]) -> Float[Array, " h"]:
        """Transverse shift magnitudes, differentiable at the zero-shift default."""
        squared = jnp.sum(self.transverse(shifts) ** 2, axis=-1)
        # The norm's derivative is singular at zero.
        safe = jnp.where(squared > 0, squared, 1.0)
        return jnp.where(squared > 0, jnp.sqrt(safe), 0.0)

    def is_valid(
        self,
        radii: Float[Array, " h"],
        center_shifts: Float[Array, "h 3"] | None = None,
    ) -> Bool[Array, ""]:
        """Check radius bounds, the reach of each hole and pair, and transversality."""
        if radii.shape != self.reference_radii.shape:
            raise ValueError("Expected one radius per selected hole")
        shifts = self.shifts_or_zero(center_shifts)
        # Translation and positive radial growth share the same clearance.
        # Squared norms avoid a singular norm derivative at the zero-shift default.
        slack = 64 * np.finfo(np.float64).eps * self.reference_radii
        grown = jnp.maximum(radii, self.reference_radii)
        remaining = self.upper_radii - grown
        reach = grown - self.reference_radii + self.shift_lengths(shifts)
        between = (
            self.hole_clearance - self.contact_margin - reach[None, :] - reach[:, None]
        )
        return (
            jnp.all(jnp.isfinite(radii))
            & jnp.all(jnp.isfinite(shifts))
            & jnp.all(jnp.abs(jnp.sum(shifts * self.axes, axis=-1)) <= slack)
            & jnp.all(
                jnp.sum(self.transverse(shifts) ** 2, axis=-1)
                <= (remaining + slack) ** 2
            )
            & jnp.all(between >= -slack[:, None])
            & jnp.all(radii >= self.lower_radii)
            & jnp.all(radii <= self.upper_radii)
        )

    def _edit(
        self,
        radii: Float[Array, " h"],
        center_shifts: Float[Array, "h 3"] | None,
    ) -> tuple[Float[Array, " h"], Float[Array, " h"], Float[Array, "h 3"]]:
        """Radius change, radius-squared change and shift in working units."""
        valid = self.is_valid(radii, center_shifts)
        # Differences keep the reference exact: a radius converted to external
        # units and back need not round-trip.
        travel = jnp.where(valid, radii - self.reference_radii, jnp.nan) / self.scale
        shifts = self.transverse(self.shifts_or_zero(center_shifts))
        moved = jnp.where(valid, shifts, jnp.nan) / self.scale
        return travel, travel * (2 * self.working_radii + travel), moved

    def volume(
        self,
        radii: Float[Array, " h"],
        center_shifts: Float[Array, "h 3"] | None = None,
    ) -> Float[Array, ""]:
        """Solid volume in input length units cubed."""
        _, squared, _ = self._edit(radii, center_shifts)
        change = -jnp.pi * jnp.sum(self.heights * squared)
        return (self.reference_volume + change) * self.scale**3

    def surface_area(
        self,
        radii: Float[Array, " h"],
        center_shifts: Float[Array, "h 3"] | None = None,
    ) -> Float[Array, ""]:
        """Solid surface area in input length units squared."""
        travel, squared, _ = self._edit(radii, center_shifts)
        change = 2 * jnp.pi * jnp.sum(self.heights * travel - squared)
        return (self.reference_area + change) * self.scale**2

    def center_of_mass(
        self,
        radii: Float[Array, " h"],
        center_shifts: Float[Array, "h 3"] | None = None,
    ) -> Float[Array, "3"]:
        """Uniform-density center of mass in external coordinates."""
        travel, squared, moved = self._edit(radii, center_shifts)
        removed = (self.working_radii + travel) ** 2
        moment = self.reference_moment - jnp.pi * jnp.sum(
            self.heights[:, None] * (squared[:, None] * self.centers)
            + (self.heights * removed)[:, None] * moved,
            axis=0,
        )
        volume = self.reference_volume - jnp.pi * jnp.sum(self.heights * squared)
        return moment / volume * self.scale + self.origin

    def ligament_bounds(
        self,
        radii: Float[Array, " h"],
        center_shifts: Float[Array, "h 3"] | None = None,
    ) -> Float[Array, " h"]:
        """Lower bound on each edited hole's distance to the rest of the boundary.

        A point of an edited wall lies within ``abs(r - r0) + norm(s)`` of the
        wall it replaces, so by the triangle inequality the distance to any
        boundary face other than the hole's own end faces falls by at most that
        travel, and to another selected hole by at most both travels. This holds
        for every recognized hole, whatever the neighboring surfaces are.

        The bound is reached when the hole only grows and the closest feature
        lies radially, and is conservative otherwise: shrinking counts as travel
        because a feature beyond the hole's opening can come closer. It is the
        distance between faces, not a part-wide wall thickness: it says nothing
        about features the selected holes do not face. NaN for invalid edits.
        """
        valid = self.is_valid(radii, center_shifts)
        moved = self.shift_lengths(self.shifts_or_zero(center_shifts))
        travel = jnp.where(
            valid, jnp.abs(radii - self.reference_radii) + moved, jnp.nan
        )
        to_holes = self.hole_clearance - travel[None, :] - travel[:, None]
        return jnp.minimum(
            self.fixed_clearance - travel,
            jnp.min(to_holes, axis=1),
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
    working_shape: TopoDS_Shape = field(repr=False)
    working_holes: tuple[ThroughHole, ...] = field(repr=False)

    def rebuild(
        self,
        radii: Float[Array, " h"],
        center_shifts: Float[Array, "h 3"] | None = None,
    ) -> TopoDS_Shape:
        """Rebuild isolated holes in OCCT and restore external coordinates.

        This host operation is not differentiable. Raises ValueError for an
        invalid radius/shift vector and RuntimeError for reconstruction failure
        or disagreement with isolated-hole volume, area or center of mass. When
        whole-solid OCCT properties disagree, the removed and added regions must
        still match the edit and stay beside the edited bores. Input CAD
        remains owned by the caller; each reconstruction uses an owned copy.
        Only holes whose radius differs from its reference or whose center
        moves are filled and cut again; the others keep their imported faces,
        since a STEP hole wall may deviate from its ideal cylinder within the
        CAD tolerance. Same-domain faces/edges are unified, so unselected
        geometry is preserved but original face partitions and IDs are not.
        """
        values = jnp.asarray(radii)
        shifts = self.model.shifts_or_zero(
            None if center_shifts is None else jnp.asarray(center_shifts)
        )
        if values.shape != self.model.reference_radii.shape or not bool(
            self.model.is_valid(values, shifts)
        ):
            raise ValueError(
                "Invalid radii or center shifts: expected finite bounded values "
                "and transverse shifts"
            )
        radius_values = np.asarray(values, dtype=np.float64)
        transverse = np.asarray(self.model.transverse(shifts))
        # Compare in caller units: converting a reference radius to working
        # units and back need not round-trip exactly.
        edited = np.flatnonzero(
            (radius_values != np.asarray(self.model.reference_radii))
            | np.any(transverse != 0, axis=1)
        )
        working = rebuild_holes(
            self.working_shape,
            tuple(self.working_holes[i] for i in edited),
            radius_values[edited] / self.frame.length_scale,
            transverse[edited] / self.frame.length_scale,
        )
        return self.frame.restore_shape(working)


def _partner_clearances(
    holes: tuple[ThroughHole, ...],
) -> tuple[Float[np.ndarray, " h"], Float[np.ndarray, "h h"]]:
    """Split recognized boundary distances into fixed faces and selected holes."""
    position = {hole.face_id: i for i, hole in enumerate(holes)}
    fixed = np.full(len(holes), np.inf)
    to_holes = np.full((len(holes), len(holes)), np.inf)
    for i, hole in enumerate(holes):
        for face, distance in hole.boundary_distances:
            j = position.get(face)
            if j is None:
                fixed[i] = min(fixed[i], distance)
            else:
                to_holes[i, j] = distance
    return fixed, to_holes


def prepare_hole_edit(
    shape: TopoDS_Shape,
    face_ids: Sequence[int] | None = None,
    *,
    frame: CadFrame | None = None,
    merge_same_domain: bool = False,
) -> PreparedHoleEdit:
    """Prepare bounded edits of full cylindrical through-holes in one solid.

    Accepts a B-Rep returned by ``read_step``; does not assume a stock primitive
    or a construction history. Each selected inward cylinder must be complete,
    with two perpendicular planar ends and sufficient clearance. Blind,
    oblique, intersecting and near-boundary holes are unsupported, as are holes
    with other material resting on an opening's plane. A
    cylindrical wall split into several faces, e.g. a STEP exporter's two
    half-cylinder faces or a coplanar seam splitting it axially, is also
    unsupported by default; see ``merge_same_domain``. Rejected inward
    cylinders are reported; an explicit selection of one raises.

    ``frame`` maps input units to working coordinates; by default the bounding
    box is centered and its largest extent becomes 40 working units. It is
    frozen during editing. Preparation requires JAX x64; it never changes the
    global JAX precision setting. Input CAD is copied. No mesh is built; see
    :func:`~brepax.experimental.applications.hole_mesh.prepare_hole_mesh` for
    responses without an exact law.

    ``merge_same_domain`` unifies coincident same-domain faces and edges of
    the working solid with OCCT ``ShapeUpgrade_UnifySameDomain`` before
    recognition, so a cylindrical wall split as described above can still be
    recognized as one complete hole. It also merges any other same-domain
    faces/edges in the solid, so the input's face partition is not preserved:
    when True, every returned ``face_id`` (selected, rejected, or passed in
    ``face_ids``) refers to the *unified* working solid, not the original
    input. To select specific holes, call ``prepare_hole_edit`` once with
    ``merge_same_domain=True`` and no ``face_ids`` to read their ``face_id``
    from the result, then again with those IDs and the same
    ``merge_same_domain=True``; the unification is deterministic for a given
    input shape, so the IDs line up. ``rebuild`` starts from that unified
    working solid; its result can have a different face partition, so take
    face IDs from the rebuilt solid itself. Default is False: face IDs and
    geometry are unaffected.

    Radius bounds are [0.5*r0, r0 + clearance - margin], with a margin of 30 CAD
    tolerances. Optional transverse center shifts share the growth budget: the
    reach norm(shift) + max(radius-r0, 0) must not exceed clearance - margin,
    and two selected holes must not reach further than their distance less the
    margin between them. Inside this range the laws hold, however thin the
    remaining walls are; ``HoleEditModel.ligament_bounds`` bounds those walls.
    These are local edits, not a topology-changing CAD kernel.

    Raises ValueError for invalid CAD, unsupported selections, or no supported
    holes. OCCT extraction failures propagate. When ``merge_same_domain`` is
    True, a failed or geometry-changing unification also raises ValueError or
    RuntimeError.

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
    working, frame = normalize_shape(shape, frame)
    if merge_same_domain:
        working = unify_same_domain(working)
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
    volume, moment = volume_and_moment(working)
    if volume <= 0:
        raise ValueError("Solid has a non-positive volume")
    measured = GProp_GProps()
    BRepGProp.SurfaceProperties_s(working, measured, 1e-9)
    holes = tuple(external_hole(hole, frame) for hole in selected)
    reference = jnp.array([hole.radius for hole in holes])
    fixed, to_holes = _partner_clearances(holes)
    # The rebuild check's envelope extends 10 tolerances past each bore. Its
    # result is unreliable when that surface lands on another wall, which this
    # avoids, and two envelopes stay apart too instead of being tangent.
    margin = 30 * tolerance * frame.length_scale
    axes = np.array([hole.axis for hole in selected])
    heights = np.array([hole.height for hole in selected])
    model = HoleEditModel(
        reference_volume=jnp.asarray(volume),
        reference_area=jnp.asarray(measured.Mass()),
        reference_moment=jnp.asarray(moment),
        centers=jnp.asarray(
            np.array([hole.base for hole in selected]) + heights[:, None] * axes / 2
        ),
        axes=jnp.asarray(axes),
        heights=jnp.asarray(heights),
        working_radii=jnp.array([hole.radius for hole in selected]),
        reference_radii=reference,
        lower_radii=reference * 0.5,
        upper_radii=reference + jnp.array([hole.clearance for hole in holes]) - margin,
        fixed_clearance=jnp.asarray(fixed),
        hole_clearance=jnp.asarray(to_holes),
        contact_margin=jnp.asarray(margin),
        scale=jnp.asarray(frame.length_scale),
        origin=jnp.asarray(frame.origin),
    )
    return PreparedHoleEdit(model, holes, rejected, frame, working, selected)


__all__ = [
    "HoleEditModel",
    "PreparedHoleEdit",
    "RejectedHole",
    "ThroughHole",
    "prepare_hole_edit",
]
