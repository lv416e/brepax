"""Preserve supplied local contact geometry under a common world placement.

A placed model refers to the supplied local geometry and its explicit frame.
It does not repair, reclassify, or certify a separately flattened world B-Rep.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from numpy.typing import ArrayLike

from brepax._occt.backend import TopLoc_Location
from brepax._occt.types import TopoDS_Shape
from brepax.experimental.planar_contact import (
    PlanarContactSet,
    planar_contact_set_volume_direction,
)
from brepax.experimental.planar_contact_collection_cad import (
    PreparedPlanarContactSet,
    prepare_planar_contact_set_reference,
)

_Mode = Literal["zero_residual", "projected_nominal"]
_Provenance = Literal["supplied_local_frame", "shared_cad_location"]
_FRAME_ROUNDOFF = 64 * np.finfo(np.float64).eps


class PlacedPlanarContactSet(eqx.Module):
    """Fixed local contact model with ``world = origin + scale * R @ local``.

    Create with :func:`place_planar_contact_set`. ``rotation`` is proper and
    ``length_scale`` gives world length units per local length unit. The model
    keeps local support coordinates instead of rounding every CAD vertex into
    world coordinates. Geometry and placement stay fixed during motion queries;
    differentiating either record is outside the contract.
    """

    local: PlanarContactSet
    rotation: Float[Array, "3 3"]
    origin: Float[Array, "3"]
    length_scale: Float[Array, ""]


@dataclass(frozen=True)
class PreparedPlacedPlanarContactSet:
    """Placed numerical model and the unmodified local preparation evidence.

    All dimensional pair/source diagnostics in ``local_reference`` retain local
    units. ``provenance`` distinguishes explicitly supplied local cells from
    cells whose common top-level CAD Location was retained and removed exactly.
    Neither provenance proves contact of the original CAD surfaces or of a
    separately supplied flattened world-coordinate B-Rep.
    """

    model: PlacedPlanarContactSet
    local_reference: PreparedPlanarContactSet
    provenance: _Provenance
    original_cad_contact_certified: Literal[False] = field(default=False, init=False)


def place_planar_contact_set(
    local: PlanarContactSet,
    *,
    rotation: ArrayLike,
    origin: ArrayLike,
    length_scale: float = 1.0,
) -> PlacedPlanarContactSet:
    """Attach an explicit similarity to an already prepared local contact set.

    Host validation requires x64, finite origin, a proper orthogonal rotation
    within 64 binary64 epsilons, and a positive scale whose square is a finite
    normal binary64 value. The matrix is validated without projection or
    repair. No geometry, pair relation, or source diagnostic is recomputed.
    Input arrays are copied; later host mutation does not change the model.

    A gap or overlap in the supplied local geometry remains that relation.
    Supplying the old construction frame beside a flattened B-Rep does not
    undo its rounded vertex coordinates: prepare the preserved local cells.
    """
    if jnp.asarray(0.0).dtype != np.dtype("float64"):
        raise ValueError("Contact placement requires JAX x64")
    matrix = np.array(rotation, dtype=np.float64, copy=True)
    offset = np.array(origin, dtype=np.float64, copy=True)
    scale = float(length_scale)
    if (
        matrix.shape != (3, 3)
        or offset.shape != (3,)
        or not np.isfinite(matrix).all()
        or not np.isfinite(offset).all()
    ):
        raise ValueError("Expected finite rotation (3, 3) and origin (3,)")
    if not np.allclose(
        matrix.T @ matrix, np.eye(3), rtol=0, atol=_FRAME_ROUNDOFF
    ) or not np.isclose(np.linalg.det(matrix), 1.0, rtol=0, atol=_FRAME_ROUNDOFF):
        raise ValueError("Rotation must be proper and orthogonal at roundoff scale")
    area_scale = scale * scale
    if (
        not np.isfinite(scale)
        or scale <= 0
        or not np.isfinite(area_scale)
        or area_scale < np.finfo(np.float64).tiny
    ):
        raise ValueError("Positive length_scale must have a finite normal square")
    return PlacedPlanarContactSet(
        local, jnp.asarray(matrix), jnp.asarray(offset), jnp.asarray(scale)
    )


def placed_planar_contact_set_volume_direction(
    model: PlacedPlanarContactSet,
    linear_velocity: Float[Array, "3"],
    angular_velocity: Float[Array, "3"],
    pivot: Float[Array, "3"],
) -> Float[Array, ""]:
    """Return one-sided volume rate for a world-coordinate instantaneous motion.

    Linear velocity and pivot use world length units; angular velocity uses
    radians per dimensionless time. The result is world volume per time.
    JIT and vmap over supplied directions are supported. Like the local
    evaluator, this nonlinear response is not an ordinary Boolean-volume JVP.

    The pullback is ``s**3 D(R.T v/s, R.T omega, R.T(p-origin)/s)``.
    Positive homogeneity gives the equivalent area-scaled evaluation below,
    avoiding unnecessary overflow in the cubic multiplier or linear-velocity
    division. Remaining intermediate overflow and unresolvable world-coordinate
    differences are not repaired. Keep the pivot near the geometry; a distant
    origin cannot restore information already rounded out of the supplied pivot.

    Compensating a changed pivot preserves this instantaneous response. It does
    not assert equivalence of finite paths using linearly translated rotations.
    Zero response does not certify a collision-free finite move.
    """
    rotation = model.rotation
    scale = model.length_scale
    local_velocity = rotation.T @ linear_velocity
    local_angular = scale * (rotation.T @ angular_velocity)
    local_pivot = rotation.T @ (pivot - model.origin) / scale
    return scale**2 * planar_contact_set_volume_direction(
        model.local, local_velocity, local_angular, local_pivot
    )


def prepare_placed_planar_contact_set_reference(
    fixed_local_cells: Sequence[TopoDS_Shape],
    moving_local_cells: Sequence[TopoDS_Shape],
    *,
    rotation: ArrayLike,
    origin: ArrayLike,
    length_scale: float = 1.0,
    mode: _Mode,
) -> PreparedPlacedPlanarContactSet:
    """Prepare preserved local CAD cells and retain their explicit shared frame.

    Local cells follow ``prepare_planar_contact_set_reference`` unchanged:
    same-body or initial cross-body overlap is rejected; positive gaps and
    lower-dimensional contacts remain in the inventory. The caller supplies
    the local geometry and the frame; this function does not infer either from
    transformed world geometry. Every finite edit still needs CAD validation.

    Example, with existing local stock/cutter cells and a supplied rotation::

        prepared = prepare_placed_planar_contact_set_reference(
            stock_cells, cutter_cells, rotation=rotation, origin=(2, -1, 3),
            length_scale=0.001, mode="zero_residual",
        )
        rate = placed_planar_contact_set_volume_direction(
            prepared.model, world_velocity, world_angular_velocity, world_pivot,
        )
        # Pair/source diagnostics remain in prepared.local_reference's units.
    """
    prepared = prepare_planar_contact_set_reference(
        fixed_local_cells, moving_local_cells, mode=mode
    )
    return PreparedPlacedPlanarContactSet(
        place_planar_contact_set(
            prepared.contact_set,
            rotation=rotation,
            origin=origin,
            length_scale=length_scale,
        ),
        prepared,
        "supplied_local_frame",
    )


def prepare_located_planar_contact_set_reference(
    fixed_cells: Sequence[TopoDS_Shape],
    moving_cells: Sequence[TopoDS_Shape],
    *,
    mode: _Mode,
) -> PreparedPlacedPlanarContactSet:
    """Use a genuinely shared top-level proper-rigid OCCT Location as provenance.

    Every supplied cell must have the same Location datum chain, as compared
    by OCCT ``IsEqual``. Independently created equal matrices are insufficient.
    Reflections and nonunit scales beyond frame roundoff are rejected; use
    explicit local cells and numeric unit metadata for scaled models. At least
    one non-null cell is required to supply the frame.

    Removing this top-level Location leaves the local TShape, orientation and
    nested geometry unchanged. No inverse transform is baked into vertices.
    The frame is read from the retained Location, never from a guessed placement
    or a geometry fit. This is not a general assembly-frame recovery procedure.

    An identity Location is allowed and does not recover lost provenance: a
    flattened input is classified using its existing coordinates and may reject
    exactly as before. Failures retain the local preparation's pair evidence.
    The returned model does not certify original source-CAD contact.
    """
    fixed, moving = tuple(fixed_cells), tuple(moving_cells)
    shapes = fixed + moving
    if not shapes or any(shape is None or shape.IsNull() for shape in shapes):
        raise ValueError("A shared source Location requires non-null CAD cells")
    location = shapes[0].Location()
    if any(not shape.Location().IsEqual(location) for shape in shapes):
        raise ValueError("All cells must have the same explicit top-level CAD Location")
    transform = location.Transformation()
    scale = float(transform.ScaleFactor())
    if transform.IsNegative() or not np.isclose(
        scale, 1.0, rtol=0, atol=_FRAME_ROUNDOFF
    ):
        raise ValueError(
            "Source CAD Location must be proper rigid without unit scaling"
        )
    matrix = np.array([[transform.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])
    origin = np.array([transform.Value(i, 4) for i in (1, 2, 3)])
    identity = TopLoc_Location()
    prepared = prepare_planar_contact_set_reference(
        tuple(shape.Located(identity) for shape in fixed),
        tuple(shape.Located(identity) for shape in moving),
        mode=mode,
    )
    return PreparedPlacedPlanarContactSet(
        place_planar_contact_set(
            prepared.contact_set,
            rotation=matrix / scale,
            origin=origin,
            length_scale=scale,
        ),
        prepared,
        "shared_cad_location",
    )
