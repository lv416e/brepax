"""Matched sampled comparators for unions of disjoint-interior box cells.

Method C calls the unchanged repository dispatcher on a fixed pose grid.
Sigmoid and smoothing differentiate their own sampled forward. The default
union field measures distance to exposed boundary rectangles, avoiding the
artificial internal seams of the optional minimum-cell-field ablation.
All resulting gradients are linear direction proxies, not exact one-sided
derivatives of Boolean volume at contact.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from numpy.typing import NDArray

from benchmarks.contact_transition.baselines import RigidPrimitive
from benchmarks.face_contact.baselines import rotation_matrix
from benchmarks.face_contact.cad import IDENTITY_PLACEMENT, Placement
from benchmarks.face_contact.cad import rotation_matrix as host_rotation_matrix
from brepax.boolean.stratum import (
    _boolean_measure_with_dispatch,
    _detect_stratum_generic,
    _make_grid_nd,
)
from brepax.primitives._base import Primitive

from .fixtures import Box, ContactFixture

Representation = Literal["exterior", "cell_min"]


class BoxCellUnion(Primitive):
    """Minimum cell field, retaining internal seams as an explicit ablation.

    Its negative sublevel set agrees with the union away from shared cell
    boundaries. It is not the signed distance inside a touching-cell union.
    Floating masks keep Method C's existing full-PyTree VJP well defined.
    """

    lower: Float[Array, "C 3"]
    upper: Float[Array, "C 3"]
    active: Float[Array, " C"]

    def sdf(self, points: Float[Array, "... 3"]) -> Float[Array, "..."]:
        centers = (self.lower + self.upper) / 2
        extents = (self.upper - self.lower) / 2
        distance = jnp.abs(points[..., None, :] - centers) - extents
        positive = jnp.maximum(distance, 0.0)
        # Match Box's finite-norm regularization in fixed working units.
        outside = jnp.sqrt(jnp.sum(positive**2, axis=-1) + 1e-20)
        outside = jnp.where(jnp.max(distance, axis=-1) > 0, outside, 0.0)
        field = outside + jnp.minimum(jnp.max(distance, axis=-1), 0.0)
        return jnp.min(jnp.where(self.active > 0, field, jnp.inf), axis=-1)

    def volume(self) -> Float[Array, ""]:
        """Exact volume for the required disjoint-interior component cells."""
        return jnp.sum(jnp.prod(self.upper - self.lower, axis=-1) * self.active)

    def parameters(self) -> dict[str, Array]:
        return {"lower": self.lower, "upper": self.upper, "active": self.active}


class ExteriorBoxUnion(BoxCellUnion):
    """Union boundary distance, with internal component faces removed.

    Each boundary rectangle has equal lower/upper coordinates on its normal
    axis. The distance is exact for these represented axis-aligned box unions,
    up to floating-point evaluation. Values at distance ties remain nonsmooth.
    """

    boundary_lower: Float[Array, "F 3"]
    boundary_upper: Float[Array, "F 3"]
    boundary_active: Float[Array, " F"]

    def sdf(self, points: Float[Array, "... 3"]) -> Float[Array, "..."]:
        offsets = jnp.maximum(
            jnp.maximum(
                self.boundary_lower - points[..., None, :],
                points[..., None, :] - self.boundary_upper,
            ),
            0.0,
        )
        squared = jnp.sum(offsets**2, axis=-1)
        minimum = jnp.min(
            jnp.where(self.boundary_active > 0, squared, jnp.inf), axis=-1
        )
        distance = jnp.where(
            minimum > 0, jnp.sqrt(jnp.where(minimum > 0, minimum, 1.0)), 0.0
        )
        inside = jnp.any(
            jnp.all(
                (points[..., None, :] >= self.lower)
                & (points[..., None, :] <= self.upper),
                axis=-1,
            )
            & (self.active > 0),
            axis=-1,
        )
        return jnp.where(inside, -distance, distance)


def _box_bounds(cells: Sequence[Box]) -> NDArray[np.float64]:
    bounds = np.asarray([(cell.lower, cell.upper) for cell in cells], dtype=float)
    if (
        bounds.ndim != 3
        or bounds.shape[1:] != (2, 3)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 1] <= bounds[:, 0])
    ):
        raise ValueError("Expected nonempty finite positive-size box cells")
    for index in range(len(cells)):
        widths = np.minimum(bounds[index, 1], bounds[index + 1 :, 1]) - np.maximum(
            bounds[index, 0], bounds[index + 1 :, 0]
        )
        if np.any(np.all(widths > 0, axis=-1)):
            raise ValueError("Sampled union cells must have disjoint interiors")
    return bounds


def exterior_rectangles(
    cells: Sequence[Box],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Partition the box-coordinate grid and merge exposed coplanar rectangles.

    This host computation uses supplied disjoint-interior axis-aligned cells;
    it does not extract a decomposition from arbitrary CAD. Only occupancy
    changes across a coordinate plane contribute boundary rectangles.
    """
    bounds = _box_bounds(cells)
    axes = [np.unique(bounds[:, :, axis]) for axis in range(3)]
    centers = [axis[:-1] / 2 + axis[1:] / 2 for axis in axes]
    if any(
        np.any((center <= axis[:-1]) | (center >= axis[1:]))
        for axis, center in zip(axes, centers, strict=True)
    ):
        raise ValueError("Box coordinate intervals are below midpoint resolution")
    points = np.stack(np.meshgrid(*centers, indexing="ij"), axis=-1)
    occupied = np.any(
        np.all(
            (points[..., None, :] > bounds[:, 0])
            & (points[..., None, :] < bounds[:, 1]),
            axis=-1,
        ),
        axis=-1,
    )
    rectangles = []
    for normal in range(3):
        tangents = [axis for axis in range(3) if axis != normal]
        shape = tuple(len(centers[axis]) for axis in tangents)
        for index, coordinate in enumerate(axes[normal]):
            before = (
                np.zeros(shape, dtype=bool)
                if index == 0
                else np.take(occupied, index - 1, axis=normal)
            )
            after = (
                np.zeros(shape, dtype=bool)
                if index == len(axes[normal]) - 1
                else np.take(occupied, index, axis=normal)
            )
            # Keep opposite orientations separate even though distance itself
            # ignores normals: merging them could fill a non-boundary hole.
            for mask in (before & ~after, after & ~before):
                remaining = mask.copy()
                while np.any(remaining):
                    first, second = map(int, np.argwhere(remaining)[0])
                    end_second = second + 1
                    while end_second < shape[1] and remaining[first, end_second]:
                        end_second += 1
                    end_first = first + 1
                    while end_first < shape[0] and np.all(
                        remaining[end_first, second:end_second]
                    ):
                        end_first += 1
                    remaining[first:end_first, second:end_second] = False
                    lower, upper = np.zeros(3), np.zeros(3)
                    lower[normal] = upper[normal] = coordinate
                    lower[tangents] = (
                        axes[tangents[0]][first],
                        axes[tangents[1]][second],
                    )
                    upper[tangents] = (
                        axes[tangents[0]][end_first],
                        axes[tangents[1]][end_second],
                    )
                    rectangles.append((lower, upper))
    return (
        np.asarray([rectangle[0] for rectangle in rectangles]),
        np.asarray([rectangle[1] for rectangle in rectangles]),
    )


def prepare_union(
    cells: Sequence[Box],
    *,
    representation: Representation = "exterior",
    cell_capacity: int | None = None,
    face_capacity: int | None = None,
) -> BoxCellUnion:
    """Prepare a static-shape sampled primitive from disjoint-interior boxes.

    Capacities permit one compiled signature across the declared population.
    Preparation rejects overlapping cells rather than assigning the volume
    sum to a union with duplicated material. All lengths use working units.
    """
    if representation not in ("exterior", "cell_min"):
        raise ValueError("Unknown union field representation")
    bounds = _box_bounds(cells)
    count = len(cells)
    capacity = count if cell_capacity is None else cell_capacity
    if capacity < count:
        raise ValueError("Cell capacity is smaller than the cell count")
    lower, upper = np.zeros((capacity, 3)), np.ones((capacity, 3))
    active = np.zeros(capacity)
    lower[:count], upper[:count], active[:count] = bounds[:, 0], bounds[:, 1], 1.0
    arguments = (jnp.asarray(lower), jnp.asarray(upper), jnp.asarray(active))
    if representation == "cell_min":
        return BoxCellUnion(*arguments)
    boundary_lower, boundary_upper = exterior_rectangles(cells)
    count = len(boundary_lower)
    capacity = count if face_capacity is None else face_capacity
    if capacity < count:
        raise ValueError("Face capacity is smaller than the boundary count")
    lower, upper, active = (
        np.zeros((capacity, 3)),
        np.zeros((capacity, 3)),
        np.zeros(capacity),
    )
    lower[:count], upper[:count], active[:count] = boundary_lower, boundary_upper, 1.0
    return ExteriorBoxUnion(
        *arguments, jnp.asarray(lower), jnp.asarray(upper), jnp.asarray(active)
    )


@dataclass(frozen=True)
class SamplingCapacity:
    """Fixed cell/face array sizes selected before measuring the population."""

    fixed_cells: int
    moving_cells: int
    fixed_faces: int
    moving_faces: int


def sampling_capacity(cases: Sequence[ContactFixture]) -> SamplingCapacity:
    """Find shared sizes without observing objective values or heldout labels."""
    if not cases:
        raise ValueError("Expected at least one fixture")
    return SamplingCapacity(
        max(len(case.fixed_cells) for case in cases),
        max(len(case.moving_cells) for case in cases),
        max(len(exterior_rectangles(case.fixed_cells)[0]) for case in cases),
        max(len(exterior_rectangles(case.moving_cells)[0]) for case in cases),
    )


class SampledCase(eqx.Module):
    """Prepared geometry and fixed grid envelope; only the pose varies in AD."""

    fixed: BoxCellUnion
    moving: BoxCellUnion
    pivot: Float[Array, "3"]
    rotation: Float[Array, "3 3"]
    shift: Float[Array, "3"]
    lower_domain: Float[Array, "3"]
    upper_domain: Float[Array, "3"]


def prepare_case(
    case: ContactFixture,
    placement: Placement = IDENTITY_PLACEMENT,
    *,
    representation: Representation = "exterior",
    capacity: SamplingCapacity | None = None,
) -> SampledCase:
    """Prepare freshly placed geometry in normalized working-length coordinates.

    Actual rotation is retained. External units are divided by placement.scale
    before sampling; output volume is restored by scale**3 when required.
    The stationary AABB receives a 0.25*length margin on every axis, shared by
    all methods and frozen during pose AD. This encloses the declared twists
    through t=0.04; it is not a generic motion-bound certificate.
    """
    sizes = sampling_capacity((case,)) if capacity is None else capacity
    rotation = host_rotation_matrix(placement.rotation_vector)
    if not np.isfinite(placement.scale) or placement.scale <= 0:
        raise ValueError("Placement scale must be finite and positive")
    shift = np.asarray(placement.origin, dtype=float) / placement.scale
    if shift.shape != (3,) or not np.isfinite(shift).all():
        raise ValueError("Placement origin must contain three finite lengths")
    corners = np.asarray(
        [
            corner
            for cell in (*case.fixed_cells, *case.moving_cells)
            for corner in product(*zip(cell.lower, cell.upper, strict=True))
        ]
    )
    world = corners @ rotation.T + shift
    margin = 0.25 * case.length
    return SampledCase(
        prepare_union(
            case.fixed_cells,
            representation=representation,
            cell_capacity=sizes.fixed_cells,
            face_capacity=sizes.fixed_faces,
        ),
        prepare_union(
            case.moving_cells,
            representation=representation,
            cell_capacity=sizes.moving_cells,
            face_capacity=sizes.moving_faces,
        ),
        jnp.asarray(case.pivot),
        jnp.asarray(rotation),
        jnp.asarray(shift),
        jnp.asarray(world.min(axis=0) - margin),
        jnp.asarray(world.max(axis=0) + margin),
    )


Measure = Callable[[Array, SampledCase, float, float], Array]


@dataclass(frozen=True)
class Estimators:
    """Matched measures with dynamic geometry/temperatures and a fixed grid size."""

    stratum: Measure
    sigmoid: Measure
    smoothing: Measure
    classify: Callable[[Array, SampledCase], Array]
    resolution: int


def estimators(resolution: int) -> Estimators:
    """Use the unchanged Method C and matched sampled smooth integrands.

    Pose q=(translation, rotation vector) acts about the prepared pivot before
    the common placement. Its Jacobian at q=0 matches the supplied rigid twist.
    Dimensionless k and beta multiply geometric-mean grid spacing. This
    factory is called once per resolution, outside timed hot loops.
    """
    if resolution < 2:
        raise ValueError("Resolution must be at least two")

    def pair(q: Array, case: SampledCase) -> tuple[Primitive, Primitive]:
        local_rotation = rotation_matrix(q[3:])
        local_shift = case.pivot - local_rotation @ case.pivot + q[:3]
        return (
            RigidPrimitive(case.fixed, case.rotation, case.shift),
            RigidPrimitive(
                case.moving,
                case.rotation @ local_rotation,
                case.shift + case.rotation @ local_shift,
            ),
        )

    def stratum(
        q: Array, case: SampledCase, unused_k: float, unused_beta: float
    ) -> Array:
        return _boolean_measure_with_dispatch(
            *pair(q, case),
            "intersect",
            resolution,
            case.lower_domain,
            case.upper_domain,
        )

    def operands(q: Array, case: SampledCase) -> tuple[Array, Array, Array, Array]:
        grid, cell = _make_grid_nd(case.lower_domain, case.upper_domain, resolution)
        first, second = pair(q, case)
        return first.sdf(grid), second.sdf(grid), cell, cell ** (1 / 3)

    def sigmoid(q: Array, case: SampledCase, unused_k: float, beta: float) -> Array:
        first, second, cell, width = operands(q, case)
        return (
            jnp.sum(jax.nn.sigmoid(-jnp.maximum(first, second) / (width * beta))) * cell
        )

    def smoothing(q: Array, case: SampledCase, k: float, beta: float) -> Array:
        first, second, cell, width = operands(q, case)
        sdf = width * k * jnp.logaddexp(first / (width * k), second / (width * k))
        return jnp.sum(jax.nn.sigmoid(-sdf / (width * beta))) * cell

    def classify(q: Array, case: SampledCase) -> Array:
        grid, _ = _make_grid_nd(case.lower_domain, case.upper_domain, resolution)
        return _detect_stratum_generic(*pair(q, case), grid)

    return Estimators(stratum, sigmoid, smoothing, classify, resolution)
