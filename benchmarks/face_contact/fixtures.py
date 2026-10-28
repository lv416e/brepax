"""Deterministic convex-prism inputs for grouped face-contact experiments.

Coordinates are local working lengths. The fixed stock occupies z in [-1, 0]
and the moving prism starts at z=gap; its CCW footprint lies strictly inside
[-2, 2]^2. The rigid path is p + exp(t [omega]x) (x-p) + t v with dimensionless
t. Common placements transform geometry, pivots and directions together.

Each base polygon is one split group: all directions, placements and numerical
settings remain together. Five repeated splits measure fixture robustness;
parameter-free methods are not fitted or evidence of learned generalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]
Family = Literal["rectangle", "triangle", "trapezoid", "pentagon", "narrow_hexagon"]

FAMILIES: tuple[Family, ...] = (
    "rectangle",
    "triangle",
    "trapezoid",
    "pentagon",
    "narrow_hexagon",
)
FIXTURE_SEED = 20261028
SPLIT_SEEDS = (2718, 31415, 161803, 141421, 173205)
ACTION_STEP = 0.02
CAD_PROBE_STEPS = (0.01, 0.005, 0.0025)
SAME_FORWARD_FD_STEPS = (0.001, 0.0001, 0.00001)
MAIN_RESOLUTION = 24
CONTROL_RESOLUTIONS = (48, 49)
WARM_REPETITIONS = 5
TEMPERATURES = ((0.25, 0.25), (0.5, 0.5), (1.0, 1.0), (0.25, 0.5), (0.5, 0.25))
TILT_CANDIDATE_COUNT = 8


@dataclass(frozen=True)
class PolygonFixture:
    """CCW polygon in local XY, with positive prism height and stock depth."""

    group_id: str
    family: Family
    variant: int
    vertices: tuple[Vector2, ...]
    gap: float = 0.0
    height: float = 1.0
    stock_bounds: tuple[float, float, float, float] = (-2.0, 2.0, -2.0, 2.0)
    stock_depth: float = 1.0


@dataclass(frozen=True)
class Placement:
    """World placement x_world = origin + scale * rotation @ x_local."""

    name: str
    scale: float
    rotation: tuple[Vector3, Vector3, Vector3]
    origin: Vector3


@dataclass(frozen=True)
class Direction:
    """Translation in working lengths and angular velocity in radians per t."""

    name: str
    translation: Vector3
    angular: Vector3
    pivot: Vector3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class GroupSplit:
    """Fifteen calibration and ten heldout groups, stratified by family."""

    seed: int
    calibration_groups: tuple[str, ...]
    heldout_groups: tuple[str, ...]


def _rotation(axis: Vector3, angle: float) -> NDArray[np.float64]:
    unit = np.asarray(axis, dtype=float)
    unit /= np.linalg.norm(unit)
    x, y, z = unit
    skew = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)), dtype=float)
    return np.asarray(
        np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew),
        dtype=np.float64,
    )


def _tuple3(values: NDArray[np.float64]) -> Vector3:
    return float(values[0]), float(values[1]), float(values[2])


def _matrix_tuple(matrix: NDArray[np.float64]) -> tuple[Vector3, Vector3, Vector3]:
    return _tuple3(matrix[0]), _tuple3(matrix[1]), _tuple3(matrix[2])


def placements() -> tuple[Placement, ...]:
    """Moderate units; geometry, pivots and twists transform together."""
    return (
        Placement("identity", 1.0, _matrix_tuple(np.eye(3)), (0.0, 0.0, 0.0)),
        Placement(
            "small_rigid",
            0.1,
            _matrix_tuple(_rotation((1.0, 2.0, 3.0), 0.7)),
            (0.3, -0.2, 0.4),
        ),
        Placement(
            "large_rigid",
            10.0,
            _matrix_tuple(_rotation((2.0, -1.0, 1.0), -0.4)),
            (30.0, -20.0, 40.0),
        ),
    )


def transform_direction(direction: Direction, placement: Placement) -> Direction:
    """v'=s R v, omega'=R omega, p'=origin+s R p for the same dimensionless t."""
    matrix = np.asarray(placement.rotation)
    return Direction(
        direction.name,
        _tuple3(placement.scale * matrix @ np.asarray(direction.translation)),
        _tuple3(matrix @ np.asarray(direction.angular)),
        _tuple3(
            np.asarray(placement.origin)
            + placement.scale * matrix @ np.asarray(direction.pivot)
        ),
    )


def polygon_area_centroid(
    vertices: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """Return area and centroid of ordered CCW vertices of shape (n, 2).

    Nonpositive signed area raises ValueError. Callers supply finite convex
    inputs; this geometric moment calculation does not validate convexity.
    """
    following = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1]
    area = float(np.sum(cross) / 2)
    if area <= 0:
        raise ValueError("A strictly positive CCW area is required")
    return area, np.sum((vertices + following) * cross[:, None], axis=0) / (6 * area)


def fixtures() -> tuple[PolygonFixture, ...]:
    """Five deterministic affine variants per convex base-polygon family."""
    angles5 = 0.3 + np.arange(5) * 2 * np.pi / 5
    angles6 = np.arange(6) * 2 * np.pi / 6
    prototypes = (
        np.array(((-1.0, -0.7), (1.0, -0.7), (1.0, 0.7), (-1.0, 0.7))),
        np.array(((-1.0, -0.8), (1.0, -0.65), (0.15, 1.0))),
        np.array(((-1.0, -0.7), (1.0, -0.7), (0.55, 0.8), (-0.65, 0.8))),
        np.column_stack((np.cos(angles5), np.sin(angles5))),
        np.column_stack((np.cos(angles6), 0.22 * np.sin(angles6))),
    )
    result = []
    for family_index, (family, prototype) in enumerate(
        zip(FAMILIES, prototypes, strict=True)
    ):
        random = np.random.default_rng(
            np.random.SeedSequence((FIXTURE_SEED, family_index))
        )
        for variant in range(5):
            sx, sy = random.uniform(0.8, 1.1, size=2)
            shear, angle = random.uniform(-0.2, 0.2), random.uniform(-0.55, 0.55)
            affine = np.array(((sx, shear), (0.0, sy)))
            cosine, sine = np.cos(angle), np.sin(angle)
            rotation = np.array(((cosine, -sine), (sine, cosine)))
            points = prototype @ affine.T @ rotation.T
            _, centroid = polygon_area_centroid(points)
            points -= centroid
            points *= min(1.0, 1.35 / float(np.max(np.abs(points))))
            result.append(
                PolygonFixture(
                    f"{family}_{variant:02d}",
                    family,
                    variant,
                    tuple((float(x), float(y)) for x, y in points),
                )
            )
    return tuple(result)


def audit_directions() -> tuple[Direction, ...]:
    """Signed coordinate directions plus four mixed twists; none is discarded."""
    result = []
    for component, label in enumerate(("vx", "vy", "vz", "wx", "wy", "wz")):
        for sign in (-1.0, 1.0):
            components = np.zeros(6)
            components[component] = sign
            result.append(
                Direction(
                    f"{label}_{'positive' if sign > 0 else 'negative'}",
                    _tuple3(components[:3]),
                    _tuple3(components[3:]),
                )
            )
    for index, (translation, angular) in enumerate(
        (
            ((0.2, -0.1, -0.15), (0.6, 0.0, 0.1)),
            ((-0.1, 0.3, 0.2), (-0.4, 0.5, -0.2)),
            ((0.0, -0.2, -0.3), (0.4, 0.5, 0.3)),
            ((0.1, 0.1, 0.1), (-0.5, -0.3, 0.2)),
        )
    ):
        result.append(Direction(f"mixed_{index}", translation, angular))
    return tuple(result)


def selection_directions() -> tuple[Direction, ...]:
    """Eight tilt choices followed by four unconstrained control motions.

    Tilts share v and |omega_xy|. The full panel retains normal approach,
    separation, tangential motion and spin; common success is a valid result.
    Each candidate has |v|<=1 and |omega|<=1 in local units. Equal pointwise
    speeds are not implied by these separate component bounds.
    """
    tilted = tuple(
        Direction(
            f"tilt_{index}",
            (0.2, -0.1, -0.15),
            (float(0.6 * np.cos(angle)), float(0.6 * np.sin(angle)), 0.1),
        )
        for index, angle in enumerate(np.arange(8) * np.pi / 4)
    )
    return (
        *tilted,
        Direction("normal_approach", (0.0, 0.0, -1.0), (0.0, 0.0, 0.0)),
        Direction("normal_separation", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
        Direction("tangential", (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        Direction("normal_spin", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )


def grouped_splits(population: tuple[PolygonFixture, ...]) -> tuple[GroupSplit, ...]:
    """Five repeated stratified splits; no direction or placement leaks groups."""
    result = []
    for seed in SPLIT_SEEDS:
        random = np.random.default_rng(seed)
        calibration, heldout = [], []
        for family in FAMILIES:
            groups = sorted(
                item.group_id for item in population if item.family == family
            )
            if len(groups) != 5:
                raise ValueError(
                    "Each family requires exactly five base-polygon groups"
                )
            shuffled = random.permutation(groups).tolist()
            calibration.extend(shuffled[:3])
            heldout.extend(shuffled[3:])
        result.append(
            GroupSplit(seed, tuple(sorted(calibration)), tuple(sorted(heldout)))
        )
    return tuple(result)
