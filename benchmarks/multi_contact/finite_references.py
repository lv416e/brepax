"""Independent finite-motion controls for event-delimited volume profiles.

The translating controls use products of interval intersections, independently
of polyhedral vertex enumeration. A common linear map scales those volumes by
its absolute determinant. Rotation uses world-space polygon clipping as a
separate reference to a bounded closed-form wall/prism calculation.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray

from .fixtures import Box, Vector3

Matrix3 = tuple[Vector3, Vector3, Vector3]
IDENTITY: Matrix3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


@dataclass(frozen=True)
class TranslationCase:
    """Supplied disjoint-interior box unions and their common linear map."""

    name: str
    fixed: tuple[Box, ...]
    moving: tuple[Box, ...]
    velocity: Vector3
    max_step: float
    steps: tuple[float, ...]
    # (event step, left slope, right slope), in the original box frame.
    slope_events: tuple[tuple[float, float, float], ...]
    matrix: Matrix3 = IDENTITY


def box_vertices(box: Box, matrix: Matrix3 = IDENTITY) -> NDArray[np.float64]:
    """Eight exact supplied corners followed by the declared linear map."""
    points = np.asarray(list(product(*zip(box.lower, box.upper, strict=True))))
    return np.asarray(points @ np.asarray(matrix).T, dtype=np.float64)


def translation_cases() -> tuple[TranslationCase, ...]:
    """Contact order, onset, exit, simultaneous contacts, and oblique controls."""
    cube = Box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    wall = Box((-1.0, 0.0, 0.0), (0.0, 1.0, 1.0))
    samples = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.75, 2.0)
    face = TranslationCase(
        "face",
        (wall,),
        (cube,),
        (-1.0, 0.0, 0.0),
        2.0,
        samples,
        ((1.0, 1.0, -1.0),),
    )
    return (
        face,
        TranslationCase(
            "edge",
            (Box((-1.0, -1.0, 0.0), (0.0, 0.0, 1.0)),),
            (cube,),
            (-1.0, -1.0, 0.0),
            2.0,
            samples,
            ((1.0, 2.0, -2.0),),
        ),
        TranslationCase(
            "point",
            (Box((-1.0, -1.0, -1.0), (0.0, 0.0, 0.0)),),
            (cube,),
            (-1.0, -1.0, -1.0),
            2.0,
            samples,
            ((1.0, 3.0, -3.0),),
        ),
        TranslationCase(
            "sliding_face",
            (wall,),
            (cube,),
            (-1.0, 0.5, 0.25),
            2.0,
            samples,
            ((1.0, -0.125, -0.875),),
        ),
        TranslationCase(
            "positive_gap",
            (wall,),
            (Box((0.25, 0.0, 0.0), (1.25, 1.0, 1.0)),),
            (-1.0, 0.0, 0.0),
            2.5,
            (0.0, 0.125, 0.25, 0.5, 1.0, 1.25, 1.5, 2.25, 2.5),
            ((0.25, 0.0, 1.0), (1.25, 1.0, -1.0), (2.25, -1.0, 0.0)),
        ),
        TranslationCase(
            "corner",
            (
                Box((-1.0, -1.0, 0.0), (0.0, 1.0, 1.0)),
                Box((0.0, -1.0, 0.0), (1.0, 0.0, 1.0)),
            ),
            (cube,),
            (-1.0, -1.0, 0.0),
            2.0,
            samples,
            ((1.0, 0.0, -2.0),),
        ),
        TranslationCase(
            "thin_enter_exit",
            (Box((0.0, 0.0, 0.0), (0.125, 1.0, 1.0)),),
            (Box((0.25, 0.0, 0.0), (0.375, 1.0, 1.0)),),
            (-1.0, 0.0, 0.0),
            0.5,
            (0.0, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5),
            ((0.125, 0.0, 1.0), (0.25, 1.0, -1.0), (0.375, -1.0, 0.0)),
        ),
        TranslationCase(
            "oblique_sliding_face",
            (wall,),
            (cube,),
            (-1.0, 0.5, 0.25),
            2.0,
            samples,
            ((1.0, -0.125, -0.875),),
            ((1.0, 0.5, 0.0), (0.0, 1.0, 0.25), (0.0, 0.0, 1.0)),
        ),
    )


def interval_volume(case: TranslationCase, step: float) -> float:
    """Independent finite overlap for the original disjoint box partition."""
    shift = step * np.asarray(case.velocity)
    volume = 0.0
    for fixed, moving in product(case.fixed, case.moving):
        lower = np.maximum(fixed.lower, np.asarray(moving.lower) + shift)
        upper = np.minimum(fixed.upper, np.asarray(moving.upper) + shift)
        volume += float(np.prod(np.maximum(upper - lower, 0.0)))
    return volume * abs(float(np.linalg.det(np.asarray(case.matrix))))


def interval_slope(case: TranslationCase, step: float) -> float:
    """Derivative from interval endpoints, only away from a switching event."""
    velocity = np.asarray(case.velocity)
    result = 0.0
    for fixed, moving in product(case.fixed, case.moving):
        moved_lower = np.asarray(moving.lower) + step * velocity
        moved_upper = np.asarray(moving.upper) + step * velocity
        lower = np.maximum(fixed.lower, moved_lower)
        upper = np.minimum(fixed.upper, moved_upper)
        widths = upper - lower
        if np.any(widths <= 0):
            continue
        rates = np.where(moved_upper < fixed.upper, velocity, 0.0)
        rates -= np.where(moved_lower > fixed.lower, velocity, 0.0)
        result += sum(float(rates[k] * np.prod(np.delete(widths, k))) for k in range(3))
    return result * abs(float(np.linalg.det(np.asarray(case.matrix))))


def rotation_volume(
    step: float, width: float, upper: float, height: float, pivot_x: float
) -> float:
    """Wall overlap for the specified uncapped or zero-pivot capped chamber.

    The moving prism is [0,width] x [-upper,upper] x [0,height]. Its path is
    p + R_z(t)(x-p) + t v with p=(pivot_x,0,0), v=(0,pivot_x,0).
    These paths share their initial velocity field but not their second terms.
    The finite wall is assumed to contain the returned overlap completely.
    """
    if not 0.0 < step < np.pi / 2 or min(width, upper, height) <= 0 or pivot_x < 0:
        raise ValueError("Expected a positive small-angle prism configuration")
    sine, cosine = np.sin(step), np.cos(step)
    # 2 sin(t/2)^2 avoids losing 1-cos(t) at small angles.
    numerator = upper * sine - pivot_x * 2.0 * np.sin(step / 2.0) ** 2
    depth = numerator / cosine
    cutoff = pivot_x * np.tan(step / 2.0)
    if cutoff >= upper or depth <= 0:
        raise ValueError("No positive triangular chamber in this configuration")
    if depth <= width:
        return float(height * numerator**2 / (2.0 * sine * cosine))
    if pivot_x != 0:
        raise ValueError("Capped formula is restricted to zero pivot offset")
    return float(height * (width * upper - width**2 / (2.0 * np.tan(step))))


def clipped_rotation_volume(
    step: float, width: float, upper: float, height: float, pivot_x: float
) -> float:
    """Independent transformed-polygon clipping against the world wall x<=0."""
    points = np.array([[0.0, -upper], [width, -upper], [width, upper], [0.0, upper]])
    cosine, sine = np.cos(step), np.sin(step)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    pivot = np.array([pivot_x, 0.0])
    shift = pivot - rotation @ pivot + step * np.array([0.0, pivot_x])
    points = points @ rotation.T + shift
    clipped = []
    previous = points[-1]
    for current in points:
        previous_inside, current_inside = previous[0] <= 0, current[0] <= 0
        if previous_inside != current_inside:
            fraction = -previous[0] / (current[0] - previous[0])
            clipped.append(previous + fraction * (current - previous))
        if current_inside:
            clipped.append(current)
        previous = current
    if len(clipped) < 3:
        return 0.0
    polygon = np.asarray(clipped)
    area = 0.5 * abs(
        float(
            np.sum(
                polygon[:, 0] * np.roll(polygon[:, 1], -1)
                - polygon[:, 1] * np.roll(polygon[:, 0], -1)
            )
        )
    )
    return area * height
