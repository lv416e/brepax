"""Section formulas and polygon clipping independent of plane-triple models."""

from dataclasses import dataclass
from itertools import pairwise, product

import numpy as np
from numpy.typing import NDArray

from .parametric_fixtures import ParametricFixture, PrismBodies, SecondParameter


@dataclass(frozen=True)
class ReferenceSolutions:
    """Analytic-family roots and constant solution intervals on the given path."""

    roots: tuple[float, ...]
    intervals: tuple[tuple[float, float], ...]
    maximum: float


def _primitive(kind: str, x: float, a: float) -> float:
    if x <= 0:
        return 0.0
    if kind == "ramp":
        return min(x, a) ** 2 / 2
    if x < a:
        return x * x / 2
    if x < 2 * a:
        return a * a - (2 * a - x) ** 2 / 2
    return a * a


def _section(kind: str, x: float, a: float) -> float:
    if kind == "ramp":
        return x if 0 < x < a else 0.0
    return max(0.0, min(x, 2 * a - x))


def reference_volume(
    fixture: ParametricFixture,
    step: float,
    theta: tuple[float, float] = (0.0, 0.0),
    *,
    second: SecondParameter = "stock_height",
) -> float:
    """Overlap from one-dimensional sections, with no hull/event enumeration."""
    if fixture.family in ("ramp", "tent"):
        gap = fixture.gap + fixture.a * theta[0]
        width = fixture.width + (fixture.a * theta[1] if second == "slab_width" else 0)
        height = fixture.height + (
            fixture.a * theta[1] if second == "stock_height" else 0
        )
        u = step - gap
        return height * (
            _primitive(fixture.family, u, fixture.a)
            - _primitive(fixture.family, u - width, fixture.a)
        )
    if fixture.family == "split":
        lo, hi = step - fixture.gap - fixture.width, step - fixture.gap
        return sum(
            max(0.0, min(start + fixture.a, hi) - max(start, lo))
            for start in (0.0, fixture.separation)
        )
    return fixture.a**3 * max(0.0, 1.0 - abs(step - 1.0)) ** fixture.codimension


def reference_derivatives(
    fixture: ParametricFixture,
    step: float,
    *,
    second: SecondParameter = "stock_height",
) -> tuple[float, NDArray[np.float64]]:
    """Step and dimensionless design derivatives away from section events."""
    if fixture.family not in ("ramp", "tent"):
        raise ValueError("Design derivatives are for ramp/tent families")
    u = step - fixture.gap
    a, h = fixture.a, fixture.height
    slope = h * (
        _section(fixture.family, u, a) - _section(fixture.family, u - fixture.width, a)
    )
    dimension = (
        a * reference_volume(fixture, step) / h
        if second == "stock_height"
        else a * h * _section(fixture.family, u - fixture.width, a)
    )
    return slope, np.array([-a * slope, dimension], dtype=np.float64)


def reference_maximum(fixture: ParametricFixture) -> float:
    """Exact family maximum on the entire supplied path, not a sampled maximum."""
    if fixture.family == "ramp":
        return fixture.height * (fixture.a * fixture.width - fixture.width**2 / 2)
    if fixture.family == "tent":
        return fixture.height * (fixture.a * fixture.width - fixture.width**2 / 4)
    if fixture.family == "split":
        return min(fixture.a, fixture.width)
    return fixture.a**3


def reference_stock_volume(fixture: ParametricFixture) -> float:
    """Exact complete material volume, distinct from attainable path overlap."""
    if fixture.family == "ramp":
        return fixture.height * fixture.a**2 / 2
    if fixture.family == "tent":
        return fixture.height * fixture.a**2
    if fixture.family == "split":
        return 2 * fixture.a
    return fixture.a**3


def reference_knots(fixture: ParametricFixture) -> tuple[float, ...]:
    """Analytic event and stationary positions that delimit monotone sections."""
    if fixture.family == "orthant":
        return 0.0, 1.0, 2.0, fixture.max_step
    positions = {0.0, fixture.max_step}
    if fixture.family == "split":
        for start in (0.0, fixture.separation):
            for boundary in (start, start + fixture.a):
                positions.add(fixture.gap + boundary)
                positions.add(fixture.gap + fixture.width + boundary)
    else:
        boundaries: tuple[float, ...] = (0.0, fixture.a)
        if fixture.family == "tent":
            boundaries = (*boundaries, 2 * fixture.a)
            positions.add(fixture.gap + fixture.a + fixture.width / 2)
        for boundary in boundaries:
            positions.add(fixture.gap + boundary)
            positions.add(fixture.gap + fixture.width + boundary)
    return tuple(sorted(positions))


def reference_solutions(
    fixture: ParametricFixture, target: float
) -> ReferenceSolutions:
    """Invert known monotone analytic sections; retain plateaus and endpoint roots."""
    roots: set[float] = set()
    intervals: list[tuple[float, float]] = []
    for lo, hi in pairwise(reference_knots(fixture)):
        left = reference_volume(fixture, lo) - target
        right = reference_volume(fixture, hi) - target
        if left == 0 and right == 0:
            if intervals and intervals[-1][1] == lo:
                intervals[-1] = intervals[-1][0], hi
            else:
                intervals.append((lo, hi))
        elif left == 0:
            roots.add(lo)
        elif right == 0:
            roots.add(hi)
        elif left * right < 0:
            for _ in range(60):
                midpoint = (lo + hi) / 2
                value = reference_volume(fixture, midpoint) - target
                if value == 0 or midpoint in (lo, hi):
                    lo = hi = midpoint
                    break
                if left * value < 0:
                    hi = midpoint
                else:
                    lo, left = midpoint, value
            roots.add((lo + hi) / 2)
    isolated = tuple(
        root
        for root in sorted(roots)
        if not any(lo <= root <= hi for lo, hi in intervals)
    )
    return ReferenceSolutions(isolated, tuple(intervals), reference_maximum(fixture))


def _cross(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def polygon_intersection_area(
    subject: NDArray[np.float64], clip: NDArray[np.float64]
) -> float:
    """Clip one convex CCW polygon against another's oriented edge halfplanes."""
    points = list(subject)
    for start, end in zip(clip, np.roll(clip, -1, axis=0), strict=True):
        if not points:
            return 0.0
        edge = end - start
        result = []
        previous = points[-1]
        before = _cross(edge, previous - start)
        for point in points:
            after = _cross(edge, point - start)
            if (after >= 0) != (before >= 0):
                result.append(previous + before / (before - after) * (point - previous))
            if after >= 0:
                result.append(point)
            previous, before = point, after
        points = result
    if len(points) < 3:
        return 0.0
    polygon = np.asarray(points)
    return (
        abs(
            float(
                np.sum(
                    polygon[:, 0] * np.roll(polygon[:, 1], -1)
                    - polygon[:, 1] * np.roll(polygon[:, 0], -1)
                )
            )
        )
        / 2
    )


def clipped_volume(bodies: PrismBodies, step: float) -> float:
    """Separate 2D clipping and z intervals, independent of section formulas."""
    shift = step * np.asarray(bodies.velocity)
    volume = 0.0
    for fixed, moving in product(bodies.fixed, bodies.moving):
        height = max(
            0.0,
            min(fixed.upper_z, moving.upper_z + shift[2])
            - max(fixed.lower_z, moving.lower_z + shift[2]),
        )
        volume += height * polygon_intersection_area(
            np.asarray(moving.footprint) + shift[:2], np.asarray(fixed.footprint)
        )
    return volume
