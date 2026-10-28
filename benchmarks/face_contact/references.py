"""Independent float64 section integrals for exact planar face contact.

These host references neither import BRepAX nor clip/triangulate the positive
polygon. Vertical integration is exact for an affine field; two-point Gauss
quadrature integrates each resulting quadratic piece. Roundoff remains and
the result is not a certified error bound. All lengths share one unit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _polygon(vertices: ArrayLike) -> NDArray[np.float64]:
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("Expected ordered convex vertices with shape (n>=3, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("Polygon coordinates must be finite")
    keep = np.concatenate(([True], np.any(points[1:] != points[:-1], axis=1)))
    points = points[keep]
    if len(points) > 1 and np.array_equal(points[0], points[-1]):
        points = points[:-1]
    signs = []
    for start, end in zip(points, np.roll(points, -1, axis=0), strict=True):
        edge, relative = end - start, points - start
        crosses = edge[0] * relative[:, 1] - edge[1] * relative[:, 0]
        signs.extend(crosses[crosses != 0])
    if signs and min(signs) < 0 < max(signs):
        raise ValueError("Expected a convex polygon in boundary order")
    return points


def _section_integral(
    x: float,
    lower: tuple[float, float],
    upper: tuple[float, float],
    coefficients: tuple[float, float, float],
) -> float:
    a, b, c = coefficients
    low, high = lower[0] + lower[1] * x, upper[0] + upper[1] * x
    intercept = a + b * x
    if c == 0:
        return max(intercept, 0.0) * (high - low)
    root = -intercept / c
    start, end = (max(low, root), high) if c > 0 else (low, min(high, root))
    if start >= end:
        return 0.0
    # Endpoint values avoid subtracting two antiderivatives. At a clipped
    # endpoint its value is exactly zero by construction.
    first = 0.0 if start == root else max(intercept + c * start, 0.0)
    last = 0.0 if end == root else max(intercept + c * end, 0.0)
    return (end - start) * (first + last) / 2


def positive_affine_reference(
    vertices: ArrayLike, coefficients: Sequence[float]
) -> float:
    """Integrate ``max(a+b*x+c*y, 0)`` on an ordered convex polygon.

    Both boundary orientations are accepted. Degenerate zero-area patches
    return zero. The algorithm splits at polygon vertices and where the
    affine zero line crosses a section endpoint; its integrand is quadratic
    on each piece. No absolute cutoff discards small positive regions.

    This float64 benchmark reference targets well-scaled finite inputs.
    Coordinate or coefficient cancellation below float64 resolution and
    overflow/underflow are not certified. For example, a unit square centered
    at the origin with coefficients (0, 1, 0) returns 1.
    """
    points = _polygon(vertices)
    if len(coefficients) != 3:
        raise ValueError("Expected three affine coefficients")
    a, b, c = map(float, coefficients)
    if not all(math.isfinite(value) for value in (a, b, c)):
        raise ValueError("Affine coefficients must be finite")
    if len(points) < 3:
        return 0.0
    speeds = np.array([math.fsum((a, b * x, c * y)) for x, y in points])
    if np.max(speeds) <= 0:
        return 0.0
    # Anchor at the most positive vertex, so a tiny cap stays near zero in
    # local coordinates instead of subtracting almost equal world positions.
    anchor_index = int(np.argmax(speeds))
    origin = points[anchor_index].copy()
    a = float(speeds[anchor_index])
    points = points - origin
    pieces = []
    for left, right in pairwise(sorted(set(points[:, 0]))):
        middle = left + (right - left) / 2
        lines = []
        for start, end in zip(points, np.roll(points, -1, axis=0), strict=True):
            if min(start[0], end[0]) < middle < max(start[0], end[0]):
                slope = (end[1] - start[1]) / (end[0] - start[0])
                lines.append((float(start[1] - slope * start[0]), float(slope)))
        if not lines:
            continue
        if len(lines) != 2:
            raise ValueError("Expected two convex polygon section crossings")
        lower, upper = sorted(lines, key=lambda line: line[0] + line[1] * middle)
        cuts = {float(left), float(right)}
        for intercept, slope in (lower, upper):
            rate = b + c * slope
            if rate != 0:
                root = -(a + c * intercept) / rate
                if left < root < right:
                    cuts.add(root)
        for start, end in pairwise(sorted(cuts)):
            half = (end - start) / 2
            center = start + half
            shift = half / math.sqrt(3)
            pieces.append(
                half
                * math.fsum(
                    _section_integral(x, lower, upper, (a, b, c))
                    for x in (center - shift, center + shift)
                )
            )
    return math.fsum(pieces)


def directional_reference(
    vertices: ArrayLike,
    velocity: Sequence[float],
    angular_velocity: Sequence[float],
    pivot: Sequence[float],
) -> float:
    """One-sided intersection-volume rate at exact opposing contact in z=0.

    The fixed solid lies below the patch and the moving solid above it, with
    positive thickness and no other first-order contact contribution. The
    path is ``pivot + R(t*angular_velocity)*(x-pivot) + t*velocity`` with
    dimensionless t. Linear velocity has length units; angular velocity is in
    radians. Return units are length cubed per unit t.

    This is not a gap/contact detector or a reference for pre-existing volume
    overlap. Opposite unit tilts of a centered square [-1,1]^2 each return 1;
    that nonlinear direction map cannot be an ordinary volume JVP.
    """
    motion = [
        np.asarray(vector, dtype=float)
        for vector in (velocity, angular_velocity, pivot)
    ]
    if any(
        vector.shape != (3,) or not np.all(np.isfinite(vector)) for vector in motion
    ):
        raise ValueError(
            "Velocity, angular velocity, and pivot must be finite 3-vectors"
        )
    v, omega, p = motion
    constant = math.fsum((-v[2], omega[0] * p[1], -omega[1] * p[0]))
    return positive_affine_reference(vertices, (constant, omega[1], -omega[0]))
