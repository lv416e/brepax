"""Independent host-side references for the bounded contact experiments.

These routines do not import BRepAX or differentiate the candidate kernel.
Quadrature errors are QUADPACK estimates, not certified roundoff bounds.
All lengths share one unit; areas, volumes, and derivatives retain their
corresponding physical units. SciPy is an existing benchmark dependency.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

# SciPy is an existing optional benchmark dependency without bundled stubs.
from scipy.integrate import quad  # type: ignore[import-untyped]


@dataclass(frozen=True)
class ContactReference:
    """Volume and penetration derivative; unequal one-sided slopes mean a kink."""

    value: float
    derivative: float | None
    left_derivative: float
    right_derivative: float
    quadrature_error: float = 0.0


@dataclass(frozen=True)
class AreaReference:
    """Intersection area and estimated absolute quadrature error, in L²."""

    value: float
    quadrature_error: float


@dataclass(frozen=True)
class RadiusReference:
    """A measure and its classical radius derivative in matching length units."""

    value: float
    derivative: float


@dataclass(frozen=True)
class RadiusDifference:
    """Area/radius differences in L, with the actual physical step in L."""

    step: float
    central: float
    left: float
    right: float


def contact_reference(
    kind: Literal["sphere", "cylinder", "box"], penetration: float
) -> ContactReference:
    """Equal-unit-body intersection V and dV/dδ for δ = 2 - separation.

    The cylinders have parallel axes and fixed common height 2; their endcaps
    do not cross. Boxes have half-extents (1, 1, 1) and meet face to face.
    Only δ in [-0.5, 0.5] is supported, excluding coincidence and other events.
    At box contact no classical derivative exists: its left/right slopes are
    0 and 4. Sphere/cylinder contact has classical derivative 0.
    """
    if kind not in ("sphere", "cylinder", "box"):
        raise ValueError(f"Unknown contact kind: {kind}")
    delta = float(penetration)
    if not math.isfinite(delta) or not -0.5 <= delta <= 0.5:
        raise ValueError("Penetration must be finite and in [-0.5, 0.5]")
    if kind == "box" and delta == 0:
        return ContactReference(0.0, None, 0.0, 4.0)
    if delta <= 0:
        return ContactReference(0.0, 0.0, 0.0, 0.0)
    error = 0.0
    if kind == "sphere":
        value = math.pi * delta**2 * (0.5 - delta / 12)
        derivative = math.pi * delta * (1 - delta / 4)
    elif kind == "cylinder":
        # Integrate the disk lens derivative from contact. Substitution s=δt²
        # removes its square-root endpoint without subtracting two large arcs.
        integral, error = quad(
            lambda t: 2 * t * t * math.sqrt(4 - delta * t * t),
            0,
            1,
            epsabs=2e-13,
            epsrel=2e-13,
        )
        value = 2 * delta**1.5 * integral
        error *= 2 * delta**1.5
        derivative = 2 * math.sqrt(delta * (4 - delta))
    else:
        value, derivative = 4 * delta, 4.0
    return ContactReference(value, derivative, derivative, derivative, error)


def circle_rectangle_area(
    center: Sequence[float], radius: float, half_extents: Sequence[float]
) -> AreaReference:
    """Integrate vertical circle chords clipped to a centered rectangle.

    ``center`` and ``half_extents`` contain two lengths; radius and both
    half-extents must be positive and finite. The change x=cx+r*sin(t) removes
    circle endpoint singularities. Integration is split where a horizontal
    rectangle edge meets the circle. This is a section-integral oracle, not a
    Green boundary formula. Requested quadrature tolerance is 2e-13*r² per
    piece; callers should assess the returned error and their own scale.

    Example: center=(2, 0), radius=1, half_extents=(2, 3) returns pi/2.
    """
    if len(center) != 2 or len(half_extents) != 2:
        raise ValueError("Center and half-extents must each contain two lengths")
    cx, cy = map(float, center)
    hx, hy = map(float, half_extents)
    r = float(radius)
    if not all(math.isfinite(x) for x in (cx, cy, hx, hy, r)):
        raise ValueError("Geometry must be finite")
    if min(hx, hy, r) <= 0:
        raise ValueError("Radius and half-extents must be positive")
    # Normalize by radius, so quadrature tolerances co-scale with model units.
    lo, hi = max(-1.0, (-hx - cx) / r), min(1.0, (hx - cx) / r)
    if lo >= hi:
        return AreaReference(0.0, 0.0)
    lower, upper = (-hy - cy) / r, (hy - cy) / r
    cuts = [lo, hi]
    for y in (lower, upper):
        if abs(y) < 1:
            crossing = math.sqrt((1 - y) * (1 + y))
            cuts.extend(x for x in (-crossing, crossing) if lo < x < hi)
    angles = [math.asin(x) for x in sorted(set(cuts))]

    def chord_integrand(angle: float) -> float:
        half_chord = math.cos(angle)
        clipped_chord = max(0.0, min(upper, half_chord) - max(lower, -half_chord))
        return clipped_chord * half_chord

    values, errors = [], []
    for start, end in pairwise(angles):
        value, error = quad(chord_integrand, start, end, epsabs=2e-13, epsrel=2e-13)
        values.append(value)
        errors.append(error)
    return AreaReference(r * r * math.fsum(values), r * r * math.fsum(errors))


def radius_finite_differences(
    center: Sequence[float],
    radius: float,
    half_extents: Sequence[float],
    relative_steps: Sequence[float] = (1e-3, 1e-4, 1e-5),
) -> tuple[RadiusDifference, ...]:
    """Retain central and both one-sided area/radius differences for every step.

    Physical step is ``relative_step * radius``. At a C¹ contact the errors
    can converge only as sqrt(step), so a small step is not itself evidence
    of an accurate derivative. No best step or derivative is silently chosen.
    """
    base = circle_rectangle_area(center, radius, half_extents).value
    differences = []
    for relative in relative_steps:
        if not math.isfinite(relative) or not 0 < relative < 1:
            raise ValueError("Relative finite-difference steps must lie in (0, 1)")
        step = float(relative * radius)
        if radius + step == radius or radius - step == radius:
            raise ValueError(
                "Finite-difference step is below floating-point resolution"
            )
        plus = circle_rectangle_area(center, radius + step, half_extents).value
        minus = circle_rectangle_area(center, radius - step, half_extents).value
        differences.append(
            RadiusDifference(
                step,
                (plus - minus) / (2 * step),
                (base - minus) / step,
                (plus - base) / step,
            )
        )
    return tuple(differences)


def single_edge_circle_reference(
    radius: float, edge_distance: float
) -> RadiusReference:
    """Circle area inside one half-plane, with center inside that half-plane.

    ``edge_distance >= 0`` measures center-to-line distance. This formula is
    valid for a rectangle only while no other edge clips the disk. Its radius
    derivative is classical and continuous at first tangency, r=distance.
    """
    r, distance = float(radius), float(edge_distance)
    if not all(math.isfinite(x) for x in (r, distance)) or r <= 0 or distance < 0:
        raise ValueError("Use finite positive radius and nonnegative edge distance")
    if r <= distance:
        return RadiusReference(math.pi * r * r, 2 * math.pi * r)
    chord_half = math.sqrt((r - distance) * (r + distance))
    angle = math.atan2(chord_half, distance)
    area = r * r * (math.pi - angle) + distance * chord_half
    return RadiusReference(area, 2 * r * (math.pi - angle))


def breakout_volume_reference(radius: float) -> RadiusReference:
    """Remaining plate volume and dV/dr for the approved radius-driven fixture.

    Half-extents=(3, 2), disk center=(1.5, 0.2), extrusion height=2. For
    0 < radius <= 1.8 only the right edge clips the disk (the top edge is
    tangent at 1.8). The remaining volume is 2*(24-intersection_area), so
    its radius derivative is negative. The derivative exists at 1.5 and 1.8.
    """
    if not math.isfinite(radius) or not 0 < radius <= 1.8:
        raise ValueError("The single-edge breakout oracle requires 0 < radius <= 1.8")
    area = single_edge_circle_reference(radius, 1.5)
    return RadiusReference(2 * (24 - area.value), -2 * area.derivative)
