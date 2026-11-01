"""Small affine design tasks and independent section-integral references.

The references are evaluator-only. Online policies receive geometry and task
bounds, never the section formulas, reachable components, or optimal answers.
Parameters and translation steps are dimensionless; volume uses working units
cubed. Each supplied event exchange is at theta=0.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from math import comb, isqrt
from time import perf_counter
from typing import Literal, cast

import numpy as np

from benchmarks.face_contact.cad import boundary_tolerance, cad_volume, make_prism
from brepax._occt.backend import (
    BRepAlgoAPI_Fuse,
    BRepBuilderAPI_Copy,
    BRepCheck_Analyzer,
)
from brepax._occt.types import TopoDS_Shape
from brepax.experimental.polyhedral_parameter import AffinePolyhedralCell

from .event_boundary_cad import CadCallCounts
from .event_boundary_fixtures import AffineBodies, VertexBody
from .parametric_cad import CadScene
from .parametric_fixtures import Prism, PrismBodies, rectangle

Mechanism = Literal["linked_ramp", "linked_box", "two_lane", "ramp"]
Polynomial = tuple[Fraction, ...]
Bracket = tuple[Fraction, Fraction]


@dataclass(frozen=True)
class FeasibleComponent:
    """Closed feasible component with rational enclosures of both endpoints."""

    lower: Bracket
    upper: Bracket


@dataclass(frozen=True)
class OptimalityReference:
    """Independent minimum dimension-change enclosure for a volume band.

    ``components`` enclose the complete feasible dimension set. Exact singleton
    components are retained. Objective bounds include rational root-enclosure
    uncertainty, rather than claiming the midpoint is an exact optimizer.
    """

    status: Literal["origin_feasible", "infeasible", "exact", "bracketed"]
    components: tuple[FeasibleComponent, ...]
    objective: Bracket | None


@dataclass(frozen=True)
class DesignTask:
    """Frozen bounded CAD task; methods below the geometry API are audit-only."""

    name: str
    mechanism: Mechanism
    target: Fraction
    theta0: Fraction
    bounds: Bracket
    extent: float = 3.0
    velocity: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def bodies(self, theta: Fraction | float | int) -> PrismBodies:
        """Construct explicit affine prisms without Boolean observations."""
        parameter = Fraction(theta)
        if not self.bounds[0] <= parameter <= self.bounds[1]:
            raise ValueError("Dimension is outside the requested closed domain")
        value = float(parameter)
        width = 1 + value
        fixed: tuple[Prism, ...]
        moving: tuple[Prism, ...]
        if self.mechanism in ("linked_ramp", "ramp"):
            height = 1 - value / 2 if self.mechanism == "linked_ramp" else 1.0
            fixed = (Prism(((0, 0), (1, 0), (1, 1)), 0, height),)
            moving = (rectangle(-0.25 - width, -0.25, -0.25, 1.25, -0.25, 1.5),)
        elif self.mechanism == "linked_box":
            fixed = (rectangle(0, 1, 0, 1, 0, 1),)
            moving = (
                rectangle(
                    -0.25 - width,
                    -0.25,
                    0.25 + value / 2,
                    0.75 - value / 2,
                    -0.25,
                    1.25,
                ),
            )
        else:
            fixed = (
                rectangle(0, 1, 0, 2, 0, 2),
                rectangle(0, 1, 4, 6, 0, 2),
            )
            left = -1.25 - value / 8
            moving = (
                rectangle(left, -0.25, 0.25, 1.25 + value, 0.25, 1.25 + value),
                rectangle(left, -0.25, 4.25, 5.25 - value, 0.25, 1.25 - value),
            )
        return PrismBodies(fixed, moving, self.velocity, self.extent)

    def vertices(self, theta: Fraction | float | int) -> tuple[VertexBody, VertexBody]:
        """Physical represented vertices for rebuilding a translation profile."""
        bodies = self.bodies(theta)
        return (
            tuple(prism.vertices() for prism in bodies.fixed),
            tuple(prism.vertices() for prism in bodies.moving),
        )

    def affine_cells(self) -> AffineBodies:
        """Exact dyadic affine displacements with fixed facet normals."""
        base, shifted = self.vertices(0), self.vertices(Fraction(1, 8))
        return cast(
            AffineBodies,
            tuple(
                tuple(
                    AffinePolyhedralCell(points, ((changed - points) * 8)[None, ...])
                    for points, changed in zip(body, displaced, strict=True)
                )
                for body, displaced in zip(base, shifted, strict=True)
            ),
        )

    def scene(
        self, theta: Fraction | float | int, counts: CadCallCounts
    ) -> tuple[CadScene, float]:
        """Build CAD with copied-input source fusions and counted attempts."""
        started = perf_counter()
        bodies = self.bodies(theta)
        groups = tuple(
            tuple(
                make_prism(
                    prism.footprint, prism.lower_z, prism.upper_z - prism.lower_z
                )
                for prism in body
            )
            for body in (bodies.fixed, bodies.moving)
        )
        fixed, moving = groups
        fixed_solid, moving_solid = _fuse(fixed, counts), _fuse(moving, counts)
        tolerances = [
            tolerance
            for shape in (*fixed, *moving)
            if (tolerance := boundary_tolerance(shape)) is not None
        ]
        scene = CadScene(
            fixed,
            moving,
            fixed_solid,
            moving_solid,
            np.asarray(self.velocity),
            1.0,
            perf_counter() - started,
            counts.fuse,
            bool(
                BRepCheck_Analyzer(fixed_solid).IsValid()
                and BRepCheck_Analyzer(moving_solid).IsValid()
            ),
            max(tolerances, default=None),
        )
        return scene, cad_volume(fixed_solid)

    def reference_volume(
        self, theta: Fraction | float | int, step: Fraction | float | int
    ) -> Fraction:
        """Independent exact affine-family integral; rounded CAD vertices may differ."""
        parameter, position = Fraction(theta), Fraction(step) - Fraction(1, 4)
        width = 1 + parameter
        if self.mechanism in ("linked_ramp", "ramp"):
            height = (
                1 - parameter / 2 if self.mechanism == "linked_ramp" else Fraction(1)
            )
            return height * (
                _ramp_integral(position) - _ramp_integral(position - width)
            )
        if self.mechanism == "linked_box":
            area = Fraction(1, 2) - parameter
        else:
            width = 1 + parameter / 8
            area = 2 + 2 * parameter**2
        overlap = max(
            Fraction(0), min(Fraction(1), position) - max(Fraction(0), position - width)
        )
        return area * overlap

    def reference_maximum(self, theta: Fraction | float | int) -> Fraction:
        """Independent maximum over the complete prescribed translation path."""
        parameter = Fraction(theta)
        for lower, upper, coefficients in self._maximum_pieces():
            if lower <= parameter <= upper:
                return _evaluate(coefficients, parameter)
        raise ValueError("Dimension is outside the requested closed domain")

    def reference_optimality(
        self,
        volume_tolerance: Fraction,
        theta_tolerance: Fraction = Fraction(1, 10**10),
    ) -> OptimalityReference:
        """Audit the closest band-feasible dimension using rational polynomials.

        All tasks start at zero removed volume and have continuous translation
        profiles, so a positive lower band endpoint is reachable exactly when
        the complete-path maximum reaches it. This is not an online solver.
        """
        if volume_tolerance < 0 or theta_tolerance <= 0:
            raise ValueError(
                "Expected nonnegative volume and positive dimension tolerances"
            )
        lower_target = self.target - volume_tolerance
        if lower_target <= 0:
            component = FeasibleComponent((self.bounds[0],) * 2, (self.bounds[1],) * 2)
            return OptimalityReference(
                "origin_feasible", (component,), (Fraction(0),) * 2
            )
        components: list[FeasibleComponent] = []
        for lower, upper, coefficients in self._maximum_pieces():
            polynomial = (coefficients[0] - lower_target, *coefficients[1:])
            components.extend(
                _nonnegative_components(polynomial, lower, upper, theta_tolerance)
            )
        components = list(_merge_components(components))
        if not components:
            return OptimalityReference("infeasible", (), None)
        if self.reference_maximum(self.theta0) >= lower_target:
            return OptimalityReference(
                "origin_feasible", tuple(components), (Fraction(0),) * 2
            )
        distances = [
            _component_distance(component, self.theta0) for component in components
        ]
        objective = (
            min(pair[0] for pair in distances),
            min(pair[1] for pair in distances),
        )
        return OptimalityReference(
            "exact" if objective[0] == objective[1] else "bracketed",
            tuple(components),
            objective,
        )

    def _maximum_pieces(self) -> tuple[tuple[Fraction, Fraction, Polynomial], ...]:
        negative: Polynomial
        positive: Polynomial
        if self.mechanism == "linked_ramp":
            negative = (
                Fraction(1, 2),
                -Fraction(1, 4),
                -Fraction(1, 2),
                Fraction(1, 4),
            )
            positive = (Fraction(1, 2), -Fraction(1, 4))
        elif self.mechanism == "linked_box":
            negative = (Fraction(1, 2), -Fraction(1, 2), -Fraction(1))
            positive = (Fraction(1, 2), -Fraction(1))
        elif self.mechanism == "two_lane":
            negative = (Fraction(2), Fraction(1, 4), Fraction(2), Fraction(1, 4))
            positive = (Fraction(2), Fraction(0), Fraction(2))
        else:
            negative = (Fraction(1, 2), Fraction(0), -Fraction(1, 2))
            positive = (Fraction(1, 2),)
        return (
            (self.bounds[0], Fraction(0), negative),
            (Fraction(0), self.bounds[1], positive),
        )


def _fuse(cells: tuple[TopoDS_Shape, ...], counts: CadCallCounts) -> TopoDS_Shape:
    result = cells[0]
    for cell in cells[1:]:
        counts.fuse += 1
        operation = BRepAlgoAPI_Fuse(
            BRepBuilderAPI_Copy(result, True, False).Shape(),
            BRepBuilderAPI_Copy(cell, True, False).Shape(),
        )
        if not operation.IsDone() or operation.Shape().IsNull():
            raise RuntimeError("Source fusion did not produce a shape")
        result = operation.Shape()
    return result


def _ramp_integral(position: Fraction) -> Fraction:
    return (
        Fraction(0)
        if position <= 0
        else Fraction(1, 2)
        if position >= 1
        else position**2 / 2
    )


def _evaluate(coefficients: Polynomial, value: Fraction) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(coefficients):
        result = result * value + coefficient
    return result


def _divide_root(coefficients: Polynomial, root: Fraction) -> Polynomial:
    quotient = [Fraction(0)] * (len(coefficients) - 1)
    quotient[-1] = coefficients[-1]
    for index in range(len(quotient) - 1, 0, -1):
        quotient[index - 1] = coefficients[index] + root * quotient[index]
    if coefficients[0] + root * quotient[0] != 0:
        raise ValueError("Reference polynomial has no root at the supplied point")
    return tuple(quotient)


def _rational_stationary_points(coefficients: Polynomial) -> tuple[Fraction, ...]:
    derivative = tuple(
        index * value for index, value in enumerate(coefficients) if index
    )
    if len(derivative) == 2:
        return (-derivative[0] / derivative[1],)
    if len(derivative) != 3:
        return ()
    constant, linear, quadratic = derivative
    discriminant = linear**2 - 4 * quadratic * constant
    if discriminant < 0:
        return ()
    numerator, denominator = (
        isqrt(discriminant.numerator),
        isqrt(discriminant.denominator),
    )
    if (
        numerator**2 != discriminant.numerator
        or denominator**2 != discriminant.denominator
    ):
        return ()
    root = Fraction(numerator, denominator)
    return ((-linear - root) / (2 * quadratic), (-linear + root) / (2 * quadratic))


def _variations(coefficients: Polynomial, lower: Fraction, upper: Fraction) -> int:
    """Exact Bernstein sign variation bounds the number of interior roots."""
    degree = len(coefficients) - 1
    power = tuple(
        sum(
            (
                coefficients[index]
                * comb(index, exponent)
                * lower ** (index - exponent)
                * (upper - lower) ** exponent
                for index in range(exponent, degree + 1)
            ),
            Fraction(0),
        )
        for exponent in range(degree + 1)
    )
    bernstein = tuple(
        sum(
            (
                power[index] * Fraction(comb(position, index), comb(degree, index))
                for index in range(position + 1)
            ),
            Fraction(0),
        )
        for position in range(degree + 1)
    )
    signs = [value > 0 for value in bernstein if value]
    return sum(left != right for left, right in pairwise(signs))


def _roots(
    coefficients: Polynomial, lower: Fraction, upper: Fraction, tolerance: Fraction
) -> tuple[Bracket, ...]:
    # Every repeated root of a rational polynomial of degree at most three is
    # rational. Remove these explicitly so exact singleton feasibility survives.
    polynomial = coefficients
    while len(polynomial) > 1 and polynomial[-1] == 0:
        polynomial = polynomial[:-1]
    if len(polynomial) > 4:
        raise ValueError("Independent reference supports degree at most three")
    roots: list[Bracket] = []
    candidates = (lower, upper, *_rational_stationary_points(polynomial))
    for candidate in candidates:
        if (
            lower <= candidate <= upper
            and len(polynomial) > 1
            and _evaluate(polynomial, candidate) == 0
        ):
            roots.append((candidate, candidate))
            while len(polynomial) > 1 and _evaluate(polynomial, candidate) == 0:
                polynomial = _divide_root(polynomial, candidate)
    pending = [(lower, upper, 0)]
    while pending and len(polynomial) > 1:
        left, right, depth = pending.pop()
        variation = _variations(polynomial, left, right)
        if not variation:
            continue
        if variation == 1 and right - left <= tolerance:
            roots.append((left, right))
            continue
        if depth >= 256:
            raise ValueError("Independent root separation exceeded the reference cap")
        midpoint = (left + right) / 2
        if _evaluate(polynomial, midpoint) == 0:
            roots.append((midpoint, midpoint))
        pending.extend(((left, midpoint, depth + 1), (midpoint, right, depth + 1)))
    return tuple(sorted(set(roots)))


def _nonnegative_components(
    coefficients: Polynomial, lower: Fraction, upper: Fraction, tolerance: Fraction
) -> tuple[FeasibleComponent, ...]:
    roots = _roots(coefficients, lower, upper, tolerance)
    components = [FeasibleComponent(root, root) for root in roots]
    endpoints = sorted({(lower, lower), *roots, (upper, upper)})
    for left, right in pairwise(endpoints):
        if (
            left[1] < right[0]
            and _evaluate(coefficients, (left[1] + right[0]) / 2) >= 0
        ):
            components.append(FeasibleComponent(left, right))
    return _merge_components(components)


def _merge_components(
    components: list[FeasibleComponent],
) -> tuple[FeasibleComponent, ...]:
    merged: list[FeasibleComponent] = []
    for component in sorted(components, key=lambda item: item.lower):
        if merged and merged[-1].upper == component.lower:
            previous = merged.pop()
            merged.append(
                FeasibleComponent(previous.lower, max(previous.upper, component.upper))
            )
        else:
            merged.append(component)
    return tuple(merged)


def _component_distance(component: FeasibleComponent, origin: Fraction) -> Bracket:
    return (
        max(Fraction(0), component.lower[0] - origin, origin - component.upper[1]),
        max(Fraction(0), component.lower[1] - origin, origin - component.upper[0]),
    )


TASKS = (
    DesignTask(
        "linked_ramp_island",
        "linked_ramp",
        Fraction(67, 128),
        Fraction(1, 8),
        (Fraction(-3, 8), Fraction(1, 4)),
    ),
    DesignTask(
        "two_lane_disconnected",
        "two_lane",
        Fraction(257, 128),
        Fraction(-1, 32),
        (Fraction(-1, 4), Fraction(1, 4)),
    ),
    DesignTask(
        "origin_feasible",
        "linked_ramp",
        Fraction(3, 8),
        Fraction(1, 8),
        (Fraction(-3, 8), Fraction(1, 4)),
    ),
    DesignTask(
        "linked_box_unreachable",
        "linked_box",
        Fraction(37, 64),
        Fraction(0),
        (Fraction(-3, 8), Fraction(1, 4)),
    ),
    DesignTask(
        "linked_box_peak_band",
        "linked_box",
        Fraction(9, 16),
        Fraction(0),
        (Fraction(-3, 8), Fraction(1, 4)),
    ),
    DesignTask(
        "ramp_plateau_control",
        "ramp",
        Fraction(1, 2),
        Fraction(-1, 8),
        (Fraction(-1, 4), Fraction(1, 4)),
    ),
)
