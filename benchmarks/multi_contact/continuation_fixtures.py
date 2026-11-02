"""Stepped and oblique CAD mechanisms with evaluator-only analytic references.

The task inputs contain no event location or continuation radius. Geometry is
affine in one dimensionless parameter, with fixed normals and translation
velocity. Volume uses working length units cubed. Reference methods integrate
independent sections; online policies must not consult them.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from time import perf_counter
from typing import Literal, cast

import numpy as np

from benchmarks.face_contact.cad import boundary_tolerance, cad_volume, make_prism
from brepax._occt.backend import BRepCheck_Analyzer
from brepax.experimental.polyhedral_parameter import AffinePolyhedralCell

from .design_search_fixtures import (
    Bracket,
    FeasibleComponent,
    OptimalityReference,
    Polynomial,
    _component_distance,
    _evaluate,
    _fuse,
    _merge_components,
    _nonnegative_components,
)
from .event_boundary_cad import CadCallCounts
from .event_boundary_fixtures import AffineBodies, VertexBody
from .parametric_cad import CadScene
from .parametric_fixtures import Prism, PrismBodies, rectangle

Scalar = Fraction | float | int
Mechanism = Literal["stepped", "oblique"]


@dataclass(frozen=True)
class ContinuationTask:
    """Closed design search task with an affine origin independent of its bounds.

    Geometry construction permits positive dimensions outside the requested
    search interval so affine origin/displacement construction also works when
    the initial parameter and requested domain are far from zero.
    """

    name: str
    target: Fraction
    theta0: Fraction
    bounds: Bracket
    mechanism: Mechanism = "stepped"
    width_base: Fraction = Fraction(1, 2)
    extent: float = 4.0
    velocity: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def bodies(self, theta: Scalar) -> PrismBodies:
        """Build source prisms; the search runner enforces the requested domain."""
        parameter = Fraction(theta)
        width, height = (
            self.width_base + 3 * parameter / 2,
            Fraction(3, 4) - parameter / 4,
        )
        if width <= 0 or height <= 0:
            raise ValueError("Expected positive cutter width and height")
        fixed: tuple[Prism, ...]
        if self.mechanism == "stepped":
            fixed = (
                rectangle(0, 0.75, 0, 0.5, 0, 1),
                rectangle(0.75, 1.875, 0, 1, 0, 1),
            )
            zmin, zmax = -0.25, 1.25
        else:
            fixed = (Prism(((0, 0), (1.5, 0), (1.5, 1), (1, 1)), 0, 1),)
            zmin, zmax = 0.0, 1.0
        moving = (
            rectangle(
                float(-Fraction(1, 4) - width), -0.25, 0, float(height), zmin, zmax
            ),
        )
        return PrismBodies(fixed, moving, self.velocity, self.extent)

    def vertices(self, theta: Scalar) -> tuple[VertexBody, VertexBody]:
        """Return represented vertices for the exact-profile rebuild baseline."""
        bodies = self.bodies(theta)
        return (
            tuple(prism.vertices() for prism in bodies.fixed),
            tuple(prism.vertices() for prism in bodies.moving),
        )

    def affine_cells(self) -> AffineBodies:
        """Use dyadic source points to recover the fixed affine displacement."""
        base, displaced = self.vertices(0), self.vertices(Fraction(1, 8))
        return cast(
            AffineBodies,
            tuple(
                tuple(
                    AffinePolyhedralCell(points, ((changed - points) * 8)[None, ...])
                    for points, changed in zip(body, shifted, strict=True)
                )
                for body, shifted in zip(base, displaced, strict=True)
            ),
        )

    def scene(self, theta: Scalar, counts: CadCallCounts) -> tuple[CadScene, float]:
        """Build actual CAD sources with copied-input fusions and counted failures."""
        started, previous_fusions = perf_counter(), counts.fuse
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
            counts.fuse - previous_fusions,
            bool(
                BRepCheck_Analyzer(fixed_solid).IsValid()
                and BRepCheck_Analyzer(moving_solid).IsValid()
            ),
            max(tolerances, default=None),
        )
        return scene, cad_volume(fixed_solid)

    def reference_volume(self, theta: Scalar, step: Scalar) -> Fraction:
        """Exact affine-family section integral; CAD vertices can be rounded."""
        parameter, right = Fraction(theta), Fraction(step) - Fraction(1, 4)
        width, height = (
            self.width_base + 3 * parameter / 2,
            Fraction(3, 4) - parameter / 4,
        )
        if width <= 0 or height <= 0:
            raise ValueError("Expected positive cutter width and height")
        if self.mechanism == "stepped":
            return _overlap(0, Fraction(3, 4), right - width, right) * min(
                Fraction(1, 2), height
            ) + _overlap(Fraction(3, 4), Fraction(15, 8), right - width, right) * min(
                Fraction(1), height
            )
        return _oblique_integral(right, height) - _oblique_integral(
            right - width, height
        )

    def reference_maximum(self, theta: Scalar) -> Fraction:
        """Independent complete-path maximum on the task's requested domain."""
        parameter = Fraction(theta)
        for lower, upper, polynomial in self._maximum_pieces():
            if lower <= parameter <= upper:
                return _evaluate(polynomial, parameter)
        raise ValueError("Dimension is outside the requested closed domain")

    def reference_optimality(
        self,
        volume_tolerance: Fraction,
        theta_tolerance: Fraction = Fraction(1, 10**10),
    ) -> OptimalityReference:
        """Enclose the nearest volume-band feasible dimension independently.

        Every path starts at zero volume and is continuous. A positive lower
        band endpoint is therefore reachable iff the complete-path maximum
        reaches it. Root/component operations use existing rational polynomial
        helpers and never the continuation model or its search output.
        """
        if volume_tolerance < 0 or theta_tolerance <= 0:
            raise ValueError(
                "Expected nonnegative volume and positive dimension tolerances"
            )
        target = self.target - volume_tolerance
        if target <= 0:
            component = FeasibleComponent((self.bounds[0],) * 2, (self.bounds[1],) * 2)
            return OptimalityReference(
                "origin_feasible", (component,), (Fraction(0),) * 2
            )
        components: list[FeasibleComponent] = []
        for lower, upper, polynomial in self._maximum_pieces():
            residual = (polynomial[0] - target, *polynomial[1:])
            components.extend(
                _nonnegative_components(residual, lower, upper, theta_tolerance)
            )
        merged = _merge_components(components)
        if not merged:
            return OptimalityReference("infeasible", (), None)
        if self.reference_maximum(self.theta0) >= target:
            return OptimalityReference("origin_feasible", merged, (Fraction(0),) * 2)
        distances = [
            _component_distance(component, self.theta0) for component in merged
        ]
        objective = (
            min(pair[0] for pair in distances),
            min(pair[1] for pair in distances),
        )
        return OptimalityReference(
            "exact" if objective[0] == objective[1] else "bracketed", merged, objective
        )

    def _maximum_pieces(self) -> tuple[tuple[Fraction, Fraction, Polynomial], ...]:
        """Expand independent section maxima, without fitting model observations."""
        polynomials: tuple[Polynomial, ...]
        thresholds: tuple[Fraction, ...]
        if self.mechanism == "stepped":
            if self.width_base != Fraction(1, 2):
                raise ValueError("Stepped maximum reference requires width_base=1/2")
            thresholds = (Fraction(5, 12), Fraction(11, 12), Fraction(1))
            polynomials = (
                (Fraction(3, 8), Fraction(1), -Fraction(3, 8)),
                (Fraction(17, 32), Fraction(15, 32)),
                (Fraction(39, 32), -Fraction(9, 32)),
                (Fraction(45, 32), -Fraction(15, 32)),
            )
        else:
            base = self.width_base
            # The maximum is H*W, then L*H-H^2/2-(L-W)^2/2, then
            # L*H-H^2/2; L=3/2, H=3/4-theta/4, W=base+3*theta/2.
            thresholds = ((3 - 4 * base) / 5, (3 - 2 * base) / 3)
            polynomials = (
                (3 * base / 4, Fraction(9, 8) - base / 4, -Fraction(3, 8)),
                (
                    Fraction(27, 32) - (Fraction(3, 2) - base) ** 2 / 2,
                    Fraction(33, 16) - 3 * base / 2,
                    -Fraction(37, 32),
                ),
                (Fraction(27, 32), -Fraction(3, 16), -Fraction(1, 32)),
            )
        lower, upper = self.bounds
        boundaries = sorted(
            {lower, upper, *(p for p in thresholds if lower < p < upper)}
        )
        return tuple(
            (
                left,
                right,
                polynomials[sum((left + right) / 2 > point for point in thresholds)],
            )
            for left, right in pairwise(boundaries)
        )


def _overlap(lower: Scalar, upper: Scalar, left: Fraction, right: Fraction) -> Fraction:
    return max(Fraction(0), min(Fraction(upper), right) - max(Fraction(lower), left))


def _oblique_integral(position: Fraction, height: Fraction) -> Fraction:
    position = min(Fraction(3, 2), max(Fraction(0), position))
    return position**2 / 2 if position <= height else height * position - height**2 / 2


TASKS = (
    ContinuationTask(
        "stepped_first_event",
        Fraction(19, 32),
        Fraction(0),
        (Fraction(-1, 16), Fraction(5, 8)),
    ),
    ContinuationTask(
        "stepped_second_event",
        Fraction(25, 32),
        Fraction(0),
        (Fraction(-1, 16), Fraction(5, 8)),
    ),
    ContinuationTask(
        "stepped_unreachable",
        Fraction(7, 8),
        Fraction(0),
        (Fraction(-1, 16), Fraction(5, 8)),
    ),
    ContinuationTask(
        "oblique_second_event",
        Fraction(45, 64),
        Fraction(0),
        (Fraction(-1, 16), Fraction(5, 8)),
        mechanism="oblique",
    ),
    ContinuationTask(
        "oblique_simultaneous_control",
        Fraction(5, 8),
        Fraction(-1, 16),
        (Fraction(-1, 8), Fraction(1, 8)),
        mechanism="oblique",
        width_base=Fraction(3, 4),
    ),
    ContinuationTask(
        "stepped_static_guard_control",
        Fraction(1),
        Fraction(15, 16),
        (Fraction(15, 16), Fraction(9, 8)),
    ),
)
