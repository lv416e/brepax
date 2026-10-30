"""Exact local parameter charts for fixed-normal convex vertex families.

Candidate vertices and their translation-event positions are affine in the
parameters. Strict linear guards retain each cell, separating certificates,
feasible vertex paths and event ordering. The resulting chamber volumes are
multivariate cubics, derived from oriented facets rather than observations.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, combinations_with_replacement, pairwise, product
from math import comb
from typing import cast

import numpy as np
from numpy.typing import NDArray

from brepax.experimental._polyhedral_contact import VertexHull, vertex_hull
from brepax.experimental._polyhedral_translation import (
    ExactPiece,
    Polynomial,
    Vector,
    _cross,
    _det,
    _dot,
    _face_hull,
    _solve,
    hull_planes,
)

Affine = tuple[Fraction, ...]
_ZERO = Fraction(0)


def _subtract(first: Affine, second: Affine) -> Affine:
    return tuple(a - b for a, b in zip(first, second, strict=True))


def _scale(values: Affine, factor: Fraction | int) -> Affine:
    return tuple(value * factor for value in values)


def affine_value(coefficients: Affine, parameters: Affine) -> Fraction:
    return coefficients[0] + _dot(coefficients[1:], parameters)


def _guard(expression: Affine, guards: set[Affine]) -> None:
    if not any(expression):
        return
    if expression[0] == 0:
        raise ValueError("Reference parameters lie on a nonregular chart boundary")
    if expression[0] < 0:
        raise ValueError("Inconsistent parameter validity certificate")
    if any(expression[1:]):
        guards.add(_scale(expression, 1 / expression[0]))


def _fraction_vector(values: Sequence[float] | NDArray[np.float64]) -> Vector:
    return cast(Vector, tuple(Fraction(float(value)) for value in values))


@dataclass(frozen=True)
class _Cell:
    vertices: tuple[Vector, ...]
    displacements: tuple[tuple[Vector, ...], ...]
    hull: VertexHull
    planes: tuple[tuple[Vector, Affine], ...]


def _cell(
    vertices: NDArray[np.float64],
    displacements: NDArray[np.float64],
    guards: set[Affine],
) -> _Cell:
    hull = vertex_hull(vertices)
    count = displacements.shape[0]
    ownership: dict[tuple[float, ...], int] = {}
    for index, vertex in enumerate(vertices):
        key = tuple(map(float, vertex))
        if key in ownership and not np.array_equal(
            displacements[:, index], displacements[:, ownership[key]]
        ):
            raise ValueError(
                "Duplicate vertices have conflicting parameter displacements"
            )
        ownership[key] = index
    exact_vertices = tuple(
        cast(Vector, tuple(Fraction(value, hull.denominator) for value in vertex))
        for vertex in hull.vertices
    )
    exact_displacements = tuple(
        tuple(
            _fraction_vector(displacements[k, ownership[tuple(map(float, vertex))]])
            for k in range(count)
        )
        for vertex in exact_vertices
    )
    planes = []
    for normal, offset, _ in hull_planes(hull, (_ZERO,) * 3):
        exposed = [
            index
            for index, vertex in enumerate(exact_vertices)
            if _dot(normal, vertex) == offset
        ]
        rates = tuple(_dot(normal, d) for d in exact_displacements[exposed[0]])
        if any(
            tuple(_dot(normal, d) for d in exact_displacements[index]) != rates
            for index in exposed
        ):
            raise ValueError("Parameter displacements change a cell facet normal")
        coefficients = (offset, *rates)
        for vertex, motions in zip(exact_vertices, exact_displacements, strict=True):
            projections = (_dot(normal, vertex), *(_dot(normal, d) for d in motions))
            _guard(_subtract(coefficients, projections), guards)
        planes.append((normal, coefficients))

    # Preserve even currently infeasible plane triples. A previously infeasible
    # vertex must not enter the cell without invalidating its local certificate.
    for indices in combinations(range(len(planes)), 3):
        rows = cast(tuple[Vector, Vector, Vector], tuple(planes[i][0] for i in indices))
        columns = tuple(
            _solve(rows, cast(Vector, tuple(planes[i][1][k] for i in indices)))
            for k in range(count + 1)
        )
        if columns[0] is None:
            continue
        points = cast(tuple[Vector, ...], columns)
        for normal, plane_offset in planes:
            residual = _subtract(tuple(_dot(normal, p) for p in points), plane_offset)
            _guard(residual if residual[0] >= 0 else _scale(residual, -1), guards)
    return _Cell(exact_vertices, exact_displacements, hull, tuple(planes))


def _separate(first: _Cell, second: _Cell, guards: set[Affine]) -> None:
    axes = {cast(Vector, tuple(Fraction(x) for x in a)) for a in first.hull.face_axes}
    axes.update(
        cast(Vector, tuple(Fraction(x) for x in a)) for a in second.hull.face_axes
    )
    axes.update(
        _cross(
            cast(Vector, tuple(Fraction(x) for x in a)),
            cast(Vector, tuple(Fraction(x) for x in b)),
        )
        for a, b in product(first.hull.edge_directions, second.hull.edge_directions)
    )
    axes.discard((_ZERO,) * 3)
    for normal in sorted(axes):
        projections = [
            [
                (_dot(normal, vertex), *(_dot(normal, d) for d in motions))
                for vertex, motions in zip(
                    body.vertices, body.displacements, strict=True
                )
            ]
            for body in (first, second)
        ]
        left, right = projections
        if max(p[0] for p in left) <= min(p[0] for p in right):
            pass
        elif max(p[0] for p in right) <= min(p[0] for p in left):
            left, right = right, left
        else:
            continue
        trial: set[Affine] = set()
        try:
            for a, b in product(left, right):
                _guard(_subtract(b, a), trial)
        except ValueError:
            # Another separator may support an open parameter neighborhood.
            continue
        guards.update(trial)
        return
    raise ValueError("Cells overlap or lack a regular parameter separation certificate")


@dataclass(frozen=True)
class _Vertex:
    columns: tuple[Vector, ...]
    lower: Affine
    upper: Affine

    def reference_at(self, step: Fraction) -> Vector:
        return cast(
            Vector,
            tuple(
                a + step * b
                for a, b in zip(self.columns[0], self.columns[1], strict=True)
            ),
        )


def _vertices(
    planes: tuple[tuple[Vector, Affine], ...],
    parameter_count: int,
    maximum_step: Fraction,
    guards: set[Affine],
) -> tuple[_Vertex, ...]:
    zero = (_ZERO,) * (parameter_count + 1)
    end = (maximum_step, *zero[1:])
    result: dict[tuple[Vector, ...], _Vertex] = {}
    for indices in combinations(range(len(planes)), 3):
        rows = cast(tuple[Vector, Vector, Vector], tuple(planes[i][0] for i in indices))
        solved = tuple(
            _solve(rows, cast(Vector, tuple(planes[i][1][k] for i in indices)))
            for k in range(parameter_count + 2)
        )
        if solved[0] is None:
            continue
        columns = cast(tuple[Vector, ...], solved)
        lower, upper, static = [zero], [end], []
        for normal, offset in planes:
            residual = _subtract(tuple(_dot(normal, c) for c in columns), offset)
            no_step = (residual[0], *residual[2:])
            if residual[1] > 0:
                upper.append(_scale(no_step, -1 / residual[1]))
            elif residual[1] < 0:
                lower.append(_scale(no_step, -1 / residual[1]))
            else:
                static.append(no_step)
        violated = next((r for r in static if r[0] > 0), None)
        if violated is not None:
            _guard(violated, guards)
            continue
        for residual in static:
            _guard(_scale(residual, -1), guards)
        lo, hi = max(lower, key=lambda q: q[0]), min(upper, key=lambda q: q[0])
        if lo == hi:
            continue
        if lo[0] == hi[0]:
            raise ValueError("Distinct affine events tie at the reference parameters")
        if lo[0] > hi[0]:
            _guard(_subtract(lo, hi), guards)
            continue
        for candidate in lower:
            _guard(_subtract(lo, candidate), guards)
        for candidate in upper:
            _guard(_subtract(candidate, hi), guards)
        _guard(_subtract(hi, lo), guards)
        result[columns] = _Vertex(columns, lo, hi)
    return tuple(result.values())


@dataclass(frozen=True)
class ExactParameterGeometry:
    """Cubic coefficients in (1, normalized step, parameters), and strict guards."""

    knots: tuple[Affine, ...]
    guards: tuple[Affine, ...]
    origins: tuple[Affine, ...]
    widths: tuple[Fraction, ...]
    terms: tuple[tuple[int, int, int], ...]
    coefficients: tuple[tuple[Fraction, ...], ...]
    pair_count: int

    def specialize(self, parameters: Affine) -> tuple[ExactPiece, ...]:
        if not all(affine_value(g, parameters) > 0 for g in self.guards):
            raise ValueError("Parameters are outside the exact open validity region")
        knots = tuple(affine_value(k, parameters) for k in self.knots)
        pieces = []
        for index, (lo, hi) in enumerate(pairwise(knots)):
            original = [_ZERO] * 4
            origin = affine_value(self.origins[index], parameters)
            for term, coefficient in zip(
                self.terms, self.coefficients[index], strict=True
            ):
                degree = term.count(1)
                factor = coefficient / self.widths[index] ** degree
                for variable in term:
                    if variable >= 2:
                        factor *= parameters[variable - 2]
                for power in range(degree + 1):
                    original[power] += (
                        factor * comb(degree, power) * (-origin) ** (degree - power)
                    )
            pieces.append(ExactPiece(lo, hi, cast(Polynomial, tuple(original))))
        return tuple(pieces)


def shift_parameter_polynomial(
    coefficients: tuple[Fraction, ...],
    terms: tuple[tuple[int, int, int], ...],
    displacement: Affine,
) -> tuple[Fraction, ...]:
    """Replace normalized step u by u + an affine parameter expression."""
    size = len(displacement) + 1
    substitutions = [tuple(Fraction(i == j) for j in range(size)) for i in range(size)]
    substitutions[1] = (displacement[0], Fraction(1), *displacement[1:])
    result = dict.fromkeys(terms, _ZERO)
    for term, coefficient in zip(terms, coefficients, strict=True):
        if coefficient == 0:
            continue
        factors = [
            [(i, c) for i, c in enumerate(substitutions[variable]) if c]
            for variable in term
        ]
        for selection in product(*factors):
            key = cast(tuple[int, int, int], tuple(sorted(i for i, _ in selection)))
            result[key] += (
                coefficient * selection[0][1] * selection[1][1] * selection[2][1]
            )
    return tuple(result[term] for term in terms)


def parameter_derivative_polynomials(
    coefficients: tuple[Fraction, ...],
    terms: tuple[tuple[int, int, int], ...],
    origin: Affine,
    width: Fraction,
) -> tuple[tuple[Fraction, ...], ...]:
    """Differentiate with respect to physical step/theta before float packing.

    Collecting identical monomials here preserves structural zero derivatives,
    including cancellation of a moving normalization origin's chain rule.
    """
    derivatives = [dict.fromkeys(terms, _ZERO) for _ in origin]
    for term, coefficient in zip(terms, coefficients, strict=True):
        for variable in range(1, len(origin) + 1):
            multiplicity = term.count(variable)
            if not multiplicity or not coefficient:
                continue
            reduced = list(term)
            reduced[reduced.index(variable)] = 0
            key = cast(tuple[int, int, int], tuple(sorted(reduced)))
            factor = coefficient * multiplicity
            if variable == 1:
                derivatives[0][key] += factor / width
                for parameter in range(1, len(origin)):
                    derivatives[parameter][key] -= factor * origin[parameter] / width
            else:
                derivatives[variable - 1][key] += factor
    return tuple(tuple(d[term] for term in terms) for d in derivatives)


def prepare_geometry(
    fixed: Sequence[tuple[NDArray[np.float64], NDArray[np.float64]]],
    moving: Sequence[tuple[NDArray[np.float64], NDArray[np.float64]]],
    velocity: NDArray[np.float64],
    maximum_step: float,
) -> ExactParameterGeometry:
    guards: set[Affine] = set()
    count = fixed[0][1].shape[0]
    fixed_cells = tuple(_cell(v, d, guards) for v, d in fixed)
    moving_cells = tuple(_cell(v, d, guards) for v, d in moving)
    for body in (fixed_cells, moving_cells):
        for a, b in combinations(body, 2):
            _separate(a, b, guards)
    for a, b in product(fixed_cells, moving_cells):
        _separate(a, b, guards)
    vector = _fraction_vector(velocity)
    pairs = []
    for a, b in product(fixed_cells, moving_cells):
        planes = tuple((n, (c[0], _ZERO, *c[1:])) for n, c in a.planes) + tuple(
            (n, (c[0], _dot(n, vector), *c[1:])) for n, c in b.planes
        )
        pairs.append((planes, _vertices(planes, count, Fraction(maximum_step), guards)))
    knots = {(_ZERO,) * (count + 1), (Fraction(maximum_step), *(_ZERO,) * count)}
    for _, vertices in pairs:
        for vertex in vertices:
            knots.update((vertex.lower, vertex.upper))
    ordered = tuple(sorted(knots, key=lambda k: k[0]))
    for lo, hi in pairwise(ordered):
        _guard(_subtract(hi, lo), guards)

    terms = tuple(combinations_with_replacement(range(count + 2), 3))
    coefficients, origins, widths = [], [], []
    # A common exact origin removes large world translations before packing.
    anchor = fixed_cells[0].vertices[0]
    for lo, hi in pairwise(ordered):
        midpoint = (lo[0] + hi[0]) / 2
        # Match the actual packed affine origin, not an unrepresentable exact
        # event or a fixed midpoint far from a moving overlap onset.
        origin = tuple(Fraction(float(value)) for value in lo)
        width = Fraction(float(hi[0] - lo[0]))
        if width <= 0 or float(width) < np.finfo(np.float64).tiny:
            raise ValueError("Parameter chamber normalization underflows float64")
        origins.append(origin)
        widths.append(width)
        polynomial = dict.fromkeys(terms, _ZERO)
        for planes, vertices in pairs:
            active = tuple(v for v in vertices if v.lower[0] < midpoint < v.upper[0])
            points = tuple(v.reference_at(midpoint) for v in active)
            if len(points) < 4:
                continue
            differences = [
                cast(Vector, _subtract(point, points[0])) for point in points[1:]
            ]
            if not any(_det(*triple) for triple in combinations(differences, 3)):
                # The reference intersection has no volume. Verify that this
                # is an identity of the affine family rather than a collapsed
                # reference which becomes full-dimensional under perturbation.
                affine_differences = [
                    tuple(
                        cast(Vector, _subtract(a, b))
                        for a, b in zip(vertex.columns, active[0].columns, strict=True)
                    )
                    for vertex in active[1:]
                ]
                for triple in combinations(affine_differences, 3):
                    rank_polynomial = dict.fromkeys(terms, _ZERO)
                    for powers in product(range(count + 2), repeat=3):
                        key = cast(tuple[int, int, int], tuple(sorted(powers)))
                        rank_polynomial[key] += _det(
                            *(c[p] for c, p in zip(triple, powers, strict=True))
                        )
                    if any(rank_polynomial.values()):
                        raise ValueError(
                            "Intersection dimension changes at reference parameters"
                        )
                continue
            seen: set[tuple[int, ...]] = set()
            for normal, offset in planes:
                face = tuple(
                    i
                    for i, point in enumerate(points)
                    if _dot(normal, point) == offset[0] + offset[1] * midpoint
                )
                if len(face) < 3 or face in seen:
                    continue
                seen.add(face)
                hull = _face_hull(face, points, normal)
                for i in range(1, len(hull) - 1):
                    triangle = tuple(
                        active[j].columns for j in (hull[0], hull[i], hull[i + 1])
                    )
                    shifted = tuple(
                        (
                            cast(
                                Vector,
                                tuple(
                                    a + origin[0] * b - c
                                    for a, b, c in zip(
                                        cols[0], cols[1], anchor, strict=True
                                    )
                                ),
                            ),
                            cast(Vector, _scale(cols[1], width)),
                            *(
                                cast(
                                    Vector,
                                    tuple(
                                        a + origin[k + 1] * b
                                        for a, b in zip(column, cols[1], strict=True)
                                    ),
                                )
                                for k, column in enumerate(cols[2:])
                            ),
                        )
                        for cols in triangle
                    )
                    for powers in product(range(count + 2), repeat=3):
                        key = cast(tuple[int, int, int], tuple(sorted(powers)))
                        polynomial[key] += (
                            _det(*(c[p] for c, p in zip(shifted, powers, strict=True)))
                            / 6
                        )
        coefficients.append(tuple(polynomial[term] for term in terms))
    return ExactParameterGeometry(
        ordered,
        tuple(sorted(guards)),
        tuple(origins),
        tuple(widths),
        terms,
        tuple(coefficients),
        len(pairs),
    )
