"""Exact separation predicates for small represented vertex hulls.

These predicates concern the convex hull of supplied binary64 coordinates,
not the original CAD surfaces or their tolerances. Integer arithmetic avoids
discarding a nearly parallel separating axis or rounding a gap to contact.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from math import gcd
from typing import Literal

import numpy as np
from numpy.typing import NDArray

_Vector = tuple[int, int, int]
_Point2 = tuple[Fraction, Fraction]
_Relation = Literal[
    "separated", "face_contact", "edge_contact", "point_contact", "overlap"
]


def _subtract(first: _Vector, second: _Vector) -> _Vector:
    return first[0] - second[0], first[1] - second[1], first[2] - second[2]


def _cross(first: _Vector, second: _Vector) -> _Vector:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _dot(first: _Vector, second: _Vector) -> int:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _axis(vector: _Vector) -> _Vector:
    divisor = gcd(*vector)
    if divisor == 0:
        return vector
    first = next(value for value in vector if value)
    if first < 0:
        divisor = -divisor
    return vector[0] // divisor, vector[1] // divisor, vector[2] // divisor


@dataclass(frozen=True)
class VertexHull:
    """Dyadic vertex coordinates and exact hull facet/edge directions."""

    vertices: tuple[_Vector, ...]
    denominator: int
    face_axes: tuple[_Vector, ...]
    edge_directions: tuple[_Vector, ...]


@dataclass(frozen=True)
class ContactSupport:
    """Exact exposed features on an oriented common support plane.

    ``axis`` points from the first hull toward the second. The plane equation
    in world coordinates is ``axis dot x = offset / denominator``. Vertex
    indices address the corresponding hull's original ``vertices`` tuple.
    This is separate from the rounded largest-gap diagnostic in HullRelation.
    """

    axis: _Vector
    offset: int
    denominator: int
    first_vertices: tuple[int, ...]
    second_vertices: tuple[int, ...]


@dataclass(frozen=True)
class HullRelation:
    """Exact combinatorial relation with rounded length diagnostics.

    ``axis_gap`` is the largest normalized projection separation, not Euclidean
    distance or penetration depth. Its sign can underflow on conversion to a
    float; ``relation`` is determined from the original integer signs.
    """

    relation: _Relation
    axis_gap: float
    axis: tuple[float, float, float]
    tested_axis_count: int
    support: ContactSupport | None = None


def _orient2(first: _Point2, second: _Point2, third: _Point2) -> Fraction:
    return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
        third[0] - first[0]
    )


def _polygon_hull(points: list[_Point2]) -> list[_Point2]:
    ordered = sorted(set(points))
    if len(ordered) <= 1:
        return ordered
    halves = []
    for sequence in (ordered, ordered[::-1]):
        half: list[_Point2] = []
        for point in sequence:
            while len(half) >= 2 and _orient2(half[-2], half[-1], point) <= 0:
                half.pop()
            half.append(point)
        halves.append(half)
    return halves[0][:-1] + halves[1][:-1]


def _segments(polygon: list[_Point2]) -> list[tuple[_Point2, _Point2]]:
    if len(polygon) < 2:
        return []
    if len(polygon) == 2:
        return [(polygon[0], polygon[1])]
    return list(zip(polygon, polygon[1:] + polygon[:1], strict=True))


def _inside(point: _Point2, polygon: list[_Point2]) -> bool:
    if len(polygon) == 1:
        return point == polygon[0]
    if len(polygon) == 2:
        first, second = polygon
        return _orient2(first, second, point) == 0 and all(
            min(first[index], second[index])
            <= point[index]
            <= max(first[index], second[index])
            for index in (0, 1)
        )
    return all(
        _orient2(first, second, point) >= 0 for first, second in _segments(polygon)
    )


def _touch_dimension(first: list[_Vector], second: list[_Vector], axis: _Vector) -> int:
    # Both exposed features lie in the same support plane. Dropping a coordinate
    # with a nonzero normal component is injective on that plane, including for
    # a face against an edge or two crossing edges.
    omitted = next(index for index, value in enumerate(axis) if value)
    retained = [index for index in range(3) if index != omitted]

    def project(vertices: list[_Vector]) -> list[_Point2]:
        return _polygon_hull(
            [
                (Fraction(vertex[retained[0]]), Fraction(vertex[retained[1]]))
                for vertex in vertices
            ]
        )

    left, right = project(first), project(second)
    common = [point for point in left if _inside(point, right)]
    common.extend(point for point in right if _inside(point, left))
    for (a, b), (c, d) in product(_segments(left), _segments(right)):
        r = b[0] - a[0], b[1] - a[1]
        s = d[0] - c[0], d[1] - c[1]
        denominator = r[0] * s[1] - r[1] * s[0]
        if denominator == 0:
            # Collinear overlap endpoints are already included by containment.
            continue
        offset = c[0] - a[0], c[1] - a[1]
        t = (offset[0] * s[1] - offset[1] * s[0]) / denominator
        u = (offset[0] * r[1] - offset[1] * r[0]) / denominator
        if 0 <= t <= 1 and 0 <= u <= 1:
            common.append((a[0] + t * r[0], a[1] + t * r[1]))
    intersection = _polygon_hull(common)
    if not intersection:
        raise ValueError("Exact touching hulls have inconsistent exposed features")
    return min(len(intersection) - 1, 2)


def vertex_hull(points: NDArray[np.float64]) -> VertexHull:
    """Construct the needed hull axes by exact supporting-plane enumeration.

    The bounded CAD intake supplies small convex cells. Enumerating vertex
    triples is deliberately simple and costs O(n**4); this is not a general
    large-mesh hull implementation. Redundant collinear vertices are harmless.
    """
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite vertex coordinates with shape (N, 3)")
    ratios = [
        [float(value).as_integer_ratio() for value in point]
        for point in np.unique(points, axis=0)
    ]
    if len(ratios) < 4:
        raise ValueError("A represented solid needs at least four vertices")
    denominator = max(den for point in ratios for _, den in point)
    vertices = tuple(
        (
            point[0][0] * (denominator // point[0][1]),
            point[1][0] * (denominator // point[1][1]),
            point[2][0] * (denominator // point[2][1]),
        )
        for point in ratios
    )
    planes: dict[tuple[_Vector, int], frozenset[int]] = {}
    for first, second, third in combinations(range(len(vertices)), 3):
        axis = _axis(
            _cross(
                _subtract(vertices[second], vertices[first]),
                _subtract(vertices[third], vertices[first]),
            )
        )
        if axis == (0, 0, 0):
            continue
        offset = _dot(axis, vertices[first])
        key = axis, offset
        if key in planes:
            continue
        projections = [_dot(axis, point) - offset for point in vertices]
        if min(projections) >= 0 or max(projections) <= 0:
            planes[key] = frozenset(
                index for index, value in enumerate(projections) if value == 0
            )
    if len(planes) < 4:
        raise ValueError("Represented vertices have no positive-volume convex hull")
    edges = set()
    for first, second in combinations(range(len(vertices)), 2):
        containing_axes = {
            axis
            for (axis, _), indices in planes.items()
            if first in indices and second in indices
        }
        # Two nonparallel supporting planes meet along a hull edge. Using all
        # vertex pairs on that edge retains intermediate collinear vertices.
        if len(containing_axes) >= 2:
            edges.add(_axis(_subtract(vertices[second], vertices[first])))
    return VertexHull(
        vertices,
        denominator,
        tuple(sorted({axis for axis, _ in planes})),
        tuple(sorted(edges)),
    )


def hull_relation(first: VertexHull, second: VertexHull) -> HullRelation:
    """Classify two represented hulls using the complete exact SAT axis set."""
    denominator = max(first.denominator, second.denominator)
    first_factor, second_factor = (
        denominator // first.denominator,
        denominator // second.denominator,
    )
    first_vertices = tuple(
        (x * first_factor, y * first_factor, z * first_factor)
        for x, y, z in first.vertices
    )
    second_vertices = tuple(
        (x * second_factor, y * second_factor, z * second_factor)
        for x, y, z in second.vertices
    )
    axes = set(first.face_axes) | set(second.face_axes)
    axes.update(
        _axis(_cross(left, right))
        for left, right in product(first.edge_directions, second.edge_directions)
    )
    axes.discard((0, 0, 0))
    separating = False
    touching: tuple[list[_Vector], list[_Vector], _Vector] | None = None
    support: ContactSupport | None = None
    best_gap = -np.inf
    best_axis = (0.0, 0.0, 0.0)
    for axis in sorted(axes):
        first_projection = [_dot(axis, vertex) for vertex in first_vertices]
        second_projection = [_dot(axis, vertex) for vertex in second_vertices]
        gap = max(
            min(second_projection) - max(first_projection),
            min(first_projection) - max(second_projection),
        )
        separating = separating or gap > 0
        if gap == 0 and touching is None:
            first_below = max(first_projection) == min(second_projection)
            plane = max(first_projection) if first_below else min(first_projection)
            first_indices = tuple(
                index for index, value in enumerate(first_projection) if value == plane
            )
            second_indices = tuple(
                index for index, value in enumerate(second_projection) if value == plane
            )
            touching = (
                [first_vertices[index] for index in first_indices],
                [second_vertices[index] for index in second_indices],
                axis,
            )
            sign = 1 if first_below else -1
            support = ContactSupport(
                (sign * axis[0], sign * axis[1], sign * axis[2]),
                sign * plane,
                denominator,
                first_indices,
                second_indices,
            )
        largest = max(abs(value) for value in axis)
        scaled_axis = np.array([value / largest for value in axis])
        norm = float(np.linalg.norm(scaled_axis))
        rounded_gap = float(Fraction(gap, denominator * largest)) / norm
        if rounded_gap > best_gap:
            best_gap = rounded_gap
            normalized = scaled_axis / norm
            best_axis = float(normalized[0]), float(normalized[1]), float(normalized[2])
    relation: _Relation
    if separating:
        relation = "separated"
    elif touching is None:
        relation = "overlap"
    else:
        dimension = _touch_dimension(*touching)
        relation = ("point_contact", "edge_contact", "face_contact")[dimension]
    return HullRelation(
        relation,
        float(best_gap),
        best_axis,
        len(axes),
        support if relation not in ("separated", "overlap") else None,
    )
