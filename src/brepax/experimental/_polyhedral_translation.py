"""Exact host preparation for small translated convex-cell collections.

Fixed plane normals and affine offsets give affine candidate vertices.
Their feasible intervals delimit chambers; oriented facet fans give a cubic
volume on each chamber. No CAD observations or polynomial fitting are used.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, pairwise, product
from typing import cast

from brepax.experimental._polyhedral_contact import VertexHull

Vector = tuple[Fraction, Fraction, Fraction]
Polynomial = tuple[Fraction, Fraction, Fraction, Fraction]
Plane = tuple[Vector, Fraction, Fraction]
_ZERO = Fraction(0)


def _dot(first: Sequence[Fraction], second: Sequence[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(first, second, strict=True)), _ZERO)


def _cross(first: Vector, second: Vector) -> Vector:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _det(first: Vector, second: Vector, third: Vector) -> Fraction:
    return _dot(first, _cross(second, third))


def _solve(rows: tuple[Vector, Vector, Vector], rhs: Vector) -> Vector | None:
    determinant = _det(*rows)
    if not determinant:
        return None
    cofactors = (
        _cross(rows[1], rows[2]),
        _cross(rows[2], rows[0]),
        _cross(rows[0], rows[1]),
    )
    return cast(
        Vector,
        tuple(
            sum((rhs[i] * cofactors[i][j] for i in range(3)), _ZERO) / determinant
            for j in range(3)
        ),
    )


@dataclass(frozen=True)
class _VertexPath:
    start: Vector
    speed: Vector
    lower: Fraction
    upper: Fraction

    def at(self, step: Fraction) -> Vector:
        return cast(
            Vector,
            tuple(a + step * b for a, b in zip(self.start, self.speed, strict=True)),
        )


@dataclass(frozen=True)
class ExactPiece:
    """A polynomial in the original dimensionless step on an exact interval."""

    lower: Fraction
    upper: Fraction
    coefficients: Polynomial

    def value(self, step: Fraction) -> Fraction:
        c = self.coefficients
        return ((c[3] * step + c[2]) * step + c[1]) * step + c[0]

    def slope(self, step: Fraction) -> Fraction:
        c = self.coefficients
        return (3 * c[3] * step + 2 * c[2]) * step + c[1]

    def reparameterize(self, origin: Fraction, width: Fraction) -> Polynomial:
        """Use the actual stored float origin/width, supplied as rationals."""
        c = self.coefficients
        return (
            self.value(origin),
            width * self.slope(origin),
            width**2 * (c[2] + 3 * c[3] * origin),
            width**3 * c[3],
        )


def hull_planes(hull: VertexHull, velocity: Vector) -> tuple[Plane, ...]:
    """Recover oriented facets of a bounded, full-dimensional vertex hull."""
    vertices = tuple(
        cast(Vector, tuple(Fraction(x, hull.denominator) for x in vertex))
        for vertex in hull.vertices
    )
    planes = []
    for axis in hull.face_axes:
        for sign in (-1, 1):
            normal = cast(Vector, tuple(Fraction(sign * value) for value in axis))
            projections = tuple(_dot(normal, vertex) for vertex in vertices)
            offset = max(projections)
            exposed = [
                vertex
                for vertex, projection in zip(vertices, projections, strict=True)
                if projection == offset
            ]
            if len(exposed) < 3:
                continue
            vectors = [
                cast(Vector, tuple(a - b for a, b in zip(v, exposed[0], strict=True)))
                for v in exposed[1:]
            ]
            if any(_cross(a, b) != (_ZERO,) * 3 for a, b in combinations(vectors, 2)):
                planes.append((normal, offset, _dot(normal, velocity)))
    return tuple(planes)


def _vertex_paths(
    planes: tuple[Plane, ...], upper: Fraction
) -> tuple[_VertexPath, ...]:
    candidates: dict[tuple[Vector, Vector], _VertexPath] = {}
    for indices in combinations(range(len(planes)), 3):
        rows = cast(tuple[Vector, Vector, Vector], tuple(planes[i][0] for i in indices))
        start = _solve(rows, cast(Vector, tuple(planes[i][1] for i in indices)))
        if start is None:
            continue
        speed = _solve(rows, cast(Vector, tuple(planes[i][2] for i in indices)))
        assert speed is not None
        key = start, speed
        if key in candidates:
            continue
        lo, hi = _ZERO, upper
        valid = True
        for normal, offset, rate in planes:
            intercept = _dot(normal, start) - offset
            slope = _dot(normal, speed) - rate
            if slope > 0:
                hi = min(hi, -intercept / slope)
            elif slope < 0:
                lo = max(lo, -intercept / slope)
            elif intercept > 0:
                valid = False
                break
        if valid and lo < hi:
            candidates[key] = _VertexPath(start, speed, lo, hi)
    return tuple(candidates.values())


def _face_hull(
    indices: tuple[int, ...], points: tuple[Vector, ...], normal: Vector
) -> list[int]:
    # Cyclic projection preserves positive orientation toward the dropped axis.
    axis = max(range(3), key=lambda k: abs(normal[k]))
    i, j = (axis + 1) % 3, (axis + 2) % 3
    ordered = sorted(indices, key=lambda index: (points[index][i], points[index][j]))

    def turn(a: int, b: int, c: int) -> Fraction:
        return (points[b][i] - points[a][i]) * (points[c][j] - points[a][j]) - (
            points[b][j] - points[a][j]
        ) * (points[c][i] - points[a][i])

    halves = []
    for items in (ordered, ordered[::-1]):
        half: list[int] = []
        for index in items:
            while len(half) >= 2 and turn(half[-2], half[-1], index) <= 0:
                half.pop()
            half.append(index)
        halves.append(half[:-1])
    hull = halves[0] + halves[1]
    if normal[axis] < 0:
        hull.reverse()
    return hull


def _volume_coefficients(
    planes: tuple[Plane, ...], active: tuple[_VertexPath, ...], midpoint: Fraction
) -> Polynomial:
    points = tuple(vertex.at(midpoint) for vertex in active)
    if len(points) < 4:
        return (_ZERO,) * 4
    vectors = [
        cast(Vector, tuple(a - b for a, b in zip(p, points[0], strict=True)))
        for p in points[1:]
    ]
    if not any(_det(*triple) for triple in combinations(vectors, 3)):
        return (_ZERO,) * 4
    seen_faces: set[tuple[int, ...]] = set()
    coefficients = [_ZERO] * 4
    for normal, offset, rate in planes:
        face = tuple(
            i
            for i, point in enumerate(points)
            if _dot(normal, point) == offset + midpoint * rate
        )
        if len(face) < 3 or face in seen_faces:
            continue
        seen_faces.add(face)
        hull = _face_hull(face, points, normal)
        for i in range(1, len(hull) - 1):
            triangle = tuple(active[k] for k in (hull[0], hull[i], hull[i + 1]))
            for powers in product(range(2), repeat=3):
                rows = cast(
                    tuple[Vector, Vector, Vector],
                    tuple(
                        v.speed if power else v.start
                        for v, power in zip(triangle, powers, strict=True)
                    ),
                )
                coefficients[sum(powers)] += _det(*rows) / 6
    return cast(Polynomial, tuple(coefficients))


def _pair_pieces(planes: tuple[Plane, ...], upper: Fraction) -> tuple[ExactPiece, ...]:
    vertices = _vertex_paths(planes, upper)
    knots = sorted(
        {_ZERO, upper, *(v.lower for v in vertices), *(v.upper for v in vertices)}
    )
    pieces: list[ExactPiece] = []
    for lo, hi in pairwise(knots):
        midpoint = (lo + hi) / 2
        active = tuple(v for v in vertices if v.lower < midpoint < v.upper)
        piece = ExactPiece(lo, hi, _volume_coefficients(planes, active, midpoint))
        if piece.value(midpoint) < 0:
            raise ValueError("Exact translation preparation produced negative volume")
        if pieces and pieces[-1].value(lo) != piece.value(lo):
            raise ValueError(
                "Exact translation preparation produced discontinuous volume"
            )
        pieces.append(piece)
    return tuple(pieces)


def collection_pieces(
    fixed: tuple[VertexHull, ...],
    moving: tuple[VertexHull, ...],
    velocity: Vector,
    upper: Fraction,
) -> tuple[ExactPiece, ...]:
    """Sum all pairs only after disjoint-interior partitions have been checked."""
    fixed_planes = tuple(hull_planes(hull, (_ZERO,) * 3) for hull in fixed)
    moving_planes = tuple(hull_planes(hull, velocity) for hull in moving)
    pairs = tuple(
        _pair_pieces(a + b, upper) for a, b in product(fixed_planes, moving_planes)
    )
    knots = sorted({piece.lower for pair in pairs for piece in pair} | {upper})
    indices = [0] * len(pairs)
    result = []
    for lo, hi in pairwise(knots):
        coefficients = [_ZERO] * 4
        for i, pair in enumerate(pairs):
            while pair[indices[i]].upper <= lo:
                indices[i] += 1
            for k, value in enumerate(pair[indices[i]].coefficients):
                coefficients[k] += value
        result.append(ExactPiece(lo, hi, cast(Polynomial, tuple(coefficients))))
    return tuple(result)
