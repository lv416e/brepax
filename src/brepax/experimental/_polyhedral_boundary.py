"""Exact adjoining charts at one prescribed affine-event exchange."""

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations, pairwise, product
from math import prod
from typing import cast

from brepax.experimental._polyhedral_parameter import (
    Affine,
    ExactParameterGeometry,
    ParameterGuard,
    _Cell,
    _guard,
    _separate,
    affine_value,
    prepare_from_cells,
)
from brepax.experimental._polyhedral_translation import (
    ExactPiece,
    Vector,
    collection_pieces,
)

_ZERO = Fraction(0)


def _global(expression: Affine, center: Fraction) -> Affine:
    return expression[0] - expression[1] * center, expression[1]


def _rebase_cell(cell: _Cell, center: Fraction) -> _Cell:
    vertices = tuple(
        cast(Vector, tuple(x + center * dx for x, dx in zip(v, d[0], strict=True)))
        for v, d in zip(cell.vertices, cell.displacements, strict=True)
    )
    # Fixed facet normals also preserve the directions used as SAT axes.
    return replace(
        cell,
        vertices=vertices,
        planes=tuple((n, (affine_value(c, (center,)), c[1])) for n, c in cell.planes),
    )


@dataclass(frozen=True)
class _Side:
    center: Fraction
    geometry: ExactParameterGeometry

    @property
    def events(self) -> tuple[Affine, ...]:
        return tuple(_global(k, self.center) for k in self.geometry.knots)

    @property
    def guards(self) -> tuple[ParameterGuard, ...]:
        return tuple(
            ParameterGuard(
                _global(g.expression, self.center),
                g.kind,
                tuple(_global(v, self.center) for v in g.operands),
            )
            for g in self.geometry.guard_records
        )


def _prepare_side(
    fixed: tuple[_Cell, ...],
    moving: tuple[_Cell, ...],
    velocity: Vector,
    maximum: Fraction,
    initial_guards: set[ParameterGuard],
    center: Fraction,
) -> _Side:
    guards: set[Affine] = set()
    records: set[ParameterGuard] = set()
    for guard in initial_guards:
        expression = affine_value(guard.expression, (center,)), guard.expression[1]
        _guard(expression, guards, records, kind=guard.kind)
    geometry = prepare_from_cells(
        tuple(_rebase_cell(c, center) for c in fixed),
        tuple(_rebase_cell(c, center) for c in moving),
        velocity,
        maximum,
        guards,
        records,
    )
    return _Side(center, geometry)


def _interval(
    guards: tuple[ParameterGuard, ...],
) -> tuple[Fraction | None, Fraction | None]:
    lower, upper = None, None
    for guard in guards:
        a, b = guard.expression
        if not b:
            if a <= 0:
                raise ValueError("Invalid constant boundary guard")
            continue
        root = -a / b
        if b > 0:
            lower = root if lower is None else max(lower, root)
        else:
            upper = root if upper is None else min(upper, root)
    return lower, upper


def _closure_profile(side: _Side) -> tuple[ExactPiece, ...]:
    # Call only after certifying guard provenance at the tied reference.
    unguarded = replace(side.geometry, guards=())
    pieces = unguarded.specialize((-side.center,))
    result: list[ExactPiece] = []
    for piece in pieces:
        if piece.lower > piece.upper:
            raise ValueError("A boundary closure has reversed event order")
        if piece.lower == piece.upper:
            continue
        if result and result[-1].value(piece.lower) != piece.value(piece.lower):
            raise ValueError("A boundary closure has discontinuous volume")
        result.append(piece)
    return tuple(result)


def _equal_profiles(
    first: tuple[ExactPiece, ...], second: tuple[ExactPiece, ...]
) -> bool:
    knots = sorted(
        {q for seq in (first, second) for v in seq for q in (v.lower, v.upper)}
    )
    for lo, hi in pairwise(knots):
        midpoint = (lo + hi) / 2
        a = next(p for p in first if p.lower < midpoint < p.upper)
        b = next(p for p in second if p.lower < midpoint < p.upper)
        if a.coefficients != b.coefficients:
            return False
    return True


@dataclass(frozen=True)
class ExactEventBoundary:
    left: _Side
    right: _Side
    collision: tuple[Affine, Affine]
    interval: tuple[Fraction | None, Fraction | None]
    pieces: tuple[ExactPiece, ...]

    def specialize(self, theta: Fraction) -> tuple[ExactPiece, ...]:
        lo, hi = self.interval
        if (lo is not None and theta <= lo) or (hi is not None and theta >= hi):
            raise ValueError("Parameter is outside the certified open interval")
        if theta == 0:
            return self.pieces
        side = self.left if theta < 0 else self.right
        return side.geometry.specialize((theta - side.center,))


def prepare_boundary(
    fixed: tuple[_Cell, ...],
    moving: tuple[_Cell, ...],
    velocity: Vector,
    maximum: Fraction,
    initial_guards: set[ParameterGuard],
    radius: Fraction,
) -> ExactEventBoundary:
    guards: set[Affine] = set()
    for body in (fixed, moving):
        for a, b in combinations(body, 2):
            _separate(a, b, guards, initial_guards)
    for a, b in product(fixed, moving):
        _separate(a, b, guards, initial_guards)
    baseline = collection_pieces(
        tuple(c.hull for c in fixed), tuple(c.hull for c in moving), velocity, maximum
    )
    left = _prepare_side(fixed, moving, velocity, maximum, initial_guards, -radius)
    right = _prepare_side(fixed, moving, velocity, maximum, initial_guards, radius)
    events = set(left.events)
    if events != set(right.events):
        raise ValueError("Adjacent charts have different affine event inventories")
    tied = [(x, y) for x, y in combinations(sorted(events), 2) if x[0] == y[0]]
    if len(tied) != 1:
        raise ValueError("Expected exactly one isolated pair of tied affine events")
    collision = tied[0]
    if not 0 < collision[0][0] < maximum:
        raise ValueError("The event exchange touches a translation-path endpoint")
    intervals = []
    for side in (left, right):
        intervals.append(_interval(side.guards))
        for guard in side.guards:
            if guard.expression[0] < 0:
                raise ValueError("A chart closure does not reach the tied reference")
            if guard.expression[0] == 0:
                if guard.kind != "event":
                    raise ValueError(
                        "A non-event validity guard vanishes at the boundary"
                    )
                if set(guard.operands) != set(collision):
                    raise ValueError(
                        "Another candidate-event degeneracy occurs at the boundary"
                    )
    if intervals[0][1] != 0 or intervals[1][0] != 0:
        raise ValueError("Certified chart intervals do not adjoin at zero")
    closure_left, closure_right = _closure_profile(left), _closure_profile(right)
    if not _equal_profiles(closure_left, closure_right) or not _equal_profiles(
        closure_left, baseline
    ):
        raise ValueError(
            "Boundary profiles disagree with exact fixed-parameter geometry"
        )
    return ExactEventBoundary(
        left, right, collision, (intervals[0][0], intervals[1][1]), baseline
    )


def boundary_directional(
    boundary: ExactEventBoundary, step: Fraction, dt: Fraction, dtheta: Fraction
) -> Fraction:
    """Right directional derivative, selected by exact infinitesimal ray signs."""
    if not 0 <= step <= boundary.pieces[-1].upper:
        raise ValueError("Step is outside the bounded translation path")
    if dtheta == 0:
        values = {
            piece.slope(step) * dt
            for piece in boundary.pieces
            if (piece.lower < step < piece.upper)
            or (piece.lower == step and dt >= 0)
            or (piece.upper == step and dt <= 0)
        }
    else:
        side = boundary.right if dtheta > 0 else boundary.left
        geometry, values = side.geometry, set()
        for i, (lo, hi) in enumerate(pairwise(side.events)):
            lower = step - lo[0], dt - lo[1] * dtheta
            upper = hi[0] - step, hi[1] * dtheta - dt
            if lower < (_ZERO, _ZERO) or upper < (_ZERO, _ZERO):
                continue
            origin = _global(geometry.origins[i], side.center)
            variables = (
                Fraction(1),
                (step - origin[0]) / geometry.widths[i],
                -side.center,
            )
            tangent = (_ZERO, (dt - origin[1] * dtheta) / geometry.widths[i], dtheta)
            derivative = _ZERO
            for term, coefficient in zip(
                geometry.terms, geometry.coefficients[i], strict=True
            ):
                for j in range(3):
                    derivative += (
                        coefficient
                        * tangent[term[j]]
                        * prod(variables[term[k]] for k in range(3) if k != j)
                    )
            values.add(derivative)
    if len(values) != 1:
        raise ValueError(
            "Direction exits the path or incident directional limits disagree"
        )
    return values.pop()
