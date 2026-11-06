"""Certified minimal changes of dimension handles that reach a volume band.

Each handle contributes an exact univariate volume law in its own value.
Holes contribute ``-pi * height * (r**2 - r0**2)`` and planar faces
``area * offset``. Because the laws add up, the reachable volume over the
handle box is exact, and the cheapest weighted change reaching a band is
found by branch and bound with exact rational lower bounds. The certificate
applies to the represented laws, whose coefficients are the prepared
measurements; only a validated rebuild connects a witness to edited CAD.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal

import numpy as np

from brepax.experimental.applications.face_offset import PreparedFaceOffset
from brepax.experimental.applications.hole_edit import PreparedHoleEdit

# Rational enclosure of pi: its first 40 decimals, truncated, and one unit
# of the last place above.
_PI_LOWER = Fraction("3.1415926535897932384626433832795028841971")
_PI_UPPER = _PI_LOWER + Fraction(1, 10**40)

# First handle, second handle, limit, and whether only increases count.
_Limit = tuple[int, int, Fraction, bool]


@dataclass(frozen=True)
class DimensionLaw:
    """Exact volume contribution of one handle as a function of its value.

    The contribution is ``r(x) - r(x0) + pi * (q(x) - q(x0))`` with
    ``r`` and ``q`` given by ``rational`` and ``pi_coefficients`` (constant,
    linear, quadratic). It must be monotone on ``[lower, upper]``. Lengths
    and volumes use the input CAD's units; ``weight`` scales the cost
    ``weight * abs(x - reference)``.
    """

    label: str
    reference: float
    lower: float
    upper: float
    weight: float
    rational: tuple[Fraction, Fraction, Fraction]
    pi_coefficients: tuple[Fraction, Fraction, Fraction]


@dataclass(frozen=True)
class TravelLimit:
    """Limit on the summed absolute change of two handles named by label.

    ``abs(x - reference)`` of ``first`` plus that of ``second`` must not exceed
    ``limit``. Two edited holes keep a ligament this way, because their walls
    approach by at most their combined travel. With ``increases_only``, a handle
    below its reference counts as no travel: two holes cannot touch by shrinking.
    """

    first: str
    second: str
    limit: float
    increases_only: bool = False


@dataclass(frozen=True)
class DimensionQueryResult:
    """Outcome of :func:`minimal_volume_change`.

    ``optimal_within_tolerance``: ``values`` reach the band for every value
    of pi in the enclosure, and no handle values in the box reach it with a
    weighted change below ``lower_bound``, which is within ``tolerance`` of
    ``change``. ``infeasible``: no handle values in the box satisfying the
    travel limits reach the band. ``reachable`` is the exact range of the
    volume change over the box, widened by the pi enclosure and ignoring travel
    limits. ``unresolved``: the node budget ran out; ``values`` is the best
    witness found, if any, and ``lower_bound`` still bounds every solution.
    """

    status: Literal["optimal_within_tolerance", "infeasible", "unresolved"]
    values: tuple[float, ...] | None
    change: Fraction | None
    lower_bound: Fraction | None
    reachable: tuple[Fraction, Fraction]
    volume_change: tuple[Fraction, Fraction] | None
    nodes: int


def _polynomial(coefficients: Sequence[Fraction], x: Fraction) -> Fraction:
    return coefficients[0] + x * (coefficients[1] + x * coefficients[2])


class _Handle:
    """A law with pi fixed to the lower end of its enclosure."""

    def __init__(self, law: DimensionLaw) -> None:
        self.law = law
        self.x0 = Fraction(law.reference)
        self.lo = Fraction(law.lower)
        self.hi = Fraction(law.upper)
        self.w = Fraction(law.weight)
        self.p = tuple(
            r + _PI_LOWER * q
            for r, q in zip(law.rational, law.pi_coefficients, strict=True)
        )
        self.p0 = _polynomial(self.p, self.x0)

    def delta(self, x: Fraction) -> Fraction:
        return _polynomial(self.p, x) - self.p0

    def pi_spread(self) -> Fraction:
        """Largest change of the contribution over the pi enclosure."""
        q = self.law.pi_coefficients
        q0 = _polynomial(q, self.x0)
        points = [self.lo, self.hi]
        if q[2] != 0:
            vertex = -q[1] / (2 * q[2])
            if self.lo < vertex < self.hi:
                points.append(vertex)
        spread = max(abs(_polynomial(q, x) - q0) for x in points)
        return spread * (_PI_UPPER - _PI_LOWER)

    def cost(self, x: Fraction) -> Fraction:
        return self.w * abs(x - self.x0)

    def slope(self, x: Fraction) -> Fraction:
        return self.p[1] + 2 * self.p[2] * x

    def branch(
        self, near: Fraction, far: Fraction
    ) -> tuple[tuple[Fraction, Fraction], list[tuple[Fraction, Fraction]]]:
        """Base (delta, cost) at ``near`` and convex hull pieces toward ``far``.

        Each piece is ``(cost per unit delta, signed delta length)``. The
        (delta, cost) curve between the endpoints is underestimated by its
        chord where cost is concave in delta, and by the endpoint tangents
        where it is convex.
        """
        dn, df = self.delta(near), self.delta(far)
        cn, cf = self.cost(near), self.cost(far)
        if dn == df:
            return (dn, cn), []
        sign = 1 if far > near else -1
        sn, sf = self.slope(near), self.slope(far)
        chord = (cf - cn) / (df - dn)
        if sn != 0 and sf == 0:
            # Cost per unit of volume diverges toward the flat end, so the curve
            # is convex and the tangent at the near end stays below it. The
            # remaining jump at ``far`` is left to branching.
            return (dn, cn), [(self.w * sign / sn, df - dn)]
        if sn != 0 and sf != 0:
            tangent_near = self.w * sign / sn
            tangent_far = self.w * sign / sf
            if abs(tangent_far) > abs(tangent_near):
                meet = (cf - tangent_far * df - cn + tangent_near * dn) / (
                    tangent_near - tangent_far
                )
                if min(dn, df) < meet < max(dn, df):
                    return (dn, cn), [
                        (tangent_near, meet - dn),
                        (tangent_far, df - meet),
                    ]
        return (dn, cn), [(chord, df - dn)]


def _relax(
    handles: list[_Handle],
    boxes: list[tuple[Fraction, Fraction]],
    band: tuple[Fraction, Fraction],
) -> tuple[Fraction, list[Fraction]] | None:
    """Exact minimum of the hull costs with the summed delta in the band."""
    low, high = band
    deltas, cost = [], Fraction(0)
    pieces: list[tuple[Fraction, Fraction, int]] = []
    for i, (handle, (lo, hi)) in enumerate(zip(handles, boxes, strict=True)):
        if lo < handle.x0 < hi:
            starts = [handle.branch(handle.x0, lo), handle.branch(handle.x0, hi)]
            base = (Fraction(0), Fraction(0))
        else:
            near, far = (
                (lo, hi) if abs(lo - handle.x0) <= abs(hi - handle.x0) else (hi, lo)
            )
            starts = [handle.branch(near, far)]
            base = starts[0][0]
        deltas.append(base[0])
        cost += base[1]
        for _, branch_pieces in starts:
            pieces.extend((slope, length, i) for slope, length in branch_pieces)
    total = sum(deltas, Fraction(0))
    least = total + sum(
        (min(Fraction(0), length) for _, length, _ in pieces), Fraction(0)
    )
    most = total + sum(
        (max(Fraction(0), length) for _, length, _ in pieces), Fraction(0)
    )
    if most < low or least > high:
        return None
    if low <= total <= high:
        return cost, deltas
    need = low - total if total < low else high - total
    direction = 1 if need > 0 else -1
    remaining = abs(need)
    for slope, length, i in sorted(
        (
            (abs(slope), length, i)
            for slope, length, i in pieces
            if length * direction > 0
        ),
        key=lambda piece: piece[0],
    ):
        take = min(remaining, abs(length))
        cost += slope * take
        deltas[i] += direction * take
        remaining -= take
        if remaining == 0:
            return cost, deltas
    return None


def _hull_cost(
    handle: _Handle, box: tuple[Fraction, Fraction], delta: Fraction
) -> Fraction:
    lo, hi = box
    if lo < handle.x0 < hi:
        far = lo if (handle.delta(lo)) * delta > 0 else hi
        return _hull_cost(handle, (handle.x0, far), delta)
    near, far = (lo, hi) if abs(lo - handle.x0) <= abs(hi - handle.x0) else (hi, lo)
    (start, cost), pieces = handle.branch(near, far)
    position = start
    for slope, length in pieces:
        end = position + length
        if min(position, end) <= delta <= max(position, end):
            return cost + slope * (delta - position)
        cost += slope * length
        position = end
    return cost


def _float_bracket(
    handle: _Handle, lo: Fraction, hi: Fraction, target: Fraction
) -> tuple[float, float]:
    """Adjacent floats in [lo, hi] whose deltas bracket ``target``.

    Both entries are equal when a float hits the target or an end of the
    interval bounds it. An unchanged handle keeps its reference exactly.
    """
    if target == 0 and min(lo, hi) <= handle.x0 <= max(lo, hi):
        return handle.law.reference, handle.law.reference
    down, up = (lo, hi) if handle.delta(lo) <= handle.delta(hi) else (hi, lo)
    a, b = float(down), float(up)
    if handle.delta(Fraction(a)) >= target:
        return a, a
    if handle.delta(Fraction(b)) <= target:
        return b, b
    while True:
        middle = (a + b) / 2
        if middle in (a, b):
            return a, b
        value = handle.delta(Fraction(middle))
        if value == target:
            return middle, middle
        if value < target:
            a = middle
        else:
            b = middle


def _float_value(
    handle: _Handle, lo: Fraction, hi: Fraction, target: Fraction
) -> float:
    """The float in [lo, hi] whose delta is closest to ``target``."""
    a, b = _float_bracket(handle, lo, hi, target)
    below = abs(target - handle.delta(Fraction(a)))
    above = abs(handle.delta(Fraction(b)) - target)
    return a if below <= above else b


def _inside(x: float, lo: Fraction, hi: Fraction) -> float:
    value = Fraction(x)
    if value < lo:
        return math.nextafter(x, math.inf)
    if value > hi:
        return math.nextafter(x, -math.inf)
    return x


def _witness(
    handles: list[_Handle],
    boxes: list[tuple[Fraction, Fraction]],
    deltas: list[Fraction],
    band: tuple[Fraction, Fraction],
) -> tuple[Fraction, list[float]] | None:
    """Floats near the relaxed solution, adjusted to lie exactly in the band."""
    low, high = band
    values = [
        _inside(_float_value(h, lo, hi, d), lo, hi)
        for h, (lo, hi), d in zip(handles, boxes, deltas, strict=True)
    ]
    total = sum(
        (h.delta(Fraction(x)) for h, x in zip(handles, values, strict=True)),
        Fraction(0),
    )
    if not low <= total <= high:
        # Rounding moved the sum just outside the band: move one handle to
        # the float on the inner side of the nearest band edge.
        edge = low if total < low else high
        order = sorted(
            range(len(handles)),
            key=lambda i: (
                -abs(handles[i].delta(boxes[i][1]) - handles[i].delta(boxes[i][0]))
            ),
        )
        for i in order:
            rest = total - handles[i].delta(Fraction(values[i]))
            pair = _float_bracket(handles[i], *boxes[i], edge - rest)
            inside = [
                _inside(x, *boxes[i])
                for x in pair
                if low
                <= rest + handles[i].delta(Fraction(_inside(x, *boxes[i])))
                <= high
            ]
            if inside:
                values[i] = min(inside, key=lambda x: handles[i].cost(Fraction(x)))
                total = rest + handles[i].delta(Fraction(values[i]))
                break
        else:
            return None
    cost = sum(
        (h.cost(Fraction(x)) for h, x in zip(handles, values, strict=True)), Fraction(0)
    )
    return cost, values


def _travel(handle: _Handle, x: Fraction, increases_only: bool) -> Fraction:
    change = x - handle.x0
    return max(change, Fraction(0)) if increases_only else abs(change)


def _limited_witness(
    handles: list[_Handle],
    values: list[float],
    band: tuple[Fraction, Fraction],
    limits: list[_Limit],
) -> tuple[Fraction, list[float]] | None:
    """The witness with travel limits enforced, if it stays in the band."""
    fixed = _within_limits(handles, values, limits)
    if fixed is None:
        return None
    total = sum(
        (h.delta(Fraction(x)) for h, x in zip(handles, fixed, strict=True)),
        Fraction(0),
    )
    if not band[0] <= total <= band[1]:
        return None
    cost = sum(
        (h.cost(Fraction(x)) for h, x in zip(handles, fixed, strict=True)), Fraction(0)
    )
    return cost, fixed


def _tighten(
    handles: list[_Handle],
    boxes: list[tuple[Fraction, Fraction]],
    limits: list[_Limit],
) -> list[tuple[Fraction, Fraction]] | None:
    """Shrink a box by the travel limits; None when a limit excludes all of it."""
    tightened = list(boxes)
    for i, j, limit, increases_only in limits:
        least = [
            max(
                tightened[k][0] - handles[k].x0,
                Fraction(0) if increases_only else handles[k].x0 - tightened[k][1],
                Fraction(0),
            )
            for k in (i, j)
        ]
        if least[0] + least[1] > limit:
            return None
        for k, other in ((i, least[1]), (j, least[0])):
            reach = limit - other
            lo, hi = tightened[k]
            tightened[k] = (
                lo if increases_only else max(lo, handles[k].x0 - reach),
                min(hi, handles[k].x0 + reach),
            )
    return tightened


def _over_limit(
    handles: list[_Handle],
    values: Sequence[Fraction],
    limits: list[_Limit],
) -> _Limit | None:
    for limit in limits:
        i, j, most, increases_only = limit
        if (
            _travel(handles[i], values[i], increases_only)
            + _travel(handles[j], values[j], increases_only)
            > most
        ):
            return limit
    return None


def _within_limits(
    handles: list[_Handle],
    values: list[float],
    limits: list[_Limit],
) -> list[float] | None:
    """Pull floats toward their references until every travel limit holds."""
    fixed = list(values)
    for _ in range(len(limits) + 1):
        exact = [Fraction(x) for x in fixed]
        over = _over_limit(handles, exact, limits)
        if over is None:
            return fixed
        i, j, most, increases_only = over
        travels = [
            exact[k] - handles[k].x0
            if _travel(handles[k], exact[k], increases_only) > 0
            else Fraction(0)
            for k in (i, j)
        ]
        ratio = most / (abs(travels[0]) + abs(travels[1]))
        for k, travel in zip((i, j), travels, strict=True):
            if travel == 0:
                continue
            moved = float(handles[k].x0 + ratio * travel)
            # Rounding must not leave the travel longer than the scaled one.
            if abs(Fraction(moved) - handles[k].x0) > abs(ratio * travel):
                moved = math.nextafter(moved, handles[k].law.reference)
            fixed[k] = moved
    return None


def _range(handle: _Handle) -> tuple[Fraction, Fraction]:
    ends = (handle.delta(handle.lo), handle.delta(handle.hi), Fraction(0))
    return min(ends), max(ends)


def _check_monotone(law: DimensionLaw) -> None:
    for pi in (_PI_LOWER, _PI_UPPER):
        slope = [
            r + pi * q for r, q in zip(law.rational, law.pi_coefficients, strict=True)
        ]
        ends = [slope[1] + 2 * slope[2] * Fraction(x) for x in (law.lower, law.upper)]
        if ends[0] * ends[1] < 0:
            raise ValueError(f"Law of {law.label} is not monotone on its interval")


def minimal_volume_change(
    laws: Sequence[DimensionLaw],
    volume_change: tuple[float, float],
    *,
    travel_limits: Sequence[TravelLimit] = (),
    tolerance: float | None = None,
    max_nodes: int = 10_000,
) -> DimensionQueryResult:
    """Find the cheapest weighted change of handle values reaching a volume band.

    ``volume_change`` is the band ``[low, high]`` for the summed volume
    contributions, in input units cubed. The cost is
    ``sum(weight * abs(x - reference))`` over handles, each kept in its
    ``[lower, upper]``. A witness is a float vector checked exactly against
    the band for every pi in the enclosure. Lower bounds come from exact
    convex hulls of each handle's (volume, cost) curve, so the returned
    lower bound holds for every handle vector in the box. ``tolerance`` is
    the accepted absolute gap between witness cost and lower bound; it
    defaults to 1e-9 times the total weighted interval width.

    ``travel_limits`` couple pairs of handles. A witness satisfies them exactly,
    and lower bounds and infeasibility hold over the handle vectors that do.
    Closing the gap on an active limit takes more nodes, so it can end
    ``unresolved`` within ``max_nodes``.

    A minimal witness usually lies on an edge of the band, and rebuilt CAD
    matches the law only within its validation bound; narrow the band by
    that allowance when the rebuilt part itself must lie in it.

    Raises ValueError for an empty or inverted band, a law that is not
    monotone on its interval, a reference outside its interval, or a travel
    limit that is negative, names one handle twice, or names a label that does
    not identify exactly one law.
    """
    if not laws:
        raise ValueError("Expected at least one law")
    low, high = (Fraction(volume_change[0]), Fraction(volume_change[1]))
    if not low <= high:
        raise ValueError("Volume change band must satisfy low <= high")
    for law in laws:
        if not law.lower <= law.reference <= law.upper or law.weight <= 0:
            raise ValueError(
                f"{law.label}: reference must lie in [lower, upper] "
                "with positive weight"
            )
        _check_monotone(law)
    handles = [_Handle(law) for law in laws]
    positions = {law.label: i for i, law in enumerate(laws)}
    labels = [law.label for law in laws]
    limits = []
    for travel in travel_limits:
        if (
            labels.count(travel.first) != 1
            or labels.count(travel.second) != 1
            or travel.first == travel.second
            or not travel.limit >= 0
        ):
            raise ValueError(f"Unsupported travel limit: {travel}")
        limits.append(
            (
                positions[travel.first],
                positions[travel.second],
                Fraction(travel.limit),
                travel.increases_only,
            )
        )
    spread = sum((h.pi_spread() for h in handles), Fraction(0))
    ranges = [_range(h) for h in handles]
    reachable = (
        sum((lo for lo, _ in ranges), Fraction(0)) - spread,
        sum((hi for _, hi in ranges), Fraction(0)) + spread,
    )
    if reachable[1] < low or reachable[0] > high:
        return DimensionQueryResult("infeasible", None, None, None, reachable, None, 0)
    bound_band = (low - spread, high + spread)
    witness_band = (low + spread, high - spread)
    gap = Fraction(
        tolerance
        if tolerance is not None
        else 1e-9 * sum(law.weight * (law.upper - law.lower) for law in laws)
    )
    counter = itertools.count()
    root = [(h.lo, h.hi) for h in handles]
    heap: list[tuple[Fraction, int, list[tuple[Fraction, Fraction]]]] = []
    tightened = _tighten(handles, root, limits)
    relaxed = None if tightened is None else _relax(handles, tightened, bound_band)
    if tightened is None or relaxed is None:
        return DimensionQueryResult("infeasible", None, None, None, reachable, None, 0)
    heapq.heappush(heap, (relaxed[0], next(counter), tightened))
    best: tuple[Fraction, list[float]] | None = None
    # Bounds of boxes that could be neither split nor resolved, and of boxes
    # dropped because they could not improve the incumbent by more than the
    # gap; both still bound the cost of the solutions they contain.
    set_aside: list[Fraction] = []
    nodes = 0
    status: Literal["optimal_within_tolerance", "infeasible", "unresolved"]
    while True:
        if not heap:
            status = "infeasible" if best is None else "optimal_within_tolerance"
            lower_bound = None if best is None else best[0]
            break
        bound, _, boxes = heapq.heappop(heap)
        if best is not None and bound >= best[0] - gap:
            status, lower_bound = "optimal_within_tolerance", min(bound, best[0])
            break
        if nodes >= max_nodes:
            status, lower_bound = "unresolved", bound
            break
        nodes += 1
        relaxed = _relax(handles, boxes, bound_band)
        if relaxed is None:
            continue
        _, deltas = relaxed
        if witness_band[0] <= witness_band[1]:
            found = _witness(handles, boxes, deltas, witness_band)
            if found is not None:
                found = _limited_witness(handles, found[1], witness_band, limits)
            if found is not None and (best is None or found[0] < best[0]):
                best = found
        if best is not None and bound >= best[0] - gap:
            set_aside.append(bound)
            continue
        points = [
            Fraction(_float_value(h, lo, hi, d))
            for h, (lo, hi), d in zip(handles, boxes, deltas, strict=True)
        ]
        gaps = [
            h.cost(x) - _hull_cost(h, box, d)
            for h, box, x, d in zip(handles, boxes, points, deltas, strict=True)
        ]
        over = _over_limit(handles, points, limits)
        if over is None:
            j = int(np.argmax([float(g) for g in gaps]))
        else:
            # The relaxation ignored this limit: only narrowing its handles
            # brings the bound up to it, whatever the hull gaps are.
            j = max(over[:2], key=lambda k: boxes[k][1] - boxes[k][0])
        lo, hi = boxes[j]
        cut = points[j] if lo < points[j] < hi else (lo + hi) / 2
        if not lo < cut < hi:
            set_aside.append(bound)
            continue
        for part in ((lo, cut), (cut, hi)):
            child = list(boxes)
            child[j] = part
            narrowed = _tighten(handles, child, limits)
            if narrowed is None:
                continue
            child = narrowed
            relaxed_child = _relax(handles, child, bound_band)
            if relaxed_child is None:
                continue
            if best is None or relaxed_child[0] < best[0] - gap:
                heapq.heappush(heap, (relaxed_child[0], next(counter), child))
            else:
                set_aside.append(relaxed_child[0])
    if set_aside:
        floor = min(set_aside)
        lower_bound = floor if lower_bound is None else min(lower_bound, floor)
        if best is None or floor < best[0] - gap:
            status = "unresolved"
    if best is None:
        return DimensionQueryResult(
            status, None, None, lower_bound, reachable, None, nodes
        )
    values = tuple(best[1])
    change = [
        sum(
            (
                _polynomial(law.rational, Fraction(x))
                - _polynomial(law.rational, Fraction(law.reference))
                + pi
                * (
                    _polynomial(law.pi_coefficients, Fraction(x))
                    - _polynomial(law.pi_coefficients, Fraction(law.reference))
                )
                for law, x in zip(laws, values, strict=True)
            ),
            Fraction(0),
        )
        for pi in (_PI_LOWER, _PI_UPPER)
    ]
    return DimensionQueryResult(
        status,
        values,
        best[0],
        lower_bound,
        reachable,
        (min(change), max(change)),
        nodes,
    )


def face_offset_laws(
    edit: PreparedFaceOffset, weights: Sequence[float] | None = None
) -> tuple[DimensionLaw, ...]:
    """Volume laws ``area * offset`` of a prepared face offset, in input units."""
    weights = [1.0] * len(edit.faces) if weights is None else list(weights)
    if len(weights) != len(edit.faces):
        raise ValueError("Expected one weight per face")
    zero = Fraction(0)
    return tuple(
        DimensionLaw(
            label=f"face {face.face_id} offset",
            reference=0.0,
            lower=face.lower,
            upper=face.upper,
            weight=weight,
            rational=(zero, Fraction(face.area), zero),
            pi_coefficients=(zero, zero, zero),
        )
        for face, weight in zip(edit.faces, weights, strict=True)
    )


def _hole_radius_laws(
    edit: PreparedHoleEdit, weights: Sequence[float] | None = None
) -> tuple[DimensionLaw, ...]:
    """Volume laws ``-pi * height * (r**2 - r0**2)`` of prepared holes.

    Radii range over the prepared radius bounds with zero center shift. Those
    bounds hold one hole at a time: pairs need the limits from
    :func:`hole_ligament_constraints`.
    """
    weights = [1.0] * len(edit.holes) if weights is None else list(weights)
    if len(weights) != len(edit.holes):
        raise ValueError("Expected one weight per hole")
    zero = Fraction(0)
    lower = np.asarray(edit.model.lower_radii, dtype=np.float64)
    upper = np.asarray(edit.model.upper_radii, dtype=np.float64)
    return tuple(
        DimensionLaw(
            label=f"hole {hole.face_id} radius",
            reference=hole.radius,
            lower=float(lo),
            upper=float(hi),
            weight=weight,
            rational=(zero, zero, zero),
            pi_coefficients=(zero, zero, -Fraction(hole.height)),
        )
        for hole, lo, hi, weight in zip(edit.holes, lower, upper, weights, strict=True)
    )


def hole_ligament_constraints(
    edit: PreparedHoleEdit,
    minimum_ligament: float | None = None,
    weights: Sequence[float] | None = None,
) -> tuple[tuple[DimensionLaw, ...], tuple[TravelLimit, ...]]:
    """Hole radius laws and travel limits for :func:`minimal_volume_change`.

    Radius intervals alone admit holes that would meet, so pass both results.
    Each pair of selected holes that could otherwise come too close gets a
    :class:`TravelLimit`.

    By default the allowed values are exactly the radii ``is_valid`` accepts:
    they keep the laws true, with walls down to the model's ``contact_margin``.
    Shrinking a hole is then free, since it cannot bring two holes into contact.

    With ``minimum_ligament``, in input units, any allowed values also keep the
    ligament bound at or above it in exact arithmetic; evaluating
    ``HoleEditModel.ligament_bounds`` in floats can round it by an ulp. Each
    radius interval narrows by its distance to fixed faces, and shrinking counts
    as travel, as it does in that bound. Infeasibility then concerns the bound,
    which is conservative except where a hole grows toward a radially closest
    feature.

    Raises ValueError for a negative minimum, or one a hole's bound already
    violates at its reference radius.
    """
    if minimum_ligament is not None and not minimum_ligament >= 0:
        raise ValueError("Minimum ligament must not be negative")
    laws = _hole_radius_laws(edit, weights)
    fixed = np.asarray(edit.model.fixed_clearance, dtype=np.float64)
    to_holes = np.asarray(edit.model.hole_clearance, dtype=np.float64)
    margin = float(edit.model.contact_margin)
    requested = 0.0 if minimum_ligament is None else minimum_ligament
    minimum = Fraction(max(requested, margin))
    narrowed = []
    for i, law in enumerate(laws):
        closest = min(float(fixed[i]), float(np.min(to_holes[i])))
        if closest < requested:
            raise ValueError(
                f"{law.label}: its ligament bound {closest} is below the minimum"
            )
        if minimum_ligament is None or np.isinf(fixed[i]):
            narrowed.append(law)
            continue
        reach = Fraction(float(fixed[i])) - minimum
        reference = Fraction(law.reference)
        narrowed.append(
            replace(
                law,
                lower=max(law.lower, _float_above(reference - reach)),
                upper=min(law.upper, _float_below(reference + reach)),
            )
        )
    limits = []
    for i in range(len(laws)):
        for j in range(i + 1, len(laws)):
            # Both holes use the smaller measured distance: a tolerance can
            # make the two OCCT queries of one pair differ slightly.
            closest = min(float(to_holes[i, j]), float(to_holes[j, i]))
            if np.isinf(closest):
                continue
            room = Fraction(closest) - minimum
            widest = sum(
                (
                    Fraction(law.upper) - Fraction(law.reference)
                    if minimum_ligament is None
                    else max(
                        Fraction(law.upper) - Fraction(law.reference),
                        Fraction(law.reference) - Fraction(law.lower),
                    )
                    for law in (narrowed[i], narrowed[j])
                ),
                Fraction(0),
            )
            if widest > room:
                limits.append(
                    TravelLimit(
                        laws[i].label,
                        laws[j].label,
                        _float_below(room),
                        increases_only=minimum_ligament is None,
                    )
                )
    return tuple(narrowed), tuple(limits)


def _float_below(value: Fraction) -> float:
    """The largest float not above ``value``."""
    rounded = float(value)
    return math.nextafter(rounded, -math.inf) if Fraction(rounded) > value else rounded


def _float_above(value: Fraction) -> float:
    """The smallest float not below ``value``."""
    rounded = float(value)
    return math.nextafter(rounded, math.inf) if Fraction(rounded) < value else rounded


__all__ = [
    "DimensionLaw",
    "DimensionQueryResult",
    "TravelLimit",
    "face_offset_laws",
    "hole_ligament_constraints",
    "minimal_volume_change",
]
