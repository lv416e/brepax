"""Exact one-parameter continuation through certified isolated event exchanges."""

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from heapq import heappop, heappush
from itertools import combinations
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from brepax.experimental._polyhedral_boundary import (
    _closure_profile,
    _equal_profiles,
    _interval,
    _prepare_side,
    _Side,
)
from brepax.experimental._polyhedral_parameter import (
    Affine,
    ParameterGuard,
    _Cell,
    _cell,
    affine_value,
)
from brepax.experimental._polyhedral_translation import (
    ExactPiece,
    Plane,
    Vector,
    _dot,
    collection_pieces_from_planes,
)

_CellArrays = tuple[NDArray[np.float64], NDArray[np.float64]]


@dataclass(frozen=True)
class ContinuationChart:
    """One exact chart and its open global-parameter validity interval."""

    side: _Side
    interval: tuple[Fraction | None, Fraction | None]


@dataclass(frozen=True)
class ContinuationBoundary:
    """A separately certified point joining two regular chart intervals."""

    parameter: Fraction
    events: tuple[Affine, Affine]
    pieces: tuple[ExactPiece, ...]


@dataclass(frozen=True)
class ContinuationGap:
    """Unconfirmed parameter coverage, not proof of invalid geometry or volume."""

    lower: Fraction
    upper: Fraction
    lower_closed: bool
    upper_closed: bool
    reason: str


@dataclass(frozen=True)
class ContinuationAttempt:
    """A retained seed, center trial, or unsupported frontier observation."""

    center: Fraction
    boundary: Fraction | None
    status: Literal[
        "accepted_seed",
        "accepted_exchange",
        "seed_unresolved",
        "frontier_unresolved",
        "trial_unresolved",
    ]
    message: str


@dataclass(frozen=True)
class ExactContinuation:
    """Certified open charts, included boundary points, and unconfirmed coverage."""

    domain: tuple[Fraction, Fraction]
    charts: tuple[ContinuationChart, ...]
    boundaries: tuple[ContinuationBoundary, ...]
    uncovered: tuple[ContinuationGap, ...]
    attempts: tuple[ContinuationAttempt, ...]

    def specialize(self, parameter: Fraction) -> tuple[ExactPiece, ...]:
        if not self.domain[0] <= parameter <= self.domain[1]:
            raise ValueError("Parameter is outside the requested closed domain")
        for boundary in self.boundaries:
            if parameter == boundary.parameter:
                return boundary.pieces
        for chart in self.charts:
            lo, hi = chart.interval
            if (lo is None or parameter > lo) and (hi is None or parameter < hi):
                return chart.side.geometry.specialize((parameter - chart.side.center,))
        raise ValueError("Parameter is not covered by a chart or certified point")


@dataclass(frozen=True)
class _Family:
    fixed: tuple[_Cell, ...]
    moving: tuple[_Cell, ...]
    velocity: Vector
    maximum: Fraction
    cell_guards: frozenset[ParameterGuard]

    def chart(self, center: Fraction) -> ContinuationChart:
        side = _prepare_side(
            self.fixed,
            self.moving,
            self.velocity,
            self.maximum,
            set(self.cell_guards),
            center,
        )
        return ContinuationChart(side, _interval(side.guards))

    def profile(self, parameter: Fraction) -> tuple[ExactPiece, ...]:
        """Independent static profile after validating the cell/separation guards.

        Do not reuse _rebase_cell's original hull coordinates or round rational
        rebased vertices. Original oriented planes retain the same affine family.
        """

        def planes(cell: _Cell, velocity: Vector) -> tuple[Plane, ...]:
            return tuple(
                (normal, affine_value(offset, (parameter,)), _dot(normal, velocity))
                for normal, offset in cell.planes
            )

        return collection_pieces_from_planes(
            tuple(planes(cell, (Fraction(0),) * 3) for cell in self.fixed),
            tuple(planes(cell, self.velocity) for cell in self.moving),
            self.maximum,
        )


def _classify(
    side: _Side, parameter: Fraction, maximum: Fraction
) -> tuple[tuple[Affine, Affine] | None, str]:
    values = [(g, affine_value(g.expression, (parameter,))) for g in side.guards]
    if any(value < 0 for _, value in values):
        raise ValueError("Requested frontier is outside the chart closure")
    vanished = [guard for guard, value in values if value == 0]
    kinds = sorted({guard.kind for guard in vanished if guard.kind != "event"})
    if kinds:
        return None, "non_event_guard:" + ",".join(kinds)
    pairs = {
        tuple(sorted(guard.operands))
        for guard in vanished
        if len(guard.operands) == 2 and guard.operands[0] != guard.operands[1]
    }
    if len(pairs) != 1 or any(len(guard.operands) != 2 for guard in vanished):
        return None, "multiple_or_untracked_event_guards"
    candidate = next(iter(pairs))
    pair = candidate[0], candidate[1]
    ties = [
        (a, b)
        for a, b in combinations(sorted(set(side.events)), 2)
        if affine_value(a, (parameter,)) == affine_value(b, (parameter,))
    ]
    if len(ties) != 1 or set(ties[0]) != set(pair):
        return None, "multiple_or_untracked_profile_events"
    if not 0 < affine_value(pair[0], (parameter,)) < maximum:
        return None, "translation_endpoint_event"
    return pair, "isolated_event_exchange"


@dataclass(frozen=True)
class _Exchange:
    chart: ContinuationChart | None
    boundary: ContinuationBoundary | None
    attempts: tuple[ContinuationAttempt, ...]


def _extend(
    family: _Family,
    chart: ContinuationChart,
    parameter: Fraction,
    pair: tuple[Affine, Affine],
    direction: int,
    limit: Fraction,
    max_center_trials: int,
) -> _Exchange:
    distances = [abs(parameter - chart.side.center), abs(limit - parameter)]
    for guard in chart.side.guards:
        a, b = guard.expression
        if b and direction * (-a / b - parameter) > 0:
            distances.append(abs(-a / b - parameter))
    radius = min(distances) / 2
    attempts = []
    for trial in range(max_center_trials):
        center = parameter + direction * radius / 2**trial
        try:
            candidate = family.chart(center)
            side = candidate.side
            if candidate.interval[0 if direction > 0 else 1] != parameter:
                raise ValueError(
                    f"Trial chart does not adjoin boundary: {candidate.interval}"
                )
            other_pair, reason = _classify(side, parameter, family.maximum)
            if other_pair != pair:
                raise ValueError(f"Trial boundary classification changed: {reason}")
            if set(side.events) != set(chart.side.events):
                raise ValueError("Affine event inventory changed")
            before = _closure_profile(chart.side, parameter)
            after = _closure_profile(side, parameter)
            baseline = family.profile(parameter)
            if not _equal_profiles(before, after) or not _equal_profiles(
                before, baseline
            ):
                raise ValueError("Closures disagree with the exact static profile")
            attempts.append(
                ContinuationAttempt(center, parameter, "accepted_exchange", "")
            )
            return _Exchange(
                candidate,
                ContinuationBoundary(parameter, pair, baseline),
                tuple(attempts),
            )
        except (ValueError, OverflowError) as error:
            attempts.append(
                ContinuationAttempt(center, parameter, "trial_unresolved", str(error))
            )
    return _Exchange(None, None, tuple(attempts))


def prepare_continuation(
    fixed: Sequence[_CellArrays],
    moving: Sequence[_CellArrays],
    velocity: Vector,
    maximum: Fraction,
    domain: tuple[Fraction, Fraction],
    seed: Fraction,
    max_charts: int,
    max_center_trials: int,
) -> ExactContinuation:
    """Continue a regular seed within validated bounds and finite work budgets."""
    attempts: list[ContinuationAttempt] = []
    cell_guards: set[ParameterGuard] = set()
    guards: set[Affine] = set()
    try:
        family = _Family(
            tuple(_cell(v, d, guards, cell_guards) for v, d in fixed),
            tuple(_cell(v, d, guards, cell_guards) for v, d in moving),
            velocity,
            maximum,
            frozenset(cell_guards),
        )
        initial = family.chart(seed)
    except (ValueError, OverflowError) as error:
        attempt = ContinuationAttempt(seed, None, "seed_unresolved", str(error))
        return ExactContinuation(
            domain,
            (),
            (),
            (ContinuationGap(*domain, True, True, "seed_preparation_failed"),),
            (attempt,),
        )
    charts = [initial]
    boundaries = []
    attempts.append(ContinuationAttempt(seed, None, "accepted_seed", ""))
    queue: list[tuple[Fraction, int, int]] = []
    stopped: dict[int, str] = {}

    def enqueue(chart: ContinuationChart, direction: int) -> None:
        endpoint = chart.interval[0 if direction < 0 else 1]
        limit = domain[0 if direction < 0 else 1]
        if endpoint is not None and direction * (endpoint - limit) <= 0:
            heappush(queue, (abs(endpoint - seed), direction, len(charts) - 1))

    enqueue(initial, -1)
    enqueue(initial, 1)
    while queue:
        _, direction, index = heappop(queue)
        chart = charts[index]
        parameter = chart.interval[0 if direction < 0 else 1]
        assert parameter is not None
        limit = domain[0 if direction < 0 else 1]
        pair, reason = _classify(chart.side, parameter, maximum)
        if pair is not None:
            if len(charts) >= max_charts:
                reason = "chart_budget"
            elif parameter == limit:
                reason = "window_endpoint_event"
            else:
                exchange = _extend(
                    family, chart, parameter, pair, direction, limit, max_center_trials
                )
                attempts.extend(exchange.attempts)
                if exchange.chart is not None and exchange.boundary is not None:
                    charts.append(exchange.chart)
                    boundaries.append(exchange.boundary)
                    enqueue(exchange.chart, direction)
                    continue
                reason = "center_trials_exhausted"
        stopped[direction] = reason
        attempts.append(
            ContinuationAttempt(
                chart.side.center, parameter, "frontier_unresolved", reason
            )
        )

    left = min((c.interval[0] for c in charts), key=lambda x: (x is not None, x))
    right = max((c.interval[1] for c in charts), key=lambda x: (x is None, x))
    uncovered = []
    # Accepted charts form one connected seed component. Its uncertified outer
    # endpoints remain in the gaps, even when a gap is only one rational point.
    if left is not None and domain[0] <= left:
        uncovered.append(
            ContinuationGap(
                domain[0],
                min(left, domain[1]),
                True,
                True,
                stopped.get(-1, "uncovered_frontier"),
            )
        )
    if right is not None and right <= domain[1]:
        uncovered.append(
            ContinuationGap(
                max(right, domain[0]),
                domain[1],
                True,
                True,
                stopped.get(1, "uncovered_frontier"),
            )
        )
    return ExactContinuation(
        domain,
        tuple(charts),
        tuple(sorted(boundaries, key=lambda b: b.parameter)),
        tuple(uncovered),
        tuple(attempts),
    )
