"""Bounded design search over certified polyhedral parameter regions."""

import math
from dataclasses import dataclass
from fractions import Fraction
from heapq import heappop, heappush
from itertools import pairwise
from typing import Literal

import jax.numpy as jnp
import numpy as np

from brepax.experimental._polyhedral_boundary import _interval, _Side
from brepax.experimental._polyhedral_continuation import (
    ContinuationBoundary,
    ContinuationChart,
    ExactContinuation,
)
from brepax.experimental._polyhedral_design import interval_volume_bound
from brepax.experimental._polyhedral_roots import value_bounds
from brepax.experimental.polyhedral_boundary import PreparedPolyhedralEventBoundary
from brepax.experimental.polyhedral_continuation import PreparedPolyhedralContinuation
from brepax.experimental.polyhedral_targets import (
    TranslationTargetResult,
    TranslationVolumeRange,
    _rational,
    _volume_at,
    polyhedral_translation_range,
    polyhedral_translation_targets,
)
from brepax.experimental.polyhedral_translation import (
    PreparedPolyhedralTranslation,
    _prepared_from_pieces,
)

_Scalar = float | int | Fraction


@dataclass(frozen=True)
class DesignTargetWitness:
    """An executable binary64 pair, checked in exact represented-model arithmetic."""

    parameter: float
    step: float
    volume: Fraction
    residual: Fraction
    objective: Fraction


@dataclass(frozen=True)
class DesignQueryFailure:
    """A valid exact parameter whose existing numerical profile could not be packed."""

    parameter: Fraction
    message: str


@dataclass(frozen=True)
class DesignTargetResult:
    """A bounded search certificate, not a complete design-feasibility atlas.

    The lower objective bound concerns continuous real parameter/step pairs in
    the queried closed domain and volume band. The upper bound belongs to the
    returned binary64 witness, whose volume and residual are exact rational
    model values. ``infeasible`` excludes the entire continuous domain;
    ``unresolved`` retains regions not excluded within the search budget.
    No status certifies source CAD, an exact central-target root, or a gradient.
    """

    target: Fraction
    witness: DesignTargetWitness | None
    objective_lower: Fraction | None
    objective_upper: Fraction | None
    status: Literal["optimal_within_tolerance", "infeasible", "unresolved"]
    remaining_intervals: tuple[tuple[Fraction, Fraction], ...]
    boxes_examined: int
    point_queries: int
    query_failures: tuple[DesignQueryFailure, ...]

    @property
    def optimality_gap(self) -> Fraction | None:
        """Certified objective gap when an executable witness is available."""
        if self.objective_upper is None or self.objective_lower is None:
            return None
        return self.objective_upper - self.objective_lower


def _float_neighbors(value: Fraction) -> tuple[float, ...]:
    try:
        rounded = float(value)
    except OverflowError:
        return ()
    return tuple(
        x
        for x in (
            rounded,
            math.nextafter(rounded, -math.inf),
            math.nextafter(rounded, math.inf),
        )
        if math.isfinite(x)
    )


def _target_steps(result: TranslationTargetResult) -> list[float]:
    return [root.step for root in result.roots if root.step is not None] + [
        interval.step for interval in result.intervals if interval.step is not None
    ]


def polyhedral_design_target(
    prepared: PreparedPolyhedralEventBoundary | PreparedPolyhedralContinuation,
    target: _Scalar,
    *,
    parameter_bounds: tuple[_Scalar, _Scalar],
    initial_parameter: _Scalar,
    parameter_tolerance: _Scalar,
    step_tolerance: _Scalar,
    volume_tolerance: _Scalar,
    max_boxes: int = 256,
    max_refinements: int = 128,
) -> DesignTargetResult:
    """Minimize design change while finding a target-volume translation action.

    For a supplied event boundary, search a finite closed parameter interval
    strictly inside its open certificate. For automatic continuation, search
    inside its requested closed domain, retaining uncovered regions. The
    objective is ``abs(theta-initial_parameter)``; the constraint is
    ``abs(V(step,theta)-target) <= volume_tolerance``. This volume
    tolerance defines a feasible band, unlike the exact-existence semantics of
    ``polyhedral_translation_targets``. Both returned coordinates are binary64
    values, rechecked as exact rationals against the original band and domain.

    Parameters and step are dimensionless; target/tolerance have volume units.
    Scale both by s**3 when scaling geometry, displacements, and velocity by s.
    Exact coordinate-frame laws and represented-input limitations of the
    prepared model apply. This host query is neither JAX-transformable nor
    a derivative of target selection, and retains existing JAX x64/profile
    packing requirements. Failed numerical packing remains explicitly recorded.

    ``parameter_tolerance`` is the allowed objective gap for an
    ``optimal_within_tolerance`` result. ``step_tolerance`` controls candidate-root
    refinement, not a separate feasibility constraint.
    ``max_boxes`` caps examined design intervals and isolated points across
    all charts; ``max_refinements`` caps each existing root-isolation path.
    Initial candidate queries at domain/chart endpoints are additional work,
    reported by ``point_queries``. Preparation has its own chart/attempt limits.
    Fixed range tolerances can leave bound floors; isolated feasible points,
    unavailable float actions, or exhausted budgets
    may remain unresolved. No finite-completeness guarantee is made.

    Every uncovered region, including an isolated boundary point, contributes
    its distance from the initial parameter to the global objective lower
    bound. Incomplete coverage can still certify a gap if all uncovered regions
    are sufficiently far away; it cannot establish global infeasibility.
    ``remaining_intervals`` contains conservative closed envelopes of regions
    not excluded, including uncovered ones. Exact coverage membership and stop
    reasons remain on the prepared continuation, separately from this status.

    Example::

        result = polyhedral_design_target(
            boundary, 0.2, parameter_bounds=(-0.1, 0.1),
            initial_parameter=0.08, parameter_tolerance=1e-4,
            step_tolerance=1e-10, volume_tolerance=1e-8,
        )
        if result.witness is not None:
            # Independently validate this model action with the actual CAD.
            theta, step = result.witness.parameter, result.witness.step
    """
    if jnp.asarray(0.0).dtype != np.dtype("float64"):
        raise ValueError("Polyhedral design search requires JAX x64")
    q = _rational(target, "target")
    lo, hi = (_rational(x, "parameter bound") for x in parameter_bounds)
    initial = _rational(initial_parameter, "initial_parameter")
    design_tol = _rational(parameter_tolerance, "parameter_tolerance")
    step_tol = _rational(step_tolerance, "step_tolerance")
    volume_tol = _rational(volume_tolerance, "volume_tolerance")
    if q < 0 or volume_tol < 0 or design_tol <= 0 or step_tol <= 0:
        raise ValueError(
            "Target/volume tolerance must be nonnegative; search tolerances positive"
        )
    for name, limit in (("max_boxes", max_boxes), ("max_refinements", max_refinements)):
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError(f"{name} must be a positive integer")
    geometry = _search_domain(prepared, lo, hi)
    segments, uncovered, cuts = _partition(geometry, lo, hi)
    pair_count = geometry.charts[0].side.geometry.pair_count if geometry.charts else 0
    band_lower = max(Fraction(0), q - volume_tol)
    paths: dict[Fraction, PreparedPolyhedralTranslation | None] = {}
    ranges: dict[Fraction, TranslationVolumeRange | None] = {}
    attempted: set[Fraction] = set()
    failures: list[DesignQueryFailure] = []
    witness: DesignTargetWitness | None = None

    def distance(a: Fraction, b: Fraction) -> Fraction:
        return max(Fraction(0), a - initial, initial - b)

    def path_at(theta: Fraction) -> PreparedPolyhedralTranslation | None:
        if theta not in paths:
            # Certificate errors are not numerical packing failures.
            pieces = geometry.specialize(theta)
            try:
                paths[theta] = _prepared_from_pieces(pieces, pair_count)
            except (ValueError, OverflowError) as error:
                paths[theta] = None
                failures.append(DesignQueryFailure(theta, str(error)))
        return paths[theta]

    def range_at(theta: Fraction) -> TranslationVolumeRange | None:
        if theta not in ranges:
            path = path_at(theta)
            ranges[theta] = (
                None
                if path is None
                else polyhedral_translation_range(
                    path,
                    step_tolerance=step_tol,
                    volume_tolerance=volume_tol,
                    max_refinements=max_refinements,
                )
            )
        return ranges[theta]

    def consider(theta: Fraction, steps: list[float]) -> bool:
        nonlocal witness
        pieces = geometry.specialize(theta)
        found = False
        for step in sorted(set(steps)):
            if (
                not math.isfinite(step)
                or not pieces[0].lower <= Fraction(step) <= pieces[-1].upper
            ):
                continue
            volume = _volume_at(pieces, Fraction(step))
            residual = volume - q
            if abs(residual) <= volume_tol:
                found = True
                candidate = DesignTargetWitness(
                    float(theta), step, volume, residual, abs(theta - initial)
                )
                if witness is None or (
                    candidate.objective,
                    candidate.parameter,
                    candidate.step,
                ) < (witness.objective, witness.parameter, witness.step):
                    witness = candidate
        return found

    def try_parameter(value: Fraction) -> None:
        for rounded in _float_neighbors(value):
            theta = Fraction(rounded)
            if (
                not lo <= theta <= hi
                or theta in attempted
                or not _covered(geometry, theta)
            ):
                continue
            attempted.add(theta)
            if band_lower == 0:
                consider(theta, [0.0])
                continue
            path = path_at(theta)
            if path is None:
                continue
            result = polyhedral_translation_targets(
                path,
                q,
                step_tolerance=step_tol,
                volume_tolerance=volume_tol,
                max_refinements=max_refinements,
            )
            if consider(theta, _target_steps(result)):
                continue
            attainable = range_at(theta)
            assert attainable is not None
            steps = [
                step
                for candidate in attainable.candidates
                for position in (
                    candidate.lower,
                    candidate.upper,
                    (candidate.lower + candidate.upper) / 2,
                )
                for step in _float_neighbors(position)
            ]
            if consider(theta, steps):
                continue
            # A central target can be unreachable while its allowed band is
            # reachable. Query an interior level proved below the maximum.
            if attainable.maximum[0] > band_lower:
                level = min(q, (band_lower + attainable.maximum[0]) / 2)
                margin = min(level - band_lower, q + volume_tol - level)
                interior = polyhedral_translation_targets(
                    path,
                    level,
                    step_tolerance=step_tol,
                    volume_tolerance=margin,
                    max_refinements=max_refinements,
                )
                consider(theta, _target_steps(interior))

    try_parameter(min(hi, max(lo, initial)))
    for theta in cuts:
        try_parameter(theta)
    queue: list[tuple[Fraction, Fraction, Fraction, int]] = []
    for index, region in enumerate(segments):
        heappush(
            queue,
            (distance(region.lower, region.upper), region.lower, region.upper, index),
        )
    examined = 0
    retained_points: list[tuple[Fraction, Fraction]] = []
    while queue and examined < max_boxes and band_lower > 0:
        if witness is not None and witness.objective - queue[0][0] <= design_tol:
            break
        lower_objective, a, b, index = heappop(queue)
        examined += 1
        if witness is not None and lower_objective >= witness.objective:
            continue
        midpoint = (a + b) / 2
        attainable = range_at(midpoint)
        side = segments[index].side
        if side is None:
            upper_volume = (
                attainable.maximum[1]
                if attainable is not None
                else max(
                    value_bounds(p.coefficients, p.lower, p.upper)[1]
                    for p in geometry.specialize(midpoint)
                )
            )
        else:
            upper_volume = interval_volume_bound(
                side, a, b, None if attainable is None else attainable.maximum[1]
            )
        if upper_volume < band_lower:
            continue
        try_parameter(midpoint)
        if a == b:
            retained_points.append((a, b))
            continue
        for left, right in ((a, midpoint), (midpoint, b)):
            heappush(queue, (distance(left, right), left, right, index))
    remaining = tuple(
        sorted((*uncovered, *retained_points, *((a, b) for _, a, b, _ in queue)))
    )
    upper_objective = None if witness is None else witness.objective
    lower_candidates = [distance(a, b) for a, b in remaining]
    if upper_objective is not None:
        lower_candidates.append(upper_objective)
    result_lower = min(lower_candidates, default=None)
    status: Literal["optimal_within_tolerance", "infeasible", "unresolved"] = (
        "unresolved"
    )
    if (
        upper_objective is not None
        and result_lower is not None
        and upper_objective - result_lower <= design_tol
    ):
        status = "optimal_within_tolerance"
    elif witness is None and not remaining:
        status = "infeasible"
    return DesignTargetResult(
        q,
        witness,
        result_lower,
        upper_objective,
        status,
        remaining,
        examined,
        len(set(paths) | attempted),
        tuple(failures),
    )


@dataclass(frozen=True)
class _DesignInterval:
    lower: Fraction
    upper: Fraction
    side: _Side | None


def _search_domain(
    prepared: PreparedPolyhedralEventBoundary | PreparedPolyhedralContinuation,
    lower: Fraction,
    upper: Fraction,
) -> ExactContinuation:
    if isinstance(prepared, PreparedPolyhedralContinuation):
        lo, hi = prepared.parameter_bounds
        if lower > upper or lower < lo or upper > hi:
            raise ValueError("Search bounds must lie inside the continuation domain")
        return prepared._geometry
    outer_lower, outer_upper = prepared.parameter_interval
    if (
        lower > upper
        or (outer_lower is not None and lower <= outer_lower)
        or (outer_upper is not None and upper >= outer_upper)
    ):
        raise ValueError(
            "Closed parameter bounds must lie strictly inside the boundary certificate"
        )
    geometry = prepared._geometry
    return ExactContinuation(
        (lower, upper),
        tuple(
            ContinuationChart(side, _interval(side.guards))
            for side in (geometry.left, geometry.right)
        ),
        (ContinuationBoundary(Fraction(0), geometry.collision, geometry.pieces),),
        (),
        (),
    )


def _chart_at(
    geometry: ExactContinuation, parameter: Fraction
) -> ContinuationChart | None:
    return next(
        (
            chart
            for chart in geometry.charts
            if (chart.interval[0] is None or chart.interval[0] < parameter)
            and (chart.interval[1] is None or parameter < chart.interval[1])
        ),
        None,
    )


def _covered(geometry: ExactContinuation, parameter: Fraction) -> bool:
    return _chart_at(geometry, parameter) is not None or any(
        point.parameter == parameter for point in geometry.boundaries
    )


def _partition(
    geometry: ExactContinuation, lower: Fraction, upper: Fraction
) -> tuple[
    tuple[_DesignInterval, ...],
    tuple[tuple[Fraction, Fraction], ...],
    tuple[Fraction, ...],
]:
    cuts = sorted(
        {lower, upper}
        | {
            endpoint
            for chart in geometry.charts
            for endpoint in chart.interval
            if endpoint is not None and lower < endpoint < upper
        }
        | {
            point.parameter
            for point in geometry.boundaries
            if lower < point.parameter < upper
        }
    )
    known: list[_DesignInterval] = []
    unknown: list[tuple[Fraction, Fraction]] = []
    for lo, hi in pairwise(cuts):
        chart = _chart_at(geometry, (lo + hi) / 2)
        if chart is None:
            unknown.append((lo, hi))
        else:
            known.append(_DesignInterval(lo, hi, chart.side))
    # Open chart interiors do not include their endpoints. Each cut point
    # requires its own certificate, even when neighboring intervals are known.
    for parameter in cuts:
        if _covered(geometry, parameter):
            known.append(_DesignInterval(parameter, parameter, None))
        else:
            unknown.append((parameter, parameter))
    return tuple(known), tuple(unknown), tuple(cuts)
