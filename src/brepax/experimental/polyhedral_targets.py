"""Host target inversion and range bounds for exact finite translation profiles.

Existence is decided using rational polynomials, not rounded JAX coefficients
or sampled brackets. Float actions are approximate and have separately checked
step and volume errors. These results concern the represented vertex-hull path;
an actual CAD edit still requires independent validation.
"""

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal

from brepax.experimental._polyhedral_roots import (
    RootBracket,
    derivative,
    isolate_roots,
    trim,
    value_bounds,
)
from brepax.experimental._polyhedral_translation import ExactPiece
from brepax.experimental.polyhedral_translation import PreparedPolyhedralTranslation


@dataclass(frozen=True)
class TranslationTargetRoot:
    """An exact root enclosure and an independently assessed binary64 action.

    ``root_count`` counts distinct roots, not multiplicity. For a nonpoint
    unresolved cluster it counts roots strictly between the bounds; endpoint
    roots appear separately. A resolved nonpoint interval has nonroot endpoints
    and contains exactly one root. ``volume_residual`` is the exact represented
    path value at ``step`` minus the target, not a JAX or source-CAD error bound.
    """

    lower: Fraction
    upper: Fraction
    root_count: int
    step: float | None
    volume_residual: Fraction | None
    resolved: bool


@dataclass(frozen=True)
class TranslationTargetInterval:
    """A closed solution interval, with its earliest binary64 member if one exists."""

    lower: Fraction
    upper: Fraction
    step: float | None


@dataclass(frozen=True)
class TranslationTargetResult:
    """Complete exact root inventory, including unresolved numerical actions.

    ``unreachable`` means no exact solution on the supplied represented path.
    ``unresolved`` means exact solutions exist but one or more cannot be returned
    as distinct float actions meeting the requested tolerances/refinement cap.
    This does not certify the correspondence or accuracy of source CAD.
    """

    target: Fraction
    roots: tuple[TranslationTargetRoot, ...]
    intervals: tuple[TranslationTargetInterval, ...]
    exact_reachable: bool
    status: Literal["reachable", "unreachable", "unresolved"]


@dataclass(frozen=True)
class TranslationRangeCandidate:
    """Knot, stationary-root bracket, or constant interval enclosing extrema.

    ``volume_bounds`` encloses all values on ``[lower, upper]``. A candidate
    need not be a local extremum; every global extremum is covered by one.
    """

    lower: Fraction
    upper: Fraction
    volume_bounds: tuple[Fraction, Fraction]
    kind: Literal["knot", "stationary", "constant"]
    resolved: bool


@dataclass(frozen=True)
class TranslationVolumeRange:
    """Rational enclosures of the minimum and maximum on the prepared path.

    Each pair brackets one extremal value. Refinement failure leaves conservative
    enclosures and ``resolved=False``; target existence is assessed separately by
    :func:`polyhedral_translation_targets`, without relying on these bounds.
    """

    minimum: tuple[Fraction, Fraction]
    maximum: tuple[Fraction, Fraction]
    candidates: tuple[TranslationRangeCandidate, ...]
    resolved: bool


def _rational(value: float | int | Fraction, name: str) -> Fraction:
    if isinstance(value, (int, Fraction)):
        return Fraction(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return Fraction(float(value))


def _query_inputs(
    prepared: PreparedPolyhedralTranslation,
    step_tolerance: float,
    volume_tolerance: float,
    max_refinements: int,
) -> tuple[tuple[ExactPiece, ...], Fraction, Fraction]:
    if not prepared._exact_pieces:
        raise ValueError("Target queries require retained exact translation pieces")
    step_tol = _rational(step_tolerance, "step_tolerance")
    volume_tol = _rational(volume_tolerance, "volume_tolerance")
    if step_tol <= 0 or volume_tol < 0:
        raise ValueError(
            "Step tolerance must be positive; volume tolerance nonnegative"
        )
    if (
        not isinstance(max_refinements, int)
        or isinstance(max_refinements, bool)
        or max_refinements < 1
    ):
        raise ValueError("max_refinements must be a positive integer")
    return prepared._exact_pieces, step_tol, volume_tol


def _volume_at(pieces: tuple[ExactPiece, ...], step: Fraction) -> Fraction:
    for piece in reversed(pieces):
        if piece.lower <= step <= piece.upper:
            return piece.value(step)
    raise ValueError("Step lies outside the exact prepared path")


def _candidate(
    pieces: tuple[ExactPiece, ...],
    lower: Fraction,
    upper: Fraction,
    target: Fraction,
    step_tolerance: Fraction,
    volume_tolerance: Fraction,
) -> tuple[float | None, Fraction | None, bool]:
    middle = float((lower + upper) / 2)
    trials = (
        middle,
        math.nextafter(middle, -math.inf),
        math.nextafter(middle, math.inf),
    )
    candidates = []
    for step in trials:
        if not math.isfinite(step):
            continue
        exact_step = Fraction(step)
        if not pieces[0].lower <= exact_step <= pieces[-1].upper:
            continue
        if lower != upper and not lower <= exact_step <= upper:
            continue
        residual = _volume_at(pieces, exact_step) - target
        distance = max(abs(exact_step - lower), abs(exact_step - upper))
        candidates.append((abs(residual), distance, step, residual))
    if not candidates:
        return None, None, False
    error, distance, step, residual = min(candidates)
    return step, residual, error <= volume_tolerance and distance <= step_tolerance


def _solution_intervals(
    pieces: tuple[ExactPiece, ...], target: Fraction
) -> tuple[TranslationTargetInterval, ...]:
    intervals: list[TranslationTargetInterval] = []
    for piece in pieces:
        if any(piece.coefficients[1:]) or piece.coefficients[0] != target:
            continue
        lower = piece.lower
        if intervals and intervals[-1].upper == lower:
            lower = intervals.pop().lower
        first = float(lower)
        if Fraction(first) < lower:
            first = math.nextafter(first, math.inf)
        step = first if lower <= Fraction(first) <= piece.upper else None
        intervals.append(TranslationTargetInterval(lower, piece.upper, step))
    return tuple(intervals)


def _target_root(
    bracket: RootBracket,
    pieces: tuple[ExactPiece, ...],
    target: Fraction,
    step_tol: Fraction,
    volume_tol: Fraction,
) -> TranslationTargetRoot:
    if bracket.count == 1:
        step, residual, resolved = _candidate(
            pieces, bracket.lower, bracket.upper, target, step_tol, volume_tol
        )
    else:
        step, residual, resolved = None, None, False
    return TranslationTargetRoot(
        bracket.lower,
        bracket.upper,
        bracket.count,
        step,
        residual,
        bracket.resolved and resolved,
    )


def polyhedral_translation_targets(
    prepared: PreparedPolyhedralTranslation,
    target: float | int | Fraction,
    *,
    step_tolerance: float,
    volume_tolerance: float,
    max_refinements: int = 128,
) -> TranslationTargetResult:
    """Find every target-volume location on a prepared finite translation path.

    This host-only query uses the original rational cubics and counts repeated
    roots once. It retains constant solution intervals and does not assume a
    monotone path. A float target denotes its exact binary64 value; pass a
    ``Fraction`` for an exact rational target.

    ``step_tolerance`` is a positive absolute dimensionless position bound;
    ``volume_tolerance`` is a nonnegative absolute error in volume units. Both
    bounds must hold for a resolved float action. Scale the volume tolerance
    and target by s**3 when scaling geometry and velocity by s. Preparation's
    exact-coordinate invariance and rounded-input limitations still apply.

    ``max_refinements`` caps bisection depth per root interval, not wall time.
    Existence remains exact when refinement or float separation fails. No
    derivative of this host selection operation is provided. In particular,
    target roots at extrema or events need their own sensitivity contract.

    Example::

        result = polyhedral_translation_targets(
            prepared, 0.2, step_tolerance=1e-12, volume_tolerance=1e-12
        )
        if result.status == "reachable":
            steps = [root.step for root in result.roots]
            # Validate the chosen step with the actual CAD before editing.
    """
    pieces, step_tol, volume_tol = _query_inputs(
        prepared, step_tolerance, volume_tolerance, max_refinements
    )
    exact_target = _rational(target, "target")
    if exact_target < 0:
        raise ValueError("Target volume must be nonnegative")
    intervals = _solution_intervals(pieces, exact_target)

    def accept(lower: Fraction, upper: Fraction) -> bool:
        return (
            upper - lower <= step_tol
            and _candidate(pieces, lower, upper, exact_target, step_tol, volume_tol)[2]
        )

    roots: dict[tuple[Fraction, Fraction], TranslationTargetRoot] = {}
    for piece in pieces:
        coefficients = (piece.coefficients[0] - exact_target, *piece.coefficients[1:])
        for bracket in isolate_roots(
            coefficients, piece.lower, piece.upper, accept, max_refinements
        ):
            if bracket.lower == bracket.upper and any(
                interval.lower <= bracket.lower <= interval.upper
                for interval in intervals
            ):
                continue
            roots[bracket.lower, bracket.upper] = _target_root(
                bracket, pieces, exact_target, step_tol, volume_tol
            )
    ordered = sorted(roots.values(), key=lambda root: (root.lower, root.upper))
    # Distinct exact solutions must not masquerade as distinct usable float actions.
    actions = [root.step for root in ordered if root.step is not None]
    ordered = [
        replace(root, resolved=False)
        if root.step is not None
        and (
            actions.count(root.step) > 1
            or any(i.lower <= Fraction(root.step) <= i.upper for i in intervals)
        )
        else root
        for root in ordered
    ]
    reachable = bool(ordered or intervals)
    status: Literal["reachable", "unreachable", "unresolved"] = (
        "unreachable"
        if not reachable
        else "reachable"
        if all(root.resolved for root in ordered)
        and all(interval.step is not None for interval in intervals)
        else "unresolved"
    )
    return TranslationTargetResult(
        exact_target, tuple(ordered), intervals, reachable, status
    )


def polyhedral_translation_range(
    prepared: PreparedPolyhedralTranslation,
    *,
    step_tolerance: float,
    volume_tolerance: float,
    max_refinements: int = 128,
) -> TranslationVolumeRange:
    """Enclose the full attainable volume range without CAD or grid sampling.

    Examine every rational knot and every root of each derivative quadratic.
    ``step_tolerance`` bounds stationary-root bracket widths, while
    ``volume_tolerance`` bounds their volume-enclosure widths. Units and limits
    match :func:`polyhedral_translation_targets`. A failed refinement retains
    conservative range bounds; it does not establish target infeasibility.

    Compute once for repeated target queries on the same prepared geometry and
    direction. Returned extrema refer only to this complete represented path.
    """
    pieces, step_tol, volume_tol = _query_inputs(
        prepared, step_tolerance, volume_tolerance, max_refinements
    )
    knots = sorted({piece.lower for piece in pieces} | {pieces[-1].upper})
    candidates = [
        TranslationRangeCandidate(t, t, (_volume_at(pieces, t),) * 2, "knot", True)
        for t in knots
    ]
    for piece in pieces:
        slope = derivative(piece.coefficients)
        if not trim(slope):
            candidates.append(
                TranslationRangeCandidate(
                    piece.lower,
                    piece.upper,
                    (piece.coefficients[0],) * 2,
                    "constant",
                    True,
                )
            )
            continue

        def accept(lower: Fraction, upper: Fraction, piece: ExactPiece = piece) -> bool:
            lo, hi = value_bounds(piece.coefficients, lower, upper)
            return upper - lower <= step_tol and hi - lo <= volume_tol

        for bracket in isolate_roots(
            slope, piece.lower, piece.upper, accept, max_refinements
        ):
            if bracket.lower == bracket.upper and bracket.lower in knots:
                continue
            candidates.append(
                TranslationRangeCandidate(
                    bracket.lower,
                    bracket.upper,
                    value_bounds(piece.coefficients, bracket.lower, bracket.upper),
                    "stationary",
                    bracket.resolved,
                )
            )
    bounds = [candidate.volume_bounds for candidate in candidates]
    return TranslationVolumeRange(
        (min(lo for lo, _ in bounds), min(hi for _, hi in bounds)),
        (max(lo for lo, _ in bounds), max(hi for _, hi in bounds)),
        tuple(
            sorted(candidates, key=lambda candidate: (candidate.lower, candidate.upper))
        ),
        all(candidate.resolved for candidate in candidates),
    )
