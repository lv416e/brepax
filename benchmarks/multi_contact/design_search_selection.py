"""Matched bounded design searches and task-wide, source-preserving CAD work."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import pairwise
from math import inf, isfinite, nextafter, sqrt, ulp
from time import perf_counter
from typing import Literal, Protocol

from benchmarks.face_contact.cad import boundary_tolerance, cad_volume
from brepax.experimental._polyhedral_continuation import (
    ContinuationAttempt,
    ContinuationGap,
)
from brepax.experimental.polyhedral_boundary import (
    PreparedPolyhedralEventBoundary,
    prepare_polyhedral_event_boundary,
)
from brepax.experimental.polyhedral_continuation import (
    PreparedPolyhedralContinuation,
    prepare_polyhedral_continuation,
)
from brepax.experimental.polyhedral_design import (
    DesignTargetResult,
    polyhedral_design_target,
)
from brepax.experimental.polyhedral_targets import (
    polyhedral_translation_range,
    polyhedral_translation_targets,
)
from brepax.experimental.polyhedral_translation import (
    PreparedPolyhedralTranslation,
    prepare_polyhedral_translation,
)

from .design_search_fixtures import OptimalityReference
from .event_boundary_cad import CadCallCounts, observe_scene
from .event_boundary_fixtures import AffineBodies, VertexBody
from .parametric_cad import CadMeasurement, CadScene, valid_observation
from .parametric_fixtures import PrismBodies

Method = Literal[
    "certified_boundary", "automatic_continuation", "rebuild_profile", "adaptive_cad"
]
BandStatus = Literal["inside", "outside", "boundary_indeterminate", "invalid"]


class BoundedDesignTask(Protocol):
    """Geometry/target inputs plus evaluator-only independent references."""

    @property
    def name(self) -> str: ...

    @property
    def target(self) -> Fraction: ...

    @property
    def theta0(self) -> Fraction: ...

    @property
    def bounds(self) -> tuple[Fraction, Fraction]: ...

    @property
    def extent(self) -> float: ...

    @property
    def velocity(self) -> tuple[float, float, float]: ...

    def bodies(self, theta: Fraction | float | int) -> PrismBodies: ...

    def vertices(
        self, theta: Fraction | float | int
    ) -> tuple[VertexBody, VertexBody]: ...

    def affine_cells(self) -> AffineBodies: ...

    def scene(
        self, theta: Fraction | float | int, counts: CadCallCounts
    ) -> tuple[CadScene, float]: ...

    def reference_volume(
        self, theta: Fraction | float | int, step: Fraction | float | int
    ) -> Fraction: ...

    def reference_optimality(
        self,
        volume_tolerance: Fraction,
        theta_tolerance: Fraction = Fraction(1, 10**10),
    ) -> OptimalityReference: ...


@dataclass(frozen=True)
class ContinuationEvidence:
    """Public preparation diagnostics without private exact model storage."""

    chart_count: int
    boundary_parameters: tuple[Fraction, ...]
    uncovered: tuple[ContinuationGap, ...]
    attempts: tuple[ContinuationAttempt, ...]


@dataclass(frozen=True)
class SearchBudget:
    """Whole-task limits; a CAD Boolean includes source and final operations."""

    boolean_operations: int = 128
    wall_seconds: float = 60.0
    model_work_limit: int = 256
    parameter_tolerance: Fraction = Fraction(1, 1000)
    volume_relative_tolerance: Fraction = Fraction(1, 1000)
    continuation_max_charts: int = 8
    continuation_max_center_trials: int = 8

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("boolean_operations", self.boolean_operations, 0),
            ("model_work_limit", self.model_work_limit, 1),
            ("continuation_max_charts", self.continuation_max_charts, 1),
            ("continuation_max_center_trials", self.continuation_max_center_trials, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer at least {minimum}")
        if not isfinite(self.wall_seconds) or self.wall_seconds <= 0:
            raise ValueError("wall_seconds must be finite and positive")
        if (
            not isfinite(self.parameter_tolerance)
            or not isfinite(self.volume_relative_tolerance)
            or self.parameter_tolerance <= 0
            or self.volume_relative_tolerance < 0
        ):
            raise ValueError(
                "Expected positive design and nonnegative volume tolerances"
            )


DEFAULT_BUDGET = SearchBudget()


@dataclass
class Observation:
    parameter: float
    measurement: CadMeasurement
    band_status: BandStatus
    source_preserved: bool
    final: bool = False
    stock_volume: float | None = None
    actual_removed_volume: float | None = None
    removed_band_status: BandStatus | None = None
    elapsed_seconds: float = 0.0
    cumulative_calls: CadCallCounts = field(default_factory=CadCallCounts)


@dataclass
class DesignSearchResult:
    """Whole-task result; stage times overlap and must not be summed.

    In particular geometry_seconds includes fresh final construction, which is
    also part of final_seconds. Only total_seconds is the end-to-end duration.
    """

    task: str
    method: Method
    target: Fraction
    volume_tolerance: Fraction
    parameter_tolerance: Fraction = Fraction(1, 1000)
    selected_parameter: float | None = None
    selected_step: float | None = None
    objective: Fraction | None = None
    operational_success: bool = False
    decision: str = "unresolved"
    model_status: str | None = None
    model_query: DesignTargetResult | None = None
    objective_lower: Fraction | None = None
    objective_upper: Fraction | None = None
    model_queries: int = 0
    boxes_examined: int = 0
    boundary_preparations: int = 0
    continuation_preparations: int = 0
    continuation: ContinuationEvidence | None = None
    profile_rebuilds: int = 0
    model_seconds: float = 0.0
    model_preparation_seconds: float = 0.0
    model_search_seconds: float = 0.0
    model_query_elapsed_seconds: float | None = None
    geometry_seconds: float = 0.0
    final_seconds: float = 0.0
    total_seconds: float = 0.0
    wall_cap_exceeded: bool = False
    resource_limit: str | None = None
    errors: list[str] = field(default_factory=list)
    calls: CadCallCounts = field(default_factory=CadCallCounts)
    observations: list[Observation] = field(default_factory=list)
    final: Observation | None = None


class TaskLimitError(Exception):
    """A shared whole-task limit stopped work; it is not infeasibility."""


def classify_band(
    value: float | None, target: Fraction, tolerance: Fraction
) -> BandStatus:
    """Use exact binary64 membership, with a separate ulp-scale ambiguity guard.

    The guard is numerical reporting, not a larger acceptance band. It does not
    estimate the independent error of an OCCT Boolean.
    """
    if value is None or not isfinite(value):
        return "invalid"
    exact = Fraction(value)
    lower, upper = target - tolerance, target + tolerance
    guard = Fraction(8 * max(ulp(value), ulp(float(target)), ulp(float(tolerance))))
    if min(abs(exact - lower), abs(exact - upper)) <= guard:
        return "boundary_indeterminate"
    return "inside" if lower <= exact <= upper else "outside"


def _signature(scene: CadScene) -> tuple[float | None, ...]:
    return (
        cad_volume(scene.fixed),
        boundary_tolerance(scene.fixed),
        cad_volume(scene.moving),
        boundary_tolerance(scene.moving),
    )


class CadLedger:
    """One task's scene cache, attempts, and reserved fresh final validation."""

    def __init__(
        self, task: BoundedDesignTask, budget: SearchBudget, result: DesignSearchResult
    ) -> None:
        self.task, self.budget, self.result = task, budget, result
        self.started = perf_counter()
        self.scenes: dict[float, tuple[CadScene, float]] = {}
        self.cache: dict[tuple[float, float], Observation] = {}
        bodies = task.bodies(task.theta0)
        self.fusion_cost = len(bodies.fixed) + len(bodies.moving) - 2
        self.final_reserve = self.fusion_cost + 2
        self.best: Observation | None = None

    @property
    def used(self) -> int:
        calls = self.result.calls
        return calls.common + calls.cut + calls.fuse

    def check(self, *, operations: int = 0, final: bool = False) -> None:
        # Stop online work early enough to attempt the already-budgeted fresh
        # final Cut. This reservation is identical across methods, not free time.
        final_time = min(2.0, self.budget.wall_seconds / 10)
        wall_limit = self.budget.wall_seconds - (0 if final else final_time)
        if perf_counter() - self.started >= wall_limit:
            raise TaskLimitError("Whole-task cooperative wall limit reached")
        reserve = 0 if final else self.final_reserve
        if self.used + operations + reserve > self.budget.boolean_operations:
            raise TaskLimitError("Whole-task Boolean budget reached")

    def scene(self, theta: float, *, fresh: bool = False) -> tuple[CadScene, float]:
        if not fresh and theta in self.scenes:
            return self.scenes[theta]
        self.check(operations=self.fusion_cost, final=fresh)
        counts = CadCallCounts()
        started = perf_counter()
        try:
            scene, stock = self.task.scene(Fraction(theta), counts=counts)
        finally:
            self.result.geometry_seconds += perf_counter() - started
            self.result.calls.common += counts.common
            self.result.calls.cut += counts.cut
            self.result.calls.fuse += counts.fuse
        if not fresh:
            self.scenes[theta] = scene, stock
        return scene, stock

    def observe(self, theta: float, step: float, *, final: bool = False) -> Observation:
        key = theta, step
        if not final and key in self.cache:
            return self.cache[key]
        geometry_cost = self.fusion_cost if final or theta not in self.scenes else 0
        self.check(operations=geometry_cost + (2 if final else 1), final=final)
        scene, stock = self.scene(theta, fresh=final)
        before = _signature(scene)
        record = observe_scene(scene, step, cut=final)
        self.result.calls.common += record.common_calls
        self.result.calls.cut += record.cut_calls
        self.result.calls.fuse += record.fusion_calls
        preserved = before == _signature(scene)
        status = classify_band(
            record.working_volume, self.task.target, self.result.volume_tolerance
        )
        if not valid_observation(record, cut=final) or not preserved:
            status = "invalid"
        removed = None if record.cut_volume is None else stock - record.cut_volume
        observation = Observation(
            theta,
            record,
            status,
            preserved,
            final,
            stock,
            removed,
            classify_band(removed, self.task.target, self.result.volume_tolerance)
            if final
            else None,
            perf_counter() - self.started,
            CadCallCounts(
                self.result.calls.common, self.result.calls.cut, self.result.calls.fuse
            ),
        )
        self.result.observations.append(observation)
        if not final:
            self.cache[key] = observation
            if status == "inside" and (
                self.best is None or self.rank(observation) < self.rank(self.best)
            ):
                self.best = observation
        return observation

    def rank(self, observation: Observation) -> tuple[Fraction, float]:
        return abs(
            Fraction(observation.parameter) - self.task.theta0
        ), observation.measurement.step


def _profile_value(profile: PreparedPolyhedralTranslation, step: float) -> Fraction:
    # Benchmark-side exact action audit; this is not an independent reference.
    rational = Fraction(step)
    return next(
        piece.value(rational)
        for piece in reversed(profile._exact_pieces)
        if piece.lower <= rational <= piece.upper
    )


def _profile_action(
    profile: PreparedPolyhedralTranslation,
    target: Fraction,
    tolerance: Fraction,
    extent: float,
) -> float | None:
    def qualified(steps: list[float]) -> float | None:
        accepted = [
            step
            for step in steps
            if isfinite(step)
            and 0 <= step <= extent
            and abs(_profile_value(profile, step) - target) <= tolerance
        ]
        return min(accepted, default=None)

    def target_steps(level: Fraction, margin: Fraction) -> list[float]:
        query = polyhedral_translation_targets(
            profile,
            level,
            step_tolerance=extent / 1000,
            volume_tolerance=margin,
        )
        return [root.step for root in query.roots if root.step is not None] + [
            interval.step for interval in query.intervals if interval.step is not None
        ]

    selected = qualified(target_steps(target, tolerance))
    if selected is not None:
        return selected
    envelope = polyhedral_translation_range(
        profile, step_tolerance=extent / 1000, volume_tolerance=tolerance
    )
    lower_band = target - tolerance
    if envelope.maximum[1] < lower_band:
        return None
    steps = [
        step
        for item in envelope.candidates
        for position in (item.lower, item.upper, (item.lower + item.upper) / 2)
        for step in (
            float(position),
            nextafter(float(position), -inf),
            nextafter(float(position), inf),
        )
    ]
    selected = qualified(steps)
    if selected is not None:
        return selected
    if envelope.maximum[0] > lower_band:
        level = min(target, (lower_band + envelope.maximum[0]) / 2)
        margin = min(level - lower_band, target + tolerance - level)
        return qualified(target_steps(level, margin))
    return None


def _seed_parameters(task: BoundedDesignTask, divisions: int) -> list[Fraction]:
    lower, upper = task.bounds
    seeds = {lower + (upper - lower) * i / divisions for i in range(divisions + 1)}
    seeds.add(task.theta0)
    if lower <= 0 <= upper:
        seeds.add(Fraction(0))
    return sorted(seeds, key=lambda theta: (abs(theta - task.theta0), theta))


def _next_parameter(
    task: BoundedDesignTask,
    seen: dict[Fraction, bool],
    iteration: int,
    tolerance: Fraction,
) -> Fraction | None:
    points = sorted(seen)
    intervals = [(a, b) for a, b in pairwise(points) if b - a > tolerance]
    if not intervals:
        return None
    # Alternate global coverage and interface refinement. Unknown interiors are
    # never discarded because two sampled endpoints share a feasibility label.
    changing = [(a, b) for a, b in intervals if seen[a] != seen[b]]
    if changing and iteration % 4:
        a, b = min(
            changing,
            key=lambda pair: (
                min(abs(pair[0] - task.theta0), abs(pair[1] - task.theta0)),
                -(pair[1] - pair[0]),
            ),
        )
    else:
        a, b = max(
            intervals,
            key=lambda pair: (
                pair[1] - pair[0],
                -min(abs(pair[0] - task.theta0), abs(pair[1] - task.theta0)),
            ),
        )
    return (a + b) / 2


def _rebuild_search(ledger: CadLedger) -> None:
    task, result = ledger.task, ledger.result
    seen: dict[Fraction, bool] = {}
    pending = _seed_parameters(task, 8)
    iteration = 0
    while result.model_queries < ledger.budget.model_work_limit:
        ledger.check()
        theta = (
            pending.pop(0)
            if pending
            else _next_parameter(
                task, seen, iteration, ledger.budget.parameter_tolerance
            )
        )
        if theta is None:
            break
        started = perf_counter()
        try:
            preparation_started = perf_counter()
            try:
                fixed, moving = task.vertices(theta)
                result.profile_rebuilds += 1
                profile = prepare_polyhedral_translation(
                    fixed, moving, task.velocity, max_step=task.extent
                )
            finally:
                result.model_preparation_seconds += perf_counter() - preparation_started
            search_started = perf_counter()
            try:
                step = _profile_action(
                    profile, task.target, result.volume_tolerance, task.extent
                )
            finally:
                result.model_search_seconds += perf_counter() - search_started
            result.model_queries += 1
        finally:
            result.model_seconds += perf_counter() - started
        seen[theta] = step is not None
        if step is not None and (
            ledger.best is None
            or abs(theta - task.theta0) < ledger.rank(ledger.best)[0]
        ):
            ledger.observe(float(theta), step)
        if ledger.best is not None and ledger.rank(ledger.best)[0] == 0:
            break
        iteration += 1
    result.model_status = "sampled_theta_only"
    result.decision = "feasible_sample" if ledger.best else "inconclusive_samples"


def _cad_at_parameter(
    ledger: CadLedger, theta: Fraction, *, broad_only: bool = False
) -> bool:
    """Broad translation coverage, all observed brackets, and sampled local peaks."""
    task = ledger.task
    value = float(theta)
    records: dict[float, Observation] = {}

    def sample(step: float) -> Observation:
        observation = ledger.observe(value, step)
        records[step] = observation
        return observation

    if ledger.best is not None:
        observation = sample(ledger.best.measurement.step)
        if observation.band_status == "inside":
            return True
    for index in range(9):
        observation = sample(task.extent * index / 8)
        if observation.band_status == "inside":
            return True
    good = sorted(
        (t, volume)
        for t, obs in records.items()
        if valid_observation(obs.measurement)
        and (volume := obs.measurement.working_volume) is not None
    )
    q = float(task.target)
    for (left, a), (right, b) in pairwise(good):
        assert a is not None and b is not None
        if (a - q) * (b - q) >= 0:
            continue
        for _ in range(24):
            middle = (left + right) / 2
            if middle in (left, right):
                break
            observed = sample(middle)
            if observed.band_status == "inside":
                return True
            v = observed.measurement.working_volume
            if v is None:
                break
            if (a - q) * (v - q) <= 0:
                right, b = middle, v
            else:
                left, a = middle, v
    if broad_only:
        return False
    peaks = [
        (good[i][1], good[i - 1][0], good[i + 1][0])
        for i in range(1, len(good) - 1)
        if good[i][1] >= max(good[i - 1][1], good[i + 1][1]) and good[i][1] > 0
    ]
    ratio = (sqrt(5) - 1) / 2
    for _, lower, upper in sorted(peaks, reverse=True):
        c, d = upper - ratio * (upper - lower), lower + ratio * (upper - lower)
        lc, rd = sample(c), sample(d)
        for _ in range(4):
            if lc.band_status == "inside" or rd.band_status == "inside":
                return True
            lv, rv = lc.measurement.working_volume, rd.measurement.working_volume
            if lv is None or rv is None:
                break
            if lv > rv:
                upper, d, rd = d, c, lc
                c = upper - ratio * (upper - lower)
                lc = sample(c)
            else:
                lower, c, lc = c, d, rd
                d = lower + ratio * (upper - lower)
                rd = sample(d)
    # A newly discovered maximum can create two target brackets even though
    # every initial grid observation lay below the target.
    good = sorted(
        (t, volume)
        for t, obs in records.items()
        if valid_observation(obs.measurement)
        and (volume := obs.measurement.working_volume) is not None
    )
    for (left, a), (right, b) in pairwise(good):
        assert a is not None and b is not None
        if (a - q) * (b - q) >= 0:
            continue
        for _ in range(16):
            middle = (left + right) / 2
            observed = sample(middle)
            if observed.band_status == "inside":
                return True
            v = observed.measurement.working_volume
            if v is None or middle in (left, right):
                break
            if (a - q) * (v - q) <= 0:
                right, b = middle, v
            else:
                left, a = middle, v
    return any(obs.band_status == "inside" for obs in records.values())


def _cad_search(ledger: CadLedger) -> None:
    seen: dict[Fraction, bool] = {}
    pending = _seed_parameters(ledger.task, 4)
    iteration = 0
    # Pay for broad dimension coverage before allowing one infeasible local
    # translation search to consume the whole task's Boolean budget.
    for theta in pending:
        seen[theta] = _cad_at_parameter(ledger, theta, broad_only=True)
        if ledger.best is not None and ledger.rank(ledger.best)[0] == 0:
            ledger.result.decision = "feasible_sample"
            return
    while iteration < ledger.budget.model_work_limit:
        candidate = (
            pending.pop(0)
            if pending
            else _next_parameter(
                ledger.task, seen, iteration, ledger.budget.parameter_tolerance
            )
        )
        if candidate is None:
            break
        seen[candidate] = _cad_at_parameter(ledger, candidate)
        if ledger.best is not None and ledger.rank(ledger.best)[0] == 0:
            break
        iteration += 1
    ledger.result.decision = (
        "feasible_sample" if ledger.best else "inconclusive_samples"
    )


def _certified_search(ledger: CadLedger) -> None:
    task, result = ledger.task, ledger.result
    ledger.check()
    started = perf_counter()
    try:
        preparation_started = perf_counter()
        prepared: PreparedPolyhedralEventBoundary | PreparedPolyhedralContinuation
        try:
            fixed, moving = task.affine_cells()
            if result.method == "automatic_continuation":
                result.continuation_preparations += 1
                prepared = prepare_polyhedral_continuation(
                    fixed,
                    moving,
                    task.velocity,
                    max_step=task.extent,
                    parameter_bounds=task.bounds,
                    seed_parameter=task.theta0,
                    max_charts=ledger.budget.continuation_max_charts,
                    max_center_trials=ledger.budget.continuation_max_center_trials,
                )
                result.continuation = ContinuationEvidence(
                    prepared.chart_count,
                    prepared.boundary_parameters,
                    prepared.uncovered,
                    prepared.attempts,
                )
            else:
                result.boundary_preparations += 1
                prepared = prepare_polyhedral_event_boundary(
                    fixed,
                    moving,
                    task.velocity,
                    max_step=task.extent,
                    parameter_radius=Fraction(1, 8),
                )
        finally:
            result.model_preparation_seconds += perf_counter() - preparation_started
        ledger.check()
        search_started = perf_counter()
        try:
            query = polyhedral_design_target(
                prepared,
                task.target,
                parameter_bounds=task.bounds,
                initial_parameter=task.theta0,
                parameter_tolerance=ledger.budget.parameter_tolerance,
                step_tolerance=task.extent / 1000,
                volume_tolerance=result.volume_tolerance,
                max_boxes=ledger.budget.model_work_limit,
            )
        finally:
            result.model_search_seconds += perf_counter() - search_started
        result.model_status = query.status
        result.model_query = query
        result.model_query_elapsed_seconds = perf_counter() - ledger.started
        result.objective_lower, result.objective_upper = (
            query.objective_lower,
            query.objective_upper,
        )
        result.model_queries, result.boxes_examined = (
            query.point_queries,
            query.boxes_examined,
        )
        result.decision = query.status
    finally:
        result.model_seconds += perf_counter() - started
    if query.witness is not None:
        ledger.observe(query.witness.parameter, query.witness.step)


def evaluate_design_search(
    task: BoundedDesignTask, method: Method, budget: SearchBudget = DEFAULT_BUDGET
) -> DesignSearchResult:
    """Run one cold task; independent references are deliberately not consulted."""
    result = DesignSearchResult(
        task.name,
        method,
        task.target,
        task.target * budget.volume_relative_tolerance,
        budget.parameter_tolerance,
    )
    ledger = CadLedger(task, budget, result)
    try:
        if method in ("certified_boundary", "automatic_continuation"):
            _certified_search(ledger)
        elif method == "rebuild_profile":
            _rebuild_search(ledger)
        elif method == "adaptive_cad":
            _cad_search(ledger)
        else:
            raise ValueError(f"Unknown method: {method}")
    except TaskLimitError as error:
        result.resource_limit = str(error)
        result.decision = "resource_limited"
    except Exception as error:
        result.errors.append(f"{type(error).__name__}: {error}")
        result.decision = "failed"
    if ledger.best is not None:
        best = ledger.best
        result.selected_parameter, result.selected_step = (
            best.parameter,
            best.measurement.step,
        )
        result.objective = ledger.rank(best)[0]
        started = perf_counter()
        try:
            result.final = ledger.observe(
                best.parameter, best.measurement.step, final=True
            )
            final = result.final
            balance = final.measurement.working_balance_error
            result.operational_success = bool(
                final.band_status == final.removed_band_status == "inside"
                and balance is not None
                and final.stock_volume is not None
                and abs(balance) <= abs(final.stock_volume) * 1e-9
            )
        except TaskLimitError as error:
            result.resource_limit = str(error)
        except Exception as error:
            result.errors.append(f"Final {type(error).__name__}: {error}")
        result.final_seconds = perf_counter() - started
    result.total_seconds = perf_counter() - ledger.started
    result.wall_cap_exceeded = result.total_seconds > budget.wall_seconds
    if result.wall_cap_exceeded:
        result.operational_success = False
    return result
