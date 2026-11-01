"""Measured CAD state and lazy model preparation for matched-accuracy edits."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import pairwise
from math import sqrt
from time import perf_counter
from typing import Literal

from brepax.experimental.polyhedral_boundary import (
    PreparedPolyhedralEventBoundary,
    prepare_polyhedral_event_boundary,
    specialize_polyhedral_event_boundary,
)
from brepax.experimental.polyhedral_targets import (
    TranslationTargetResult,
    TranslationVolumeRange,
    polyhedral_translation_range,
    polyhedral_translation_targets,
)
from brepax.experimental.polyhedral_translation import (
    PreparedPolyhedralTranslation,
    prepare_polyhedral_translation,
)

from .event_boundary_cad import (
    WALL_CAP,
    CadCallCounts,
    ResourceCapError,
    check_deadline,
    observe_scene,
    prepare_scene,
)
from .event_boundary_fixtures import Family, affine_cells, vertices
from .parametric_cad import CadMeasurement, CadScene, valid_observation

PROBE_CAP = 128
Method = Literal["rebuild_profile", "cad_search", "reusable_boundary"]


@dataclass
class BoundaryCache:
    """One family's paid reusable preparation, including an unsupported result."""

    attempted: bool = False
    prepared: PreparedPolyhedralEventBoundary | None = None
    rejection: str | None = None


@dataclass
class ModelPreparation:
    """Attempted host work and explicit fallback; no CAD/reference acceptance."""

    profile_attempted: bool = False
    profile_rebuilt: bool = False
    profile_seconds: float = 0.0
    boundary_preparation_attempted: bool = False
    boundary_preparation_seconds: float = 0.0
    boundary_specialization_seconds: float = 0.0
    boundary_specialized: bool = False
    boundary_fallback: bool = False
    boundary_fallback_reason: str | None = None
    boundary_parameter_interval: tuple[Fraction | None, Fraction | None] | None = None


@dataclass
class SelectionResult:
    """Selected step, model evidence and retained failure, before final CAD."""

    decision: str = "failed"
    selected_step: float | None = None
    seed_hit: bool = False
    model: ModelPreparation = field(default_factory=ModelPreparation)
    exact_target_query: TranslationTargetResult | None = None
    profile_range: TranslationVolumeRange | None = None
    profile_query_seconds: float = 0.0
    error: str | None = None
    resource_limit: str | None = None


@dataclass
class AttemptTimings:
    """Actual stage times; selection includes its nested model/query times."""

    geometry_seconds: float = 0.0
    selection_seconds: float = 0.0
    final_seconds: float = 0.0
    seconds: float = 0.0
    wall_cap_exceeded: bool = False


@dataclass(frozen=True)
class OperationalOutcome:
    """Eligibility derived only from measured source/final CAD and the target."""

    valid_edit: bool
    relative_common_error: float | None
    removed_stock_minus_cut_volume: float | None
    relative_removed_error: float | None
    operational_success: bool


@dataclass
class EditAttempt:
    """One dimension's owned observations, independent of later oracle audit."""

    family: str
    theta: Fraction
    method: Method
    target: Fraction
    seed: float | None
    preparation_calls: CadCallCounts = field(default_factory=CadCallCounts)
    final_preparation_calls: CadCallCounts = field(default_factory=CadCallCounts)
    source_stock_volume: float | None = None
    final_stock_volume: float | None = None
    initial: CadMeasurement | None = None
    probes: tuple[CadMeasurement, ...] = ()
    final: CadMeasurement | None = None
    selection: SelectionResult = field(default_factory=SelectionResult)
    timings: AttemptTimings = field(default_factory=AttemptTimings)
    error: str | None = None
    outcome: OperationalOutcome | None = None

    @property
    def calls(self) -> CadCallCounts:
        records = [r for r in (self.initial, *self.probes, self.final) if r is not None]
        return CadCallCounts(
            self.preparation_calls.common
            + self.final_preparation_calls.common
            + sum(r.common_calls for r in records),
            self.preparation_calls.cut
            + self.final_preparation_calls.cut
            + sum(r.cut_calls for r in records),
            self.preparation_calls.fuse
            + self.final_preparation_calls.fuse
            + sum(r.fusion_calls for r in records),
        )


def _prepare_profile(
    family: Family,
    theta: Fraction,
    method: Method,
    observations: Observations,
    cache: BoundaryCache,
    diagnostics: ModelPreparation,
) -> PreparedPolyhedralTranslation:
    check_deadline(observations.started, "model preparation")
    if method == "reusable_boundary":
        if not cache.attempted:
            cache.attempted = True
            diagnostics.boundary_preparation_attempted = True
            started = perf_counter()
            try:
                fixed, moving = affine_cells(family)
                cache.prepared = prepare_polyhedral_event_boundary(
                    fixed,
                    moving,
                    family.velocity,
                    max_step=family.extent,
                    parameter_radius=Fraction(1, 8),
                )
            except ValueError as error:
                cache.rejection = f"{type(error).__name__}: {error}"
            finally:
                diagnostics.boundary_preparation_seconds = perf_counter() - started
        if cache.prepared is not None:
            diagnostics.boundary_parameter_interval = cache.prepared.parameter_interval
            check_deadline(observations.started, "boundary specialization")
            started = perf_counter()
            try:
                model = specialize_polyhedral_event_boundary(cache.prepared, theta)
                diagnostics.boundary_specialized = True
                return model
            except ValueError as error:
                diagnostics.boundary_fallback_reason = (
                    f"{type(error).__name__}: {error}"
                )
            finally:
                diagnostics.boundary_specialization_seconds = perf_counter() - started
        else:
            diagnostics.boundary_fallback_reason = cache.rejection
        diagnostics.boundary_fallback = True
    check_deadline(observations.started, "fixed-profile preparation")
    diagnostics.profile_attempted = True
    started = perf_counter()
    try:
        fixed_vertices, moving_vertices = vertices(family, theta)
        model = prepare_polyhedral_translation(
            fixed_vertices, moving_vertices, family.velocity, max_step=family.extent
        )
        diagnostics.profile_rebuilt = True
        return model
    finally:
        diagnostics.profile_seconds += perf_counter() - started


def _value(record: CadMeasurement) -> float | None:
    return record.working_volume if valid_observation(record) else None


class Observations:
    def __init__(
        self, scene: CadScene, target: Fraction, initial: CadMeasurement, started: float
    ) -> None:
        self.scene = scene
        self.target = float(target)
        self.tolerance = float(target / 1000)
        self.started = started
        self.cache = {initial.step: initial}
        self.probes: list[CadMeasurement] = []

    def qualified(self, record: CadMeasurement) -> bool:
        value = _value(record)
        return value is not None and abs(value - self.target) <= self.tolerance

    def sample(self, step: float) -> CadMeasurement:
        step = float(step)
        if step in self.cache:
            return self.cache[step]
        if len(self.probes) >= PROBE_CAP or perf_counter() - self.started >= WALL_CAP:
            raise ResourceCapError("Online observation resource cap reached")
        record = observe_scene(self.scene, step, deadline_started=self.started)
        self.probes.append(record)
        self.cache[step] = record
        return record

    def best(self) -> CadMeasurement | None:
        candidates = [
            (value, record)
            for record in self.cache.values()
            if (value := _value(record)) is not None
        ]
        selected = min(
            candidates,
            key=lambda item: (abs(item[0] - self.target), item[1].step),
            default=None,
        )
        return None if selected is None else selected[1]


def _observed_bracket(
    observations: Observations,
) -> tuple[CadMeasurement | None, str] | None:
    ordered = sorted(
        (
            record
            for record in observations.cache.values()
            if _value(record) is not None
        ),
        key=lambda record: record.step,
    )
    for previous, current in pairwise(ordered):
        left, right = _value(previous), _value(current)
        if left is None or right is None:
            continue
        if (left - observations.target) * (right - observations.target) >= 0:
            continue
        while True:
            step = (previous.step + current.step) / 2
            if step in (previous.step, current.step):
                return observations.best(), "bracket_float_limit"
            middle = observations.sample(step)
            if observations.qualified(middle):
                return middle, "bracket_band_hit"
            value = _value(middle)
            if value is None:
                return observations.best(), "bracket_observation_failed"
            if (left - observations.target) * (value - observations.target) < 0:
                current = middle
            else:
                previous, left = middle, value
    return None


def _search_completion(
    observations: Observations, record: CadMeasurement, label: str
) -> tuple[CadMeasurement | None, str] | None:
    if observations.qualified(record):
        return record, label
    return _observed_bracket(observations)


def _cad_search(
    observations: Observations, extent: float
) -> tuple[CadMeasurement | None, str]:
    """Use the first observed bracket, including seed/local-exploration samples."""
    outcome = _observed_bracket(observations)
    if outcome is not None:
        return outcome
    grid = [observations.cache[0.0]]
    for index in range(1, 33):
        current = observations.sample(extent * index / 32)
        grid.append(current)
        outcome = _search_completion(observations, current, "grid_band_hit")
        if outcome is not None:
            return outcome
    good = [(i, r) for i, r in enumerate(grid) if _value(r) is not None]
    if not good:
        return None, "no_valid_observation"
    peak_index, _ = max(good, key=lambda item: (_value(item[1]), -item[1].step))
    lower = grid[max(0, peak_index - 1)].step
    upper = grid[min(32, peak_index + 1)].step
    ratio = (sqrt(5) - 1) / 2
    c, d = upper - ratio * (upper - lower), lower + ratio * (upper - lower)
    left_record = observations.sample(c)
    outcome = _search_completion(observations, left_record, "local_peak_band_hit")
    if outcome is not None:
        return outcome
    right_record = observations.sample(d)
    while upper > lower:
        outcome = _search_completion(observations, right_record, "local_peak_band_hit")
        if outcome is not None:
            return outcome
        left, right = _value(left_record), _value(right_record)
        if left is None or right is None:
            return observations.best(), "local_peak_observation_failed"
        if left > right:
            upper, d, right_record = d, c, left_record
            c = upper - ratio * (upper - lower)
            if c in (lower, upper, d):
                break
            left_record = observations.sample(c)
            outcome = _search_completion(
                observations, left_record, "local_peak_band_hit"
            )
            if outcome is not None:
                return outcome
        else:
            lower, c, left_record = c, d, right_record
            d = lower + ratio * (upper - lower)
            if d in (lower, upper, c):
                break
            right_record = observations.sample(d)
    return observations.best(), "local_peak_float_limit"


def _profile_search(
    family: Family,
    theta: Fraction,
    method: Method,
    observations: Observations,
    cache: BoundaryCache,
) -> SelectionResult:
    result = SelectionResult()
    try:
        model = _prepare_profile(
            family, theta, method, observations, cache, result.model
        )
        started = perf_counter()
        try:
            check_deadline(observations.started, "target query")
            query = polyhedral_translation_targets(
                model,
                family.target,
                step_tolerance=family.extent / 1000,
                volume_tolerance=float(family.target / 1000),
                max_refinements=128,
            )
            result.exact_target_query = query
            candidates = [
                r.step for r in query.roots if r.resolved and r.step is not None
            ]
            candidates.extend(i.step for i in query.intervals if i.step is not None)
            if candidates:
                result.selected_step = observations.sample(min(candidates)).step
                result.decision = "profile_root"
                return result
            if query.exact_reachable:
                result.decision = "profile_root_unresolved"
                return result
            check_deadline(observations.started, "range query")
            volume_range = polyhedral_translation_range(
                model,
                step_tolerance=family.extent / 1000,
                volume_tolerance=float(family.target / 1000),
                max_refinements=128,
            )
            result.profile_range = volume_range
            if volume_range.maximum[1] < family.target - family.target / 1000:
                result.decision = "profile_band_unreachable"
                return result
            candidates_by_value = sorted(
                volume_range.candidates,
                key=lambda item: (
                    min(abs(value - family.target) for value in item.volume_bounds),
                    item.lower,
                ),
            )
            for candidate in candidates_by_value:
                record = observations.sample(
                    float((candidate.lower + candidate.upper) / 2)
                )
                if observations.qualified(record):
                    result.selected_step = record.step
                    result.decision = "profile_extremum_band_hit"
                    return result
            best = observations.best()
            result.selected_step = None if best is None else best.step
            result.decision = "profile_extrema_unresolved"
        finally:
            result.profile_query_seconds = perf_counter() - started
    except ResourceCapError as error:
        result.decision, result.resource_limit = "resource_cap", str(error)
        best = observations.best()
        result.selected_step = None if best is None else best.step
    except Exception as error:
        result.error = f"{type(error).__name__}: {error}"
    return result


def evaluate(
    family: Family,
    theta: Fraction,
    method: Method,
    *,
    seed: float | None = None,
    boundary_cache: BoundaryCache | None = None,
) -> EditAttempt:
    """Execute selection and final validation; oracle data cannot change state."""
    if method not in ("rebuild_profile", "cad_search", "reusable_boundary"):
        raise ValueError("Unknown method")
    started = perf_counter()
    cache = BoundaryCache() if boundary_cache is None else boundary_cache
    attempt = EditAttempt(family.name, theta, method, family.target, seed)
    observations = None
    try:
        stage_started = perf_counter()
        try:
            scene, attempt.source_stock_volume = prepare_scene(
                family,
                theta,
                attempt.preparation_calls,
            )
        finally:
            attempt.timings.geometry_seconds = perf_counter() - stage_started
        attempt.initial = observe_scene(scene, 0.0, deadline_started=started)
        if not valid_observation(attempt.initial):
            raise RuntimeError("Initial CAD observation failed")
        observations = Observations(scene, family.target, attempt.initial, started)
        stage_started = perf_counter()
        try:
            if seed is not None:
                record = observations.sample(seed)
                if observations.qualified(record):
                    attempt.selection = SelectionResult(
                        decision="seed_band_hit",
                        selected_step=record.step,
                        seed_hit=True,
                    )
            if attempt.selection.selected_step is None:
                if method in ("rebuild_profile", "reusable_boundary"):
                    attempt.selection = _profile_search(
                        family, theta, method, observations, cache
                    )
                else:
                    selected, decision = _cad_search(observations, family.extent)
                    attempt.selection = SelectionResult(
                        decision=decision,
                        selected_step=None if selected is None else selected.step,
                    )
        except ResourceCapError as error:
            best = observations.best()
            attempt.selection = SelectionResult(
                decision="resource_cap",
                resource_limit=str(error),
                selected_step=None if best is None else best.step,
            )
        finally:
            attempt.timings.selection_seconds = perf_counter() - stage_started
        check_deadline(started, "final validation")
        if attempt.selection.selected_step is not None:
            stage_started = perf_counter()
            try:
                fresh, attempt.final_stock_volume = prepare_scene(
                    family,
                    theta,
                    attempt.final_preparation_calls,
                )
                attempt.final = observe_scene(
                    fresh,
                    attempt.selection.selected_step,
                    deadline_started=started,
                    cut=True,
                )
            finally:
                attempt.timings.final_seconds = perf_counter() - stage_started
    except ResourceCapError as error:
        attempt.selection.decision = "resource_cap"
        attempt.selection.resource_limit = str(error)
    except Exception as error:
        attempt.error = f"{type(error).__name__}: {error}"
    finally:
        if observations is not None:
            attempt.probes = tuple(observations.probes)
        attempt.timings.seconds = perf_counter() - started
        attempt.timings.wall_cap_exceeded = attempt.timings.seconds > WALL_CAP
    attempt.outcome = operational_outcome(attempt)
    return attempt


def operational_outcome(attempt: EditAttempt) -> OperationalOutcome:
    """Determine eligibility from measured CAD, without independent reference data."""
    target = float(attempt.target)
    stock = attempt.final_stock_volume or attempt.source_stock_volume or 0.0
    final = attempt.final
    valid = bool(
        final is not None
        and final.error is None
        and final.source_valid
        and final.common_valid
        and final.cut_valid
        and final.working_volume is not None
        and -stock * 1e-9 <= final.working_volume <= stock * (1 + 1e-9)
        and final.working_balance_error is not None
        and abs(final.working_balance_error) <= stock * 1e-9
    )
    common_error = (
        None
        if final is None or final.working_volume is None
        else abs(final.working_volume - target) / target
    )
    removed = (
        None
        if final is None
        or final.cut_volume is None
        or attempt.final_stock_volume is None
        else attempt.final_stock_volume - final.cut_volume
    )
    removed_error = None if removed is None else abs(removed - target) / target
    success = bool(
        valid
        and common_error is not None
        and common_error <= 0.001
        and removed_error is not None
        and removed_error <= 0.001
        and not attempt.timings.wall_cap_exceeded
    )
    return OperationalOutcome(valid, common_error, removed, removed_error, success)
