"""Finite CAD observations for budgeted contact-guided candidate selection.

The full grid is an offline reference. Policies see only the observations
returned by their counted queries. Deterministic reference and final-Cut
observations are physically cached across policies, while each policy retains
its full logical cost. Cached timings are not end-to-end inference timings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from benchmarks.face_contact.cad import Motion, Placement
from brepax.experimental.planar_contact import (
    planar_contact_set_volume_direction,
    prepare_planar_contact_set,
)
from brepax.experimental.planar_contact_collection_cad import (
    prepare_planar_contact_set_reference,
)

from .cad import (
    EditMeasurement,
    VolumeMeasurement,
    apply_cut,
    build_components,
    measure_volume,
)
from .fixtures import ContactFixture, candidate_motions
from .references import boxes_overlap_volume
from .selection_policies import STEPS, Observation

_BATCH = jax.jit(jax.vmap(planar_contact_set_volume_direction, in_axes=(None, 0, 0, 0)))


@dataclass(frozen=True)
class Endpoint:
    """A finite observed action, with an independent translation oracle."""

    direction: int
    step: float
    measurement: VolumeMeasurement
    interval_volume: float | None = None
    interval_absolute_error: float | None = None


@dataclass
class SelectionCase:
    """Prepared direction rates and independently observed finite CAD values."""

    fixture: ContactFixture
    cutter_volume: float
    initial: VolumeMeasurement
    endpoints: tuple[Endpoint, ...]
    rates: tuple[float, ...] | None = None
    preparation_error: str | None = None
    preparation_seconds: float = 0.0
    first_query_seconds: float = 0.0
    warm_query_seconds: float = 0.0
    patch_count: int | None = None

    @property
    def initial_valid(self) -> bool:
        return bool(
            valid_measurement(self.initial, self.cutter_volume)
            and self.initial.working_volume is not None
            and abs(self.initial.working_volume) <= 1e-10 * self.cutter_volume
        )

    def volume_matrix(self) -> NDArray[np.float64]:
        values = [
            e.measurement.working_volume
            if valid_measurement(e.measurement, self.cutter_volume)
            else np.nan
            for e in self.endpoints
        ]
        return np.asarray(values, dtype=np.float64).reshape(24, len(STEPS))


def valid_measurement(measurement: VolumeMeasurement, scale: float) -> bool:
    """A usable observation; validity alone does not certify CAD accuracy."""
    value = measurement.working_volume
    return bool(
        measurement.error is None
        and measurement.valid
        and measurement.source_valid
        and value is not None
        and np.isfinite(value)
        and value >= -1e-10 * scale
    )


def collect_case(fixture: ContactFixture) -> SelectionCase:
    """Keep intake failure, every grid attempt, and independent interval errors."""
    motions = candidate_motions(fixture)
    volume = float(
        sum(np.prod(np.asarray(c.upper) - c.lower) for c in fixture.moving_cells)
    )
    case = SelectionCase(fixture, volume, VolumeMeasurement(0.0), ())
    started = perf_counter()
    try:
        cells = build_components(fixture)
        prepared = prepare_planar_contact_set_reference(
            cells.fixed, cells.moving, mode="zero_residual"
        )
        packed = prepare_planar_contact_set(prepared.patches, capacity=24)
        case.preparation_seconds = perf_counter() - started
        case.patch_count = len(prepared.patches)
        arrays = [
            jnp.asarray([getattr(m.motion, name) for m in motions])
            for name in ("velocity", "angular_velocity", "pivot")
        ]
        start = perf_counter()
        rates = _BATCH(packed, *arrays)
        rates.block_until_ready()
        case.first_query_seconds = perf_counter() - start
        start = perf_counter()
        _BATCH(packed, *arrays).block_until_ready()
        case.warm_query_seconds = perf_counter() - start
        case.rates = tuple(map(float, rates))
    except Exception as error:
        case.preparation_seconds = perf_counter() - started
        case.preparation_error = f"{type(error).__name__}: {error}"
    case.initial = measure_volume(fixture, Motion(), 0.0)
    bounds = tuple(
        np.asarray([getattr(c, attr) for c in cells])
        for cells in (fixture.fixed_cells, fixture.moving_cells)
        for attr in ("lower", "upper")
    )
    endpoints = []
    for direction, candidate in enumerate(motions):
        for raw_step in STEPS:
            step = float(raw_step)
            measured = measure_volume(fixture, candidate.motion, step)
            exact = None
            discrepancy = None
            if np.linalg.norm(candidate.motion.angular_velocity) == 0:
                exact = boxes_overlap_volume(
                    bounds[0],
                    bounds[1],
                    bounds[2],
                    bounds[3],
                    step * np.asarray(candidate.motion.velocity),
                )
                if measured.working_volume is not None:
                    discrepancy = abs(exact - measured.working_volume)
            endpoints.append(Endpoint(direction, step, measured, exact, discrepancy))
    case.endpoints = tuple(endpoints)
    return case


@dataclass
class SelectionOracle:
    """Offline cache; only the query result is supplied to selection policies."""

    offgrid: dict[tuple[str, int, float], VolumeMeasurement] = field(
        default_factory=dict
    )
    edits: dict[tuple[str, int, float], EditMeasurement] = field(default_factory=dict)

    def query(self, case: SelectionCase, direction: int, step: float) -> Observation:
        key = (case.fixture.group_id, direction, step)
        endpoint = next(
            (e for e in case.endpoints if e.direction == direction and e.step == step),
            None,
        )
        if endpoint is not None:
            measured = endpoint.measurement
        else:
            if key not in self.offgrid:
                motion = candidate_motions(case.fixture)[direction].motion
                self.offgrid[key] = measure_volume(case.fixture, motion, step)
            measured = self.offgrid[key]
        ok = valid_measurement(measured, case.cutter_volume)
        volume = measured.working_volume
        return Observation(
            direction,
            step,
            max(0.0, volume) if ok and volume is not None else None,
            None if ok else measured.error or "Invalid CAD observation",
            measured.total_seconds,
            measured.common_calls,
            measured.fusion_calls,
        )

    def validate(self, case: SelectionCase, chosen: Observation) -> EditMeasurement:
        key = (case.fixture.group_id, chosen.direction, chosen.step)
        if key not in self.edits:
            motion = candidate_motions(case.fixture)[chosen.direction].motion
            self.edits[key] = apply_cut(case.fixture, motion, chosen.step).measurement
        return self.edits[key]

    def records(self) -> dict[str, list[dict[str, Any]]]:
        return {
            name: [
                {
                    "group_id": key[0],
                    "direction": key[1],
                    "step": key[2],
                    "measurement": asdict(value),
                }
                for key, value in records.items()
            ]
            for name, records in (("offgrid", self.offgrid), ("edits", self.edits))
        }


def placement_check(fixture: ContactFixture) -> dict[str, Any]:
    """Check selected finite paths in two frames, separately from cohort metrics.

    This is global rigid/unit covariance of fresh CAD observations. It neither
    reparses transformed contacts nor changes a pivot with only first-order
    compensation. The complete finite-motion path stays fixed.
    """
    placement = Placement(2.0, (3.0, -2.0, 4.0), (0.2, 0.4, -0.3))
    motions = candidate_motions(fixture)
    rows = []
    for index in (1, 4, 12, 22):
        reference = apply_cut(fixture, motions[index].motion, 0.02).measurement
        transformed = apply_cut(
            fixture, motions[index].motion, 0.02, placement
        ).measurement
        delta = None
        if (
            reference.working_volume is not None
            and transformed.working_volume is not None
        ):
            delta = abs(reference.working_volume - transformed.working_volume)
        passed = bool(
            reference.error is None
            and transformed.error is None
            and reference.cut_valid
            and transformed.cut_valid
            and delta is not None
            and delta <= 1e-8 * max(1.0, reference.moving_volume or 0.0)
        )
        rows.append(
            {
                "direction": index,
                "reference": asdict(reference),
                "transformed": asdict(transformed),
                "working_volume_error": delta,
                "passed": passed,
            }
        )
    return {
        "group_id": fixture.group_id,
        "passed": all(r["passed"] for r in rows),
        "rows": rows,
    }
