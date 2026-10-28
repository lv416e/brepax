"""Collect complete-contact responses and independently rebuilt finite CAD.

This module measures geometry and edits. Candidate selection and grouped
comparisons live in the evaluator; no target outcome tunes these observations.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.face_contact.cad import Motion, Placement, rotation_matrix
from brepax.experimental.planar_contact import (
    PlanarContactPatch,
    PlanarContactSet,
    planar_contact_set_volume_direction,
    planar_contact_volume_direction,
    prepare_planar_contact_set,
)
from brepax.experimental.planar_contact_collection_cad import (
    ContactPreparationError,
    PreparedPlanarContactSet,
    prepare_planar_contact_set_reference,
)

from .cad import VolumeMeasurement, apply_cut, build_components, measure_volume
from .fixtures import (
    ACTION_STEPS,
    TARGET_VOLUME_FRACTION,
    ContactFixture,
    NamedMotion,
    candidate_motions,
    fixtures,
    placements,
)
from .references import (
    box_relations,
    boxes_overlap_volume,
    directional_reference,
)

PROBE_STEP = 0.005
PACKED_CAPACITY = 16
WARM_REPETITIONS = 5
_BATCH_RESPONSE = jax.jit(
    jax.vmap(planar_contact_set_volume_direction, in_axes=(None, 0, 0, 0))
)
_BATCH_PATCH_RESPONSE = jax.jit(
    jax.vmap(planar_contact_volume_direction, in_axes=(None, 0, 0, 0))
)


def _bounds(
    fixture: ContactFixture,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([box.lower for box in fixture.fixed_cells]),
        np.array([box.upper for box in fixture.fixed_cells]),
        np.array([box.lower for box in fixture.moving_cells]),
        np.array([box.upper for box in fixture.moving_cells]),
    )


def _prepared_record(prepared: PreparedPlanarContactSet) -> dict[str, Any]:
    return {
        "reference_kind": prepared.reference_kind,
        "original_cad_contact_certified": prepared.original_cad_contact_certified,
        "patch_sources": [asdict(item) for item in prepared.patch_sources],
        "cell_pairs": [asdict(item) for item in prepared.cell_pairs],
        "partition_pairs": [asdict(item) for item in prepared.partition_pairs],
        "patch_count": len(prepared.patches),
    }


def _placed_set(contacts: PlanarContactSet, placement: Placement) -> PlanarContactSet:
    rotation = jnp.asarray(rotation_matrix(placement.rotation_vector))
    origin = jnp.asarray(placement.origin)
    return PlanarContactSet(
        contacts.vertices * placement.scale,
        origin + placement.scale * contacts.origins @ rotation.T,
        jnp.einsum("ij,tjk->tik", rotation, contacts.bases),
        contacts.active,
    )


def _motion_arrays(
    motions: Sequence[Motion], placement: Placement
) -> tuple[jax.Array, jax.Array, jax.Array]:
    rotation = np.asarray(rotation_matrix(placement.rotation_vector))
    velocity = np.array([motion.velocity for motion in motions])
    angular = np.array([motion.angular_velocity for motion in motions])
    pivot = np.array([motion.pivot for motion in motions])
    return (
        jnp.asarray(placement.scale * velocity @ rotation.T),
        jnp.asarray(angular @ rotation.T),
        jnp.asarray(
            np.asarray(placement.origin) + placement.scale * pivot @ rotation.T
        ),
    )


def _patch_moments(patch: PlanarContactPatch) -> tuple[float, np.ndarray]:
    vertices = np.asarray(patch.vertices)
    following = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * following[:, 1] - vertices[:, 1] * following[:, 0]
    area = float(np.sum(cross) / 2)
    centroid = np.sum((vertices + following) * cross[:, None], axis=0) / (6 * area)
    return area, np.asarray(patch.origin) + np.asarray(patch.basis) @ centroid


def _signed_flux(
    patches: Sequence[PlanarContactPatch], motions: Sequence[Motion]
) -> list[float]:
    rates = np.zeros(len(motions))
    for patch in patches:
        area, center = _patch_moments(patch)
        normal = np.cross(np.asarray(patch.basis)[:, 0], np.asarray(patch.basis)[:, 1])
        for index, motion in enumerate(motions):
            velocity = np.asarray(motion.velocity) + np.cross(
                motion.angular_velocity, center - np.asarray(motion.pivot)
            )
            rates[index] -= area * float(normal @ velocity)
    return list(map(float, rates))


def _failure(error: Exception) -> dict[str, Any]:
    return {
        "error": f"{type(error).__name__}: {error}",
        "pair": asdict(error.pair)
        if isinstance(error, ContactPreparationError)
        else None,
    }


def _transformed_intake(
    fixture: ContactFixture, placement: Placement
) -> dict[str, Any]:
    started = perf_counter()
    try:
        cells = build_components(fixture, placement=placement)
        prepared = prepare_planar_contact_set_reference(
            cells.fixed, cells.moving, mode="projected_nominal"
        )
        expected = {(p.fixed_index, p.moving_index) for p in fixture.patches}
        observed = {(p.fixed_cell_id, p.moving_cell_id) for p in prepared.patch_sources}
        record = {
            "accepted": True,
            "error": None,
            "design_contact_inventory_matches": observed == expected,
            **_prepared_record(prepared),
        }
    except Exception as error:
        record = {
            "accepted": False,
            "design_contact_inventory_matches": None,
            **_failure(error),
        }
    record["seconds"] = perf_counter() - started
    return record


def _valid_measurement(measurement: VolumeMeasurement, allowance: float) -> bool:
    return bool(
        measurement.error is None
        and measurement.source_valid
        and measurement.valid
        and measurement.working_volume is not None
        and np.isfinite(measurement.working_volume)
        and measurement.working_volume >= -allowance
    )


@dataclass
class _CadObservations:
    initial: VolumeMeasurement
    endpoints: list[dict[str, Any]]
    central_samples: list[list[dict[str, Any]]]
    central: list[float | None]
    errors: list[dict[str, Any]]


def _collect_cad(
    fixture: ContactFixture,
    candidates: Sequence[NamedMotion],
    placement: Placement,
    volume_tie: float,
    cutter_volume: float,
) -> _CadObservations:
    """Observe the finite grid and coordinate probes without selecting actions."""
    bounds = _bounds(fixture)
    row_errors: list[dict[str, Any]] = []
    initial = measure_volume(fixture, Motion(), 0.0, placement)
    if (
        not _valid_measurement(initial, volume_tie)
        or abs(initial.working_volume or 0.0) > volume_tie
    ):
        row_errors.append({"stage": "initial_common", "measurement": asdict(initial)})
    endpoints: list[dict[str, Any]] = []
    for step in ACTION_STEPS:
        for direction_index, candidate in enumerate(candidates):
            observed = measure_volume(fixture, candidate.motion, step, placement)
            entry = {
                "direction": direction_index,
                "direction_name": candidate.name,
                "step": step,
                "measurement": asdict(observed),
            }
            if not _valid_measurement(observed, volume_tie):
                row_errors.append({"stage": "endpoint", **entry})
            if np.linalg.norm(candidate.motion.angular_velocity) == 0:
                exact = boxes_overlap_volume(
                    *bounds, step * np.asarray(candidate.motion.velocity)
                )
                entry["interval_volume"] = exact
                entry["interval_absolute_error"] = (
                    None
                    if observed.working_volume is None
                    else abs(observed.working_volume - exact)
                )
                if (
                    observed.working_volume is not None
                    and abs(observed.working_volume - exact) > 1e-9 * cutter_volume
                ):
                    row_errors.append({"stage": "interval_endpoint", **entry})
            endpoints.append(entry)
    central_samples = []
    central = []
    for axis in range(6):
        pair = []
        for sign in (-1, 1):
            vector = np.zeros(6)
            vector[axis] = sign
            velocity = (float(vector[0]), float(vector[1]), float(vector[2]))
            angular = (float(vector[3]), float(vector[4]), float(vector[5]))
            observed = measure_volume(
                fixture,
                Motion(velocity, angular, fixture.pivot),
                PROBE_STEP,
                placement,
            )
            pair.append(observed)
            if not _valid_measurement(observed, volume_tie):
                row_errors.append(
                    {
                        "stage": "coordinate_probe",
                        "axis": axis,
                        "sign": sign,
                        "measurement": asdict(observed),
                    }
                )
        central_samples.append([asdict(item) for item in pair])
        negative, positive = pair[0].working_volume, pair[1].working_volume
        central.append(
            None
            if negative is None or positive is None
            else (positive - negative) / (2 * PROBE_STEP)
        )
    return _CadObservations(initial, endpoints, central_samples, central, row_errors)


@dataclass
class _SourcePreparation:
    prepared: PreparedPlanarContactSet | None
    packed: PlanarContactSet | None
    record: dict[str, Any]


def _prepare_local_source(fixture: ContactFixture) -> _SourcePreparation:
    """Extract the whole local cell inventory and pack its physical patches."""
    started = perf_counter()
    prepared = None
    packed = None
    try:
        cells = build_components(fixture)
        prepared = prepare_planar_contact_set_reference(
            cells.fixed, cells.moving, mode="zero_residual"
        )
        packed = prepare_planar_contact_set(prepared.patches, capacity=PACKED_CAPACITY)
        record = {
            "group_id": fixture.group_id,
            "accepted": True,
            "error": None,
            **_prepared_record(prepared),
        }
    except Exception as error:
        record = {
            "group_id": fixture.group_id,
            "accepted": False,
            **_failure(error),
        }
    record["seconds"] = perf_counter() - started
    return _SourcePreparation(prepared, packed, record)


@dataclass
class _NumericalQuery:
    values: np.ndarray
    max_normalized_error: float
    timings: dict[str, Any]
    errors: list[dict[str, Any]]


def _query_placement(
    packed: PlanarContactSet,
    motions: Sequence[Motion],
    placement: Placement,
    reference: np.ndarray,
) -> _NumericalQuery:
    """Transform the prepared model and supplied twists, then time real JAX calls."""
    placed = _placed_set(packed, placement)
    motion_arrays = _motion_arrays(motions, placement)
    started = perf_counter()
    full = _BATCH_RESPONSE(placed, *motion_arrays).block_until_ready()
    first_call_seconds = perf_counter() - started
    durations = []
    for _ in range(WARM_REPETITIONS):
        started = perf_counter()
        _BATCH_RESPONSE(placed, *motion_arrays).block_until_ready()
        durations.append(perf_counter() - started)
    values = np.asarray(full) / placement.scale**3
    disagreement = float(np.max(np.abs(values - reference)) / max(reference))
    errors = []
    if not np.isfinite(values).all() or disagreement > 1e-10:
        errors.append({"stage": "full_response", "max_normalized_error": disagreement})
    return _NumericalQuery(
        values,
        disagreement,
        {
            "first_call_seconds": first_call_seconds,
            "warm_seconds": durations,
            "warm_median_seconds": float(np.median(durations)),
            "batch_directions": len(motions),
            "packed_triangle_capacity": PACKED_CAPACITY,
        },
        errors,
    )


def collect_geometry(variants: int = 5) -> dict[str, Any]:
    """Record every group/placement/direction/amount, preserving rejected cases.

    Local design-frame intake supplies the numerical model. Transforming that
    model is explicit and separate from the actual transformed-source intake
    coverage experiment. CAD endpoints are rebuilt from transformed components.
    The full-grid truth and smaller-step probe share immutable measurements;
    their hypothetical online call budgets remain separate from actual reuse.
    """
    records: list[dict[str, Any]] = []
    preparations: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    total_common = 0
    total_fusions = 0
    for fixture in fixtures(variants):
        candidates = candidate_motions(fixture)
        motions = tuple(candidate.motion for candidate in candidates)
        bounds = _bounds(fixture)
        relations = box_relations(*bounds)
        reference = np.array(
            [directional_reference(fixture, motion) for motion in motions]
        )
        reference_total = np.sum(reference, axis=1)
        cutter_volume = float(
            sum(
                np.prod(np.asarray(box.upper) - box.lower)
                for box in fixture.moving_cells
            )
        )
        stock_volume = float(
            sum(
                np.prod(np.asarray(box.upper) - box.lower)
                for box in fixture.fixed_cells
            )
        )
        magnification = max(
            1.0,
            max(
                np.sum(np.abs(motion.velocity)) / fixture.length
                + np.sum(np.abs(motion.angular_velocity))
                for motion in motions
            ),
        )
        volume_tie = (
            256
            * np.finfo(float).eps
            * cutter_volume
            * max(ACTION_STEPS)
            / PROBE_STEP
            * magnification
        )
        source = _prepare_local_source(fixture)
        prep_record = source.record
        preparations.append(prep_record)
        if source.prepared is None or source.packed is None:
            errors.append({"stage": "local_intake", **prep_record})
            for named_placement in placements():
                records.append(
                    {
                        "group_id": fixture.group_id,
                        "family": fixture.family,
                        "fixture": asdict(fixture),
                        "placement_name": named_placement.name,
                        "placement": asdict(named_placement.transform),
                        "target_volume": TARGET_VOLUME_FRACTION * cutter_volume,
                        "cutter_volume": cutter_volume,
                        "stock_volume": stock_volume,
                        "volume_tie_tolerance": float(volume_tie),
                        "motions": [
                            {"name": candidate.name, **asdict(candidate.motion)}
                            for candidate in candidates
                        ],
                        "initial": None,
                        "endpoints": [],
                        "rates": {},
                        "errors": [{"stage": "local_intake", **prep_record}],
                    }
                )
            continue
        prepared, packed = source.prepared, source.packed
        expected_pairs = {
            (row.fixed_index, row.moving_index)
            for row in relations
            if row.kind == "face"
        }
        observed_pairs = {
            (row.fixed_cell_id, row.moving_cell_id) for row in prepared.patch_sources
        }
        if expected_pairs != observed_pairs:
            errors.append(
                {
                    "stage": "contact_inventory",
                    "group_id": fixture.group_id,
                    "expected_pairs": sorted(expected_pairs),
                    "observed_pairs": sorted(observed_pairs),
                }
            )
        local_arrays = _motion_arrays(motions, Placement())
        patch_rates = np.column_stack(
            [
                np.asarray(
                    _BATCH_PATCH_RESPONSE(patch, *local_arrays).block_until_ready()
                )
                for patch in prepared.patches
            ]
        )
        largest = int(
            np.argmax([_patch_moments(patch)[0] for patch in prepared.patches])
        )
        for named_placement in placements():
            placement = named_placement.transform
            row_errors = []
            numerical = _query_placement(packed, motions, placement, reference_total)
            full_values = numerical.values
            disagreement = numerical.max_normalized_error
            row_errors.extend(numerical.errors)
            observed_cad = _collect_cad(
                fixture, candidates, placement, volume_tie, cutter_volume
            )
            initial = observed_cad.initial
            endpoints = observed_cad.endpoints
            central_samples = observed_cad.central_samples
            central = observed_cad.central
            row_errors.extend(observed_cad.errors)
            central_rates = (
                None
                if any(value is None for value in central)
                else np.array(
                    [
                        np.r_[motion.velocity, motion.angular_velocity]
                        for motion in motions
                    ]
                )
                @ np.asarray(central)
            )
            fine = [
                entry["measurement"]["working_volume"]
                for entry in endpoints
                if entry["step"] == PROBE_STEP
            ]
            initial_value = initial.working_volume
            fine_rates = (
                None
                if initial_value is None or any(value is None for value in fine)
                else [(value - initial_value) / PROBE_STEP for value in fine]
            )
            rates = {
                "full_contact": list(map(float, full_values)),
                "first_patch": list(map(float, patch_rates[:, 0])),
                "largest_patch": list(map(float, patch_rates[:, largest])),
                "signed_linear_flux": _signed_flux(prepared.patches, motions),
                "central_coordinate_fd": None
                if central_rates is None
                else list(map(float, central_rates)),
                "cad_directional_fd": fine_rates,
            }
            all_measures = [
                asdict(initial),
                *[entry["measurement"] for entry in endpoints],
                *[item for pair in central_samples for item in pair],
            ]
            total_common += sum(item["common_calls"] for item in all_measures)
            total_fusions += sum(item["fusion_calls"] for item in all_measures)
            records.append(
                {
                    "group_id": fixture.group_id,
                    "family": fixture.family,
                    "fixture": asdict(fixture),
                    "placement_name": named_placement.name,
                    "placement": asdict(placement),
                    "target_volume": TARGET_VOLUME_FRACTION * cutter_volume,
                    "cutter_volume": cutter_volume,
                    "stock_volume": stock_volume,
                    "volume_tie_tolerance": float(volume_tie),
                    "motions": [
                        {"name": candidate.name, **asdict(candidate.motion)}
                        for candidate in candidates
                    ],
                    "initial": asdict(initial),
                    "endpoints": endpoints,
                    "rates": rates,
                    "reference_rates": list(map(float, reference_total)),
                    "reference_patch_rates": reference.tolist(),
                    "observed_patch_rates": patch_rates.tolist(),
                    "reference_pair_inventory": [asdict(item) for item in relations],
                    "max_normalized_response_error": disagreement,
                    "transformed_intake": _transformed_intake(fixture, placement),
                    "kernel_timings": numerical.timings,
                    "central_samples": central_samples,
                    "central_coordinate_fd": central,
                    "central_scores_within_noise": None
                    if central_rates is None
                    else bool(
                        np.max(np.abs(central_rates)) * max(ACTION_STEPS) <= volume_tie
                    ),
                    "errors": row_errors,
                }
            )
            errors.extend(
                {
                    "group_id": fixture.group_id,
                    "placement_name": named_placement.name,
                    **error,
                }
                for error in row_errors
            )
    return {
        "rows": records,
        "preparations": preparations,
        "steps": list(ACTION_STEPS),
        "probe_step": PROBE_STEP,
        "errors": errors,
        "passed": not errors,
        "passed_scope": (
            "Complete local design-frame contact response and independently "
            "rebuilt finite CAD. Actual transformed-source intake coverage "
            "is reported separately and is not asserted by this pass."
        ),
        "counts": {
            "groups_attempted": len(fixtures(variants)),
            "placements_recorded": len(records),
            "actual_common_calls": total_common,
            "actual_fusion_calls": total_fusions,
        },
        "cost_contract": {
            "full_contact": (
                "One actual local contact preparation and one 24-direction batch; "
                "selected Common+Cut validation adds one each."
            ),
            "central_coordinate_fd": (
                "Twelve Common probes, then one selected Common+Cut."
            ),
            "cad_directional_fd": (
                "Twenty-four positive-step Common probes and initial Common, "
                "then one selected Common+Cut. Probe outcomes are reused from "
                "truth in this offline collection."
            ),
            "finite_scan": (
                "Ninety-six Common endpoints plus initial Common, then one "
                "selected Cut (or Common+Cut when independently reconstructed)."
            ),
            "reference_inventory": (
                "The full endpoint grid is offline truth for all methods; "
                "it is not charged to analytic model online cost."
            ),
        },
    }


def validate_selections(
    selections: Sequence[dict[str, Any]], *, variants: int = 5
) -> dict[str, Any]:
    """Execute each distinct selected physical edit once and preserve failures.

    Input records identify group_id, placement_name, direction and step; any
    extra evaluator fields are retained in ``selections``. Steps must be in
    the predeclared grid. This verifies CAD and volume balance, not agreement
    with a requested target; the evaluator reports that separate criterion.
    """
    cases = {fixture.group_id: fixture for fixture in fixtures(variants)}
    positions = {placement.name: placement.transform for placement in placements()}
    unique = {}
    errors = []
    selection_rows = []
    for selection in selections:
        key = (
            selection["group_id"],
            selection["placement_name"],
            selection["direction"],
            float(selection["step"]),
        )
        key_name = "/".join(map(str, key))
        if key not in unique:
            try:
                fixture = cases[key[0]]
                placement = positions[key[1]]
                motions = candidate_motions(fixture)
                if not isinstance(key[2], int) or not 0 <= key[2] < len(motions):
                    raise ValueError("Selected direction index is outside the panel")
                if key[3] not in ACTION_STEPS:
                    raise ValueError("Selected amount is outside the predeclared grid")
                result = apply_cut(fixture, motions[key[2]].motion, key[3], placement)
                record = {
                    "edit_key": key_name,
                    "group_id": key[0],
                    "placement_name": key[1],
                    "direction": key[2],
                    "step": key[3],
                    "measurement": asdict(result.measurement),
                }
                volume = float(
                    sum(
                        np.prod(np.asarray(box.upper) - box.lower)
                        for box in fixture.moving_cells
                    )
                )
                balance = result.measurement.working_balance_error
                passed = bool(
                    _valid_measurement(result.measurement, 1e-10 * volume)
                    and result.measurement.cut_valid
                    and result.shape is not None
                    and balance is not None
                    and abs(balance) <= 1e-9 * volume
                )
                record["passed"] = passed
                if not passed:
                    errors.append(record)
            except Exception as error:
                record = {"edit_key": key_name, "passed": False, **_failure(error)}
                errors.append(record)
            unique[key] = record
        selection_record = {**selection, "edit_key": key_name}
        measurement = unique[key].get("measurement", {})
        observed_volume = measurement.get("working_volume")
        if "actual_volume" in selection and observed_volume is not None:
            volume_error = abs(observed_volume - selection["actual_volume"])
            selection_record["verification_volume_error"] = volume_error
            fixture = cases[key[0]]
            cutter_volume = float(
                sum(
                    np.prod(np.asarray(box.upper) - box.lower)
                    for box in fixture.moving_cells
                )
            )
            if not np.isfinite(volume_error) or volume_error > 1e-9 * cutter_volume:
                errors.append(
                    {
                        **selection_record,
                        "error": "Rebuilt edit volume disagrees with selection truth",
                    }
                )
        selection_rows.append(selection_record)
    return {
        "selections": selection_rows,
        "edits": list(unique.values()),
        "errors": errors,
        "passed": not errors,
        "actual_common_calls": sum(
            record.get("measurement", {}).get("common_calls", 0)
            for record in unique.values()
        ),
        "actual_cut_calls": sum(
            record.get("measurement", {}).get("cut_calls", 0)
            for record in unique.values()
        ),
    }
