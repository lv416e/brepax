"""Observable policy budgets, retained failures and independent finite CAD checks."""

import json

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.multi_contact.cad import (
    VolumeMeasurement,
    build_components,
    measure_volume,
)
from benchmarks.multi_contact.fixtures import candidate_motions
from benchmarks.multi_contact.selection_evaluation import evaluate, partitions
from benchmarks.multi_contact.selection_experiment import (
    Endpoint,
    SelectionCase,
    SelectionOracle,
    collect_case,
    placement_check,
)
from benchmarks.multi_contact.selection_fixtures import (
    fixtures,
)
from benchmarks.multi_contact.selection_policies import (
    STEPS,
    Observation,
    action_order,
    best_observation,
    grid_policy,
    refine_policy,
)
from brepax.experimental.planar_contact import planar_contact_set_volume_direction
from brepax.experimental.planar_contact_collection_cad import (
    prepare_planar_contact_set_reference,
)


def test_grid_policy_chooses_measured_candidate_under_budget():
    predictions = np.array([[0.01, 0.02, 0.04, 0.08], [0.02, 0.04, 0.08, 0.16]])
    target = 0.039
    order = action_order(predictions, target)
    observations = grid_policy(
        order,
        target,
        4,
        lambda direction, step: Observation(direction, step, (direction + 1) * step),
        adaptive=False,
    )
    assert len(observations) == 4
    assert len({(o.direction, o.step) for o in observations}) == 4
    chosen = best_observation(observations, target)
    assert chosen is not None and chosen in observations
    assert abs(chosen.volume - target) == pytest.approx(0.019)


def test_all_failed_observations_keep_costs_and_serialize():
    failed = VolumeMeasurement(0.005, error="construction failed before Common")
    initial = VolumeMeasurement(
        0.0, working_volume=0.0, source_valid=True, valid=True, common_calls=1
    )
    cases = tuple(
        SelectionCase(
            fixture,
            1.0,
            initial,
            tuple(Endpoint(i, float(t), failed) for i in range(24) for t in STEPS),
            rates=(0.0,) * 24,
        )
        for fixture in fixtures(2)
    )
    oracle = SelectionOracle()
    result = evaluate(cases, oracle)
    json.dumps(result, allow_nan=False)
    assert not oracle.edits
    for row in result["selections"]:
        assert row["probe_attempts"] == row["budget"]
        assert row["probe_common_calls"] == 0
        assert row["total_common_calls"] == 1
        assert row["total_cut_calls"] == 0
        assert row["outcome"] == "no_valid_probe"
    for row in result["summary"].values():
        assert row["mean_relative_error"] is None
        assert row["five_split_mean_errors"] == [None] * 5
        assert row["unconditional_within_one_percent"] == 0


@pytest.mark.parametrize("variants", [3, 8])
def test_group_partitions_cover_every_geometry_without_leakage(variants):
    cases = tuple(
        SelectionCase(f, 1.0, VolumeMeasurement(0.0), ()) for f in fixtures(variants)
    )
    splits = partitions(cases)
    assert len(splits) == 5
    assert len({tuple(s["test"]) for s in splits}) == min(5, variants)
    heldout = set()
    for split in splits:
        assert len(split["train"]) == 4 * (variants - min(3, variants - 1))
        assert len(split["test"]) == 4 * min(3, variants - 1)
        assert not set(split["train"]) & set(split["test"])
        heldout.update(split["test"])
    assert heldout == {c.fixture.group_id for c in cases}


def test_absent_bracket_does_not_establish_unreachable_target():
    width, thickness = 1 / 512, 1 / 1024
    target = thickness / 2

    def response(direction: int, step: float) -> Observation:
        value = max(0.0, min(step, thickness) - max(step - width, 0.0))
        return Observation(direction, step, value)

    observed = refine_policy(
        [(i, 0) for i in range(24)], target, 8, response, rates=None
    )
    assert len(observed) == 8
    assert all(o.volume == 0 for o in observed)
    assert response(0, target).volume == target


def test_positive_gap_can_create_finite_overlap_with_zero_initial_response():
    fixture = next(f for f in fixtures(1) if f.family == "gap_event")
    candidate = candidate_motions(fixture)[1]
    cells = build_components(fixture)
    prepared = prepare_planar_contact_set_reference(
        cells.fixed, cells.moving, mode="zero_residual"
    )
    response = planar_contact_set_volume_direction(
        prepared.contact_set,
        jnp.asarray(candidate.motion.velocity),
        jnp.asarray(candidate.motion.angular_velocity),
        jnp.asarray(candidate.motion.pivot),
    )
    assert float(response) == 0.0
    actual = measure_volume(fixture, candidate.motion, 0.04)
    assert actual.error is None and actual.valid
    assert actual.working_volume is not None and actual.working_volume > 0


def test_refined_observed_cut_can_beat_the_discrete_grid():
    case = collect_case(fixtures(1, cohort="pilot")[0])
    assert case.preparation_error is None and case.rates is not None
    rates = np.asarray(case.rates)
    target = 0.01 * case.cutter_volume
    order = action_order(rates[:, None] * STEPS, target)
    oracle = SelectionOracle()
    trace = refine_policy(
        order, target, 4, lambda d, t: oracle.query(case, d, t), rates=rates
    )
    chosen = best_observation(trace, target)
    assert chosen is not None
    edit = oracle.validate(case, chosen)
    assert edit.cut_valid and edit.error is None
    assert edit.working_volume is not None and edit.working_balance_error is not None
    assert abs(edit.working_balance_error) < 1e-9
    selected_error = abs(edit.working_volume - target) / target
    grid_error = float(np.min(np.abs(case.volume_matrix() - target)) / target)
    assert selected_error < 0.01
    assert selected_error - grid_error < 0


def test_complete_finite_path_has_global_placement_covariance():
    result = placement_check(fixtures(1, cohort="pilot")[0])
    assert result["passed"], result
