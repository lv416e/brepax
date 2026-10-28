"""Construction-frame accuracy without claiming recovery of source precision."""

import json

import numpy as np

from benchmarks.face_contact.cad import Fixture, Motion, Placement
from benchmarks.face_contact.resolution import resolution_sweep
from benchmarks.face_contact.stress import SQUARE
from brepax.experimental.applications.cad_frame import CadFrame


def test_fresh_design_reconstruction_resolves_both_sides_of_tiny_gap_event():
    placement = Placement(1e-3, (0.013, -0.007, 0.004), (0.2, 0.4, -0.3))
    steps = (5e-9, 1.1e-8, 2e-8)
    result = resolution_sweep(
        Fixture(SQUARE, gap=1e-8),
        Motion((0.0, 0.0, -1.0)),
        steps,
        placement=placement,
        frame=CadFrame(1e-7, placement.origin),
        route="reconstructed_design",
    )
    samples = result["samples"]
    assert all(sample["error"] is None and sample["valid"] for sample in samples)
    assert samples[0]["working_volume"] == 0
    expected_volumes = 4 * np.maximum(np.asarray(steps) - 1e-8, 0)
    np.testing.assert_allclose(
        [sample["working_difference"] for sample in samples[1:]],
        expected_volumes / steps,
        atol=2e-7,
    )
    # Returned physical volumes retain cubic external-unit scaling.
    np.testing.assert_allclose(
        [sample["volume"] for sample in samples[1:]],
        expected_volumes * placement.scale**3,
        atol=2e-24,
    )
    assert not result["source_cad_uncertainty_resolved"]
    json.dumps(result, allow_nan=False)
    transformed = resolution_sweep(
        Fixture(SQUARE, gap=1e-8),
        Motion((0.0, 0.0, -1.0)),
        (2e-8,),
        placement=placement,
        frame=CadFrame(1e-7, placement.origin),
        route="transformed_source",
    )
    # Merely changing B-Rep coordinates preserves the original uncertainty in
    # physical units. It must not acquire the fresh design's smaller tolerance.
    for sample in transformed["samples"]:
        assert sample["error"] is None
        np.testing.assert_allclose(
            sample["source_tolerance"],
            sample["original_source_tolerance"],
            rtol=2e-14,
            atol=0,
        )
        assert sample["source_tolerance"] > 1e6 * samples[0]["source_tolerance"]


def test_working_construction_recovers_milliscale_mixed_direction_window():
    placement = Placement(1e-3, (0.013, -0.007, 0.004), (0.2, 0.4, -0.3))
    result = resolution_sweep(
        Fixture(SQUARE),
        Motion((0.2, -0.1, -0.25), (0.5, 0.7, 0.3)),
        (1e-3, 1e-4),
        placement=placement,
        frame=CadFrame(1e-3, placement.origin),
        route="reconstructed_design",
    )
    # Independent vertical integration of max(.25+.7*x-.5*y, 0) on [-1,1]^2.
    expected = 1.4082738095238095238095238095238
    errors = np.abs(
        np.array([sample["working_difference"] for sample in result["samples"][1:]])
        - expected
    )
    assert errors[1] < errors[0] / 5
    assert errors[1] < 5e-7
    assert all(
        sample["error"] is None and sample["valid"] for sample in result["samples"]
    )


def test_failed_reconstruction_retains_every_step_without_claiming_boolean_calls():
    result = resolution_sweep(
        Fixture(SQUARE, height=-1),
        Motion(),
        (1e-3, 1e-4),
        placement=Placement(),
        frame=CadFrame(1.0),
        route="reconstructed_design",
    )
    assert [sample["step"] for sample in result["samples"]] == [0, 1e-3, 1e-4]
    assert all(sample["error"] for sample in result["samples"])
    assert result["common_call_count"] == 0
    assert all(sample["working_difference"] is None for sample in result["samples"])
    json.dumps(result, allow_nan=False)
