"""Finite stock edits agree with an independent interval-volume oracle."""

from dataclasses import replace

import numpy as np
import pytest

from benchmarks.face_contact.cad import Motion, Placement
from benchmarks.multi_contact.cad import apply_cut, measure_volume
from benchmarks.multi_contact.experiment import validate_selections
from benchmarks.multi_contact.fixtures import Box, fixtures
from benchmarks.multi_contact.references import boxes_overlap_volume


@pytest.mark.parametrize("fixture", fixtures(1), ids=lambda item: item.family)
@pytest.mark.parametrize(
    "placement",
    [Placement(), Placement(0.1, (0.3, -0.2, 0.4), (0.2, 0.4, -0.3))],
    ids=["local", "scaled_and_rotated"],
)
def test_rebuilt_cut_matches_interval_volume_and_preserves_stock_balance(
    fixture, placement
):
    motion = Motion((1.0, 0.0, -1.0), pivot=fixture.pivot)
    step = 0.02
    expected = boxes_overlap_volume(
        [box.lower for box in fixture.fixed_cells],
        [box.upper for box in fixture.fixed_cells],
        [box.lower for box in fixture.moving_cells],
        [box.upper for box in fixture.moving_cells],
        step * np.array(motion.velocity),
    )
    result = apply_cut(fixture, motion, step, placement)
    measurement = result.measurement
    assert measurement.error is None
    assert measurement.source_valid and measurement.valid and measurement.cut_valid
    assert result.shape is not None
    assert measurement.cut_solid_count == 1
    assert measurement.working_volume == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert measurement.working_balance_error == pytest.approx(0.0, abs=1e-9)
    assert measurement.common_calls == 1 and measurement.cut_calls == 1
    assert measurement.fusion_calls == len(fixture.fixed_cells) - 1
    assert (
        measurement.total_seconds
        >= measurement.common_seconds + measurement.cut_seconds
    )


def test_failed_source_build_remains_an_explicit_attempt_without_an_edit():
    original = fixtures(1)[0]
    malformed = replace(original, moving_cells=(Box((0, 0, 0), (1, 1, 0)),))
    observed = measure_volume(malformed, Motion(), 0.0)
    assert observed.error is not None and "positive box dimensions" in observed.error
    assert observed.working_volume is None
    assert observed.common_calls == 0
    result = apply_cut(malformed, Motion(), 0.0)
    assert result.shape is None and result.measurement.error is not None
    assert result.measurement.cut_calls == 0


def test_selected_edit_validation_reuses_geometry_and_retains_rejected_amount():
    selection = {
        "group_id": fixtures(1)[0].group_id,
        "placement_name": "identity",
        "direction": 1,
        "step": 0.02,
        "method": "full_contact",
    }
    result = validate_selections(
        [
            selection,
            {**selection, "method": "another_prediction"},
            {**selection, "step": -0.01},
            {**selection, "actual_volume": 1e9},
        ],
        variants=1,
    )
    assert len(result["selections"]) == 4 and len(result["edits"]) == 2
    assert result["selections"][0]["edit_key"] == result["selections"][1]["edit_key"]
    assert result["actual_common_calls"] == 1 and result["actual_cut_calls"] == 1
    assert result["edits"][0]["passed"]
    assert not result["passed"] and len(result["errors"]) == 2
    assert "outside the predeclared grid" in result["errors"][0]["error"]
    assert "disagrees with selection truth" in result["errors"][1]["error"]
