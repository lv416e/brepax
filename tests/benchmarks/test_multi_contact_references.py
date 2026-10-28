"""Independent interval geometry distinguishes contact from finite overlap."""

import numpy as np
import pytest

from benchmarks.multi_contact.references import (
    box_relations,
    boxes_overlap_volume,
    contact_direction,
)


def test_corner_translation_includes_both_faces_and_second_order_correction():
    fixed_lower = [[-2, -2, -1], [1, -2, 0]]
    fixed_upper = [[2, 2, 0], [2, 2, 1.5]]
    moving_lower = np.array([[-1, -1, 0]])
    moving_upper = np.array([[1, 1, 1]])
    bounds = (fixed_lower, fixed_upper, moving_lower, moving_upper)
    relations = box_relations(*bounds)
    assert [(row.normal, row.face_area) for row in relations] == [
        ((0.0, 0.0, 1.0), 4.0),
        ((-1.0, 0.0, 0.0), 2.0),
    ]
    assert contact_direction(*bounds, [1, 0, -1]) == 6
    for step in (0.001, 0.02, 0.2):
        assert boxes_overlap_volume(*bounds, [step, 0, -step]) == pytest.approx(
            6 * step - 2 * step**2, rel=1e-12
        )
    gap = 1e-8
    shift = np.array([-gap, 0, gap])
    gapped = (fixed_lower, fixed_upper, moving_lower + shift, moving_upper + shift)
    assert [row.kind for row in box_relations(*gapped)] == ["gap", "gap"]
    assert contact_direction(*gapped, [1, 0, -1]) == 0
    assert boxes_overlap_volume(*gapped, [gap / 2, 0, -gap / 2]) == 0


def test_opposing_slot_contacts_cannot_be_a_single_linear_gradient():
    bounds = (
        [[-2, -2, -1], [1, -2, -1]],
        [[-1, 2, 2], [2, 2, 2]],
        [[-1, -1, 0]],
        [[1, 1, 1]],
    )
    assert contact_direction(*bounds, [1, 0, 0]) == 2
    assert contact_direction(*bounds, [-1, 0, 0]) == 2
    assert contact_direction(*bounds, [0, 1, 0]) == 0


@pytest.mark.parametrize(
    ("fixed_lower", "fixed_upper", "direction", "kind", "power"),
    [
        ([-1, -1, 0], [0, 0, 1], [-1, -1, 0], "edge", 2),
        ([-1, -1, -1], [0, 0, 0], [-1, -1, -1], "vertex", 3),
    ],
)
def test_lower_dimensional_contacts_have_zero_first_order_volume(
    fixed_lower, fixed_upper, direction, kind, power
):
    bounds = ([fixed_lower], [fixed_upper], [[0, 0, 0]], [[1, 1, 1]])
    assert box_relations(*bounds)[0].kind == kind
    assert contact_direction(*bounds, direction) == 0
    step = 0.1
    assert boxes_overlap_volume(*bounds, step * np.array(direction)) == pytest.approx(
        step**power
    )


def test_reference_rejects_source_multiplicity_and_existing_overlap():
    with pytest.raises(ValueError, match="overlapping interiors"):
        boxes_overlap_volume(
            [[-1, -1, -1], [-1, -1, -1]],
            [[1, 1, 0], [1, 1, 0]],
            [[-1, -1, 0]],
            [[1, 1, 1]],
            [0, 0, 0],
        )
    with pytest.raises(ValueError, match="zero initial interior overlap"):
        contact_direction(
            [[-1, -1, -1]],
            [[1, 1, 0]],
            [[-1, -1, -0.1]],
            [[1, 1, 1]],
            [0, 0, 1],
        )
