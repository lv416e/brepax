"""Target inversion against interval-product and independent section formulas."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from brepax.experimental.polyhedral_targets import (
    TranslationTargetResult,
    polyhedral_translation_range,
    polyhedral_translation_targets,
)
from brepax.experimental.polyhedral_translation import prepare_polyhedral_translation


def _box(
    lower: tuple[float, float, float], upper: tuple[float, float, float]
) -> np.ndarray:
    return np.array(list(product(*zip(lower, upper, strict=True))), dtype=np.float64)


def _steps(result: TranslationTargetResult) -> list[float]:
    assert result.status == "reachable"
    assert result.exact_reachable
    assert all(root.root_count == 1 and root.resolved for root in result.roots)
    return [root.step for root in result.roots if root.step is not None]


@pytest.mark.parametrize("contact_order", [1, 2, 3])
def test_entry_and_exit_roots_with_zero_initial_directional_response(
    contact_order: int,
) -> None:
    fixed = _box((0, 0, 0), (1, 1, 1))
    offset = np.array([1.0] * contact_order + [0.0] * (3 - contact_order))
    prepared = prepare_polyhedral_translation(
        [fixed], [fixed + offset], -offset, max_step=2.0
    )
    target = Fraction(1, 8)
    result = polyhedral_translation_targets(
        prepared, target, step_tolerance=1e-12, volume_tolerance=1e-12
    )
    entry = float(target) ** (1 / contact_order)
    np.testing.assert_allclose(_steps(result), [entry, 2 - entry], atol=1e-12, rtol=0)
    for root in result.roots:
        assert abs(root.volume_residual) <= Fraction(1e-12)
    value_range = polyhedral_translation_range(
        prepared, step_tolerance=1e-12, volume_tolerance=1e-12
    )
    assert value_range.minimum == (0, 0)
    assert value_range.maximum == (1, 1)


def test_positive_gap_and_plateau_intervals_are_complete_and_canonical() -> None:
    prepared = prepare_polyhedral_translation(
        [_box((0, 0, 0), (1, 1, 1))],
        [_box((-4, 0, 0), (-1, 1, 1))],
        (1, 0, 0),
        max_step=6.0,
    )
    for target, expected in ((0, [(0, 1), (5, 6)]), (1, [(2, 4)])):
        result = polyhedral_translation_targets(
            prepared, target, step_tolerance=1e-12, volume_tolerance=0.0
        )
        assert result.status == "reachable"
        assert result.roots == ()
        assert [(i.lower, i.upper) for i in result.intervals] == expected
        assert [i.step for i in result.intervals] == [float(lo) for lo, _ in expected]
    middle = polyhedral_translation_targets(
        prepared, Fraction(1, 2), step_tolerance=1e-12, volume_tolerance=1e-12
    )
    np.testing.assert_allclose(_steps(middle), [1.5, 4.5], atol=1e-12, rtol=0)


def test_disconnected_stock_has_four_target_positions() -> None:
    prepared = prepare_polyhedral_translation(
        [_box((0, 0, 0), (1, 1, 1)), _box((2, 0, 0), (3, 1, 1))],
        [_box((-1, 0, 0), (0, 1, 1))],
        (1, 0, 0),
        max_step=5.0,
    )
    result = polyhedral_translation_targets(
        prepared, 0.5, step_tolerance=1e-12, volume_tolerance=1e-12
    )
    np.testing.assert_allclose(_steps(result), [0.5, 1.5, 2.5, 3.5], atol=1e-12, rtol=0)
    impossible = polyhedral_translation_targets(
        prepared, 1.5, step_tolerance=1e-12, volume_tolerance=1e-12
    )
    assert impossible.status == "unreachable"
    assert not impossible.exact_reachable
    # The two disjoint stock cubes contain volume 2, but this tool reaches only 1.
    assert impossible.roots == impossible.intervals == ()


def test_tangent_target_and_adjacent_float_targets_use_exact_root_counts() -> None:
    fixed = _box((0, 0, 0), (1, 1, 1))
    prepared = prepare_polyhedral_translation(
        [fixed], [fixed + np.array([1, 0, 0])], (-1, 1, 0), max_step=1.0
    )
    # Independent interval products give V(t)=t(1-t), with a double root at 1/2.
    tangent = polyhedral_translation_targets(
        prepared, 0.25, step_tolerance=1e-12, volume_tolerance=0.0
    )
    assert _steps(tangent) == [0.5]
    assert tangent.roots[0].lower == tangent.roots[0].upper == Fraction(1, 2)
    for target, expected in (
        (float(np.nextafter(0.25, 0.0)), 2),
        (float(np.nextafter(0.25, 1.0)), 0),
    ):
        result = polyhedral_translation_targets(
            prepared, target, step_tolerance=1e-12, volume_tolerance=1e-15
        )
        assert sum(root.root_count for root in result.roots) == expected
        assert result.exact_reachable == bool(expected)


def test_float_action_resolution_is_separate_from_exact_root_existence() -> None:
    fixed = _box((0, 0, 0), (1, 1, 1))
    prepared = prepare_polyhedral_translation(
        [fixed], [fixed + np.array([1, 0, 0])], (-1, 1, 0), max_step=1.0
    )
    # Two exact roots lie within one ulp; no binary64 action has exactly this volume.
    target = Fraction(1, 4) - Fraction(1, 2**120)
    result = polyhedral_translation_targets(
        prepared, target, step_tolerance=1e-12, volume_tolerance=0.0
    )
    assert result.status == "unresolved"
    assert result.exact_reachable
    assert sum(root.root_count for root in result.roots) == 2
    assert not any(root.resolved for root in result.roots)
    exhausted = polyhedral_translation_targets(
        prepared, 0.2, step_tolerance=1e-12, volume_tolerance=1e-12, max_refinements=1
    )
    assert exhausted.status == "unresolved"
    assert exhausted.exact_reachable
    assert sum(root.root_count for root in exhausted.roots) == 2


def test_nonrepresentable_event_root_retains_exact_existence() -> None:
    fixed = _box((0, 0, 0), (1, 1, 1))
    prepared = prepare_polyhedral_translation(
        [fixed], [fixed + np.array([2, 0, 0])], (-3, 0, 0), max_step=1.0
    )
    result = polyhedral_translation_targets(
        prepared, 1, step_tolerance=1e-12, volume_tolerance=0.0
    )
    assert result.status == "unresolved"
    assert result.exact_reachable
    assert len(result.roots) == 1
    root = result.roots[0]
    assert root.lower == root.upper == Fraction(2, 3)
    assert root.volume_residual != 0
    approximate = polyhedral_translation_targets(
        prepared, 1, step_tolerance=1e-12, volume_tolerance=1e-14
    )
    assert len(_steps(approximate)) == 1


def test_irrational_interior_maximum_is_enclosed_without_grid_sampling() -> None:
    fixed = _box((0, 0, 0), (1, 1, 1))
    prepared = prepare_polyhedral_translation(
        [fixed], [fixed + np.array([1, 0, 0])], (-1, 0.5, 0.25), max_step=1.0
    )
    value_range = polyhedral_translation_range(
        prepared, step_tolerance=1e-12, volume_tolerance=1e-12
    )
    assert value_range.resolved
    lo, hi = value_range.maximum
    assert lo * lo <= Fraction(4, 27) <= hi * hi
    assert hi - lo <= Fraction(1e-12)
    assert any(c.kind == "stationary" for c in value_range.candidates)
    approximate_maximum = float(2 * np.sqrt(3) / 9)
    for target in (
        float(np.nextafter(approximate_maximum, 0.0)),
        float(np.nextafter(approximate_maximum, 1.0)),
    ):
        result = polyhedral_translation_targets(
            prepared, target, step_tolerance=1e-12, volume_tolerance=1e-14
        )
        expected = Fraction(target) ** 2 < Fraction(4, 27)
        assert result.exact_reachable == expected
        assert sum(root.root_count for root in result.roots) == (2 if expected else 0)
    coarse = polyhedral_translation_range(
        prepared, step_tolerance=1e-14, volume_tolerance=1e-14, max_refinements=1
    )
    assert not coarse.resolved
    assert coarse.maximum[0] ** 2 <= Fraction(4, 27) <= coarse.maximum[1] ** 2


def test_prism_slab_targets_match_independent_section_integrals() -> None:
    stock = np.array([(x, y, z) for x, y in ((0, 0), (1, 0), (1, 1)) for z in (0, 1)])
    prepared = prepare_polyhedral_translation(
        [stock], [_box((-0.375, -1, -1), (-0.125, 2, 2))], (1, 0, 0), max_step=1.75
    )
    # V(t)=F(t-g)-F(t-g-w), F(x)=clip(x,0,1)^2/2, g=1/8,w=1/4.
    result = polyhedral_translation_targets(
        prepared, Fraction(15, 128), step_tolerance=1e-12, volume_tolerance=1e-12
    )
    np.testing.assert_allclose(_steps(result), [23 / 32, 5 / 4], atol=1e-12, rtol=0)
    value_range = polyhedral_translation_range(
        prepared, step_tolerance=1e-12, volume_tolerance=1e-12
    )
    assert value_range.maximum == (Fraction(7, 32),) * 2
    unreachable = polyhedral_translation_targets(
        prepared, Fraction(1, 4), step_tolerance=1e-12, volume_tolerance=1e-12
    )
    assert unreachable.status == "unreachable"


def test_target_actions_preserve_frame_units_and_subdivision() -> None:
    fixed = _box((0, 0, 0), (1, 1, 1))
    moving = fixed + np.array([1, 0, 0])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    for scale in (0.125, 1.0, 8.0):
        cells = [_box((0, 0, 0), (0.25, 1, 1)), _box((0.25, 0, 0), (1, 1, 1))]
        prepared = prepare_polyhedral_translation(
            [scale * c @ rotation.T + (3, 2, 1) for c in cells],
            [scale * moving @ rotation.T + (3, 2, 1)],
            scale * rotation @ np.array([-1, 0, 0]),
            max_step=2.0,
        )
        result = polyhedral_translation_targets(
            prepared,
            scale**3 / 4,
            step_tolerance=1e-12,
            volume_tolerance=scale**3 * 1e-12,
        )
        np.testing.assert_allclose(_steps(result), [0.25, 1.75], atol=1e-12, rtol=0)


def test_invalid_target_query_parameters_are_rejected() -> None:
    fixed = _box((0, 0, 0), (1, 1, 1))
    prepared = prepare_polyhedral_translation(
        [fixed], [fixed + np.array([1, 0, 0])], (-1, 0, 0), max_step=1.0
    )
    for target in (-1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match=r"[Tt]arget"):
            polyhedral_translation_targets(
                prepared, target, step_tolerance=1e-12, volume_tolerance=1e-12
            )
    with pytest.raises(ValueError, match="tolerance"):
        polyhedral_translation_targets(
            prepared, 0.5, step_tolerance=0.0, volume_tolerance=1e-12
        )
    with pytest.raises(ValueError, match="max_refinements"):
        polyhedral_translation_range(
            prepared, step_tolerance=1e-12, volume_tolerance=1e-12, max_refinements=0
        )
