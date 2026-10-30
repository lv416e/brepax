"""Finite-volume and event contracts, checked against independent box laws."""

from fractions import Fraction
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brepax.experimental.polyhedral_translation import (
    polyhedral_translation_slope,
    polyhedral_translation_volume,
    prepare_polyhedral_translation,
)


def _box(
    lower: tuple[float, float, float], upper: tuple[float, float, float]
) -> np.ndarray:
    return np.array(list(product(*(zip(lower, upper, strict=True)))), dtype=np.float64)


@pytest.mark.parametrize("codimension", [1, 2, 3])
def test_contact_order_and_exit_events(codimension: int) -> None:
    box = _box((0, 0, 0), (1, 1, 1))
    offset = np.array([1.0] * codimension + [0.0] * (3 - codimension))
    prepared = prepare_polyhedral_translation(
        [box], [box + offset], -offset, max_step=3.0
    )
    assert prepared.exact_knots == (Fraction(0), Fraction(1), Fraction(2), Fraction(3))
    profile = prepared.profile
    steps = jnp.array([0.0, 0.125, 0.5, 1.0, 1.25, 1.75, 2.0, 2.5])
    expected = np.maximum(0.0, 1.0 - np.abs(np.asarray(steps) - 1.0)) ** codimension
    np.testing.assert_allclose(
        jax.jit(jax.vmap(polyhedral_translation_volume, in_axes=(None, 0)))(
            profile, steps
        ),
        expected,
        atol=1e-15,
    )
    assert float(polyhedral_translation_slope(profile, 0.0)) == (
        1.0 if codimension == 1 else 0.0
    )
    assert float(polyhedral_translation_slope(profile, 1.0, side="left")) == codimension
    assert (
        float(polyhedral_translation_slope(profile, 1.0, side="right")) == -codimension
    )
    gradient = jax.jit(jax.grad(polyhedral_translation_volume, argnums=1))
    for t in (0.25, 0.75, 1.25, 1.75):
        expected_slope = (
            codimension * (1 - abs(t - 1)) ** (codimension - 1) * (1 if t < 1 else -1)
        )
        np.testing.assert_allclose(gradient(profile, t), expected_slope, atol=1e-14)


def test_sliding_face_cubic_and_exact_subdivision() -> None:
    box = _box((0, 0, 0), (1, 1, 1))
    moving = box + np.array([1, 0, 0])
    velocity = (-1.0, 0.5, 0.25)
    whole = prepare_polyhedral_translation([box], [moving], velocity, max_step=2.0)
    split = prepare_polyhedral_translation(
        [_box((0, 0, 0), (1, 0.25, 1)), _box((0, 0.25, 0), (1, 1, 1))],
        [moving],
        velocity,
        max_step=2.0,
    )
    steps = jnp.linspace(0.0, 2.0, 41)
    observed = jax.vmap(polyhedral_translation_volume, in_axes=(None, 0))(
        whole.profile, steps
    )
    expected = (
        jnp.maximum(0, 1 - jnp.abs(steps - 1)) * (1 - steps / 2) * (1 - steps / 4)
    )
    np.testing.assert_allclose(observed, expected, atol=1e-15)
    np.testing.assert_allclose(
        jax.vmap(polyhedral_translation_volume, in_axes=(None, 0))(
            split.profile, steps
        ),
        expected,
        atol=1e-15,
    )
    assert split.pair_count == 2


def test_corner_collection_and_simplex_point_contact() -> None:
    tool = _box((0, 0, 0), (1, 1, 1))
    corner = prepare_polyhedral_translation(
        [_box((-1, 0, 0), (0, 1, 1)), _box((-1, -1, 0), (1, 0, 1))],
        [tool],
        (-1, -1, 0),
        max_step=1.0,
    )
    for t in (0.125, 0.25, 0.75):
        assert float(polyhedral_translation_volume(corner.profile, t)) == 2 * t - t * t
    simplex = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=float)
    prepared = prepare_polyhedral_translation(
        [simplex], [_box((2, 0, 0), (3, 1, 1))], (-1, 0, 0), max_step=1.0
    )
    for t in (0.125, 0.25, 0.75):
        np.testing.assert_allclose(
            polyhedral_translation_volume(prepared.profile, t), t**3 / 6, atol=1e-16
        )


def test_positive_gap_is_an_event_and_nonrepresentable_threshold_is_not_snapped() -> (
    None
):
    box = _box((0, 0, 0), (1, 1, 1))
    # Onset is exactly 1/3, not its nearest binary64 approximation.
    prepared = prepare_polyhedral_translation(
        [box], [box + np.array([2, 0, 0])], (-3, 0, 0), max_step=1.0
    )
    assert prepared.events[0].step == Fraction(1, 3)
    assert prepared.events[0].left_derivative == 0
    assert prepared.events[0].right_derivative == 3
    floor = float(Fraction(1, 3))
    ceiling = float(np.nextafter(floor, np.inf))
    assert float(polyhedral_translation_volume(prepared.profile, floor)) == 0.0
    exact_value = float(3 * Fraction(ceiling) - 1)
    assert (
        float(polyhedral_translation_volume(prepared.profile, ceiling)) == exact_value
    )
    for side in ("left", "right"):
        assert (
            float(polyhedral_translation_slope(prepared.profile, floor, side=side))
            == 0.0
        )
        np.testing.assert_allclose(
            polyhedral_translation_slope(prepared.profile, ceiling, side=side),
            3.0,
            atol=1e-15,
            rtol=0,
        )


def test_narrow_noncollapsed_interval_uses_represented_polynomial_coordinates() -> None:
    fixed = _box((0, 0, 0), (1, 1, 1))
    width = 2.0**-51
    moving = _box((-width, 0, 0), (0, 1, 1))
    prepared = prepare_polyhedral_translation(
        [fixed], [moving], (3, 0, 0), max_step=1.0
    )
    step = float(np.nextafter(float(Fraction(1, 3)), np.inf))
    expected = float(1 + Fraction(width) - 3 * Fraction(step))
    assert expected > 0
    assert float(polyhedral_translation_volume(prepared.profile, step)) == expected
    assert (
        float(polyhedral_translation_slope(prepared.profile, step, side="left")) == -3
    )


def test_unresolvable_event_interval_is_rejected() -> None:
    fixed = _box((0, 0, 0), (1, 1, 1))
    moving = _box((-(2.0**-55), 0, 0), (0, 1, 1))
    with pytest.raises(ValueError, match="resolvable float64 chamber"):
        prepare_polyhedral_translation([fixed], [moving], (1, 0, 0), max_step=2.0)


def test_global_frame_units_and_cell_order() -> None:
    fixed = _box((0, 0, 0), (1, 1, 1))
    moving = fixed + np.array([1, 0, 0])
    velocity = np.array([-1.0, 0.5, 0.25])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    shift = np.array([3.0, -2.0, 4.0])
    original = prepare_polyhedral_translation([fixed], [moving], velocity, max_step=2.0)
    for scale in (0.125, 1.0, 8.0):
        transformed = prepare_polyhedral_translation(
            [scale * fixed[::-1] @ rotation.T + shift],
            [scale * moving[::-1] @ rotation.T + shift],
            scale * rotation @ velocity,
            max_step=2.0,
        )
        assert transformed.exact_knots == original.exact_knots
        for t in (0.125, 0.5, 1.25):
            np.testing.assert_allclose(
                polyhedral_translation_volume(transformed.profile, t),
                scale**3 * polyhedral_translation_volume(original.profile, t),
                atol=1e-13,
            )
            np.testing.assert_allclose(
                polyhedral_translation_slope(transformed.profile, t),
                scale**3 * polyhedral_translation_slope(original.profile, t),
                atol=1e-13,
            )


def test_stationary_and_forever_gapped_paths_return_zero() -> None:
    fixed = _box((0, 0, 0), (1, 1, 1))
    for velocity in ((0, 0, 0), (1, 0, 0)):
        prepared = prepare_polyhedral_translation(
            [fixed], [fixed + np.array([2, 0, 0])], velocity, max_step=2.0
        )
        assert float(polyhedral_translation_volume(prepared.profile, 1.0)) == 0.0
        assert float(polyhedral_translation_slope(prepared.profile, 1.0)) == 0.0


def test_reject_overlaps_and_degenerate_cells() -> None:
    box = _box((0, 0, 0), (1, 1, 1))
    with pytest.raises(ValueError, match=r"fixed cells.*overlapping"):
        prepare_polyhedral_translation(
            [box, box], [box + np.array([2, 0, 0])], (-1, 0, 0), max_step=2.0
        )
    with pytest.raises(ValueError, match=r"moving cells.*overlapping"):
        prepare_polyhedral_translation(
            [box],
            [box + np.array([2, 0, 0]), box + np.array([2, 0, 0])],
            (-1, 0, 0),
            max_step=2.0,
        )
    with pytest.raises(ValueError, match=r"Initial.*overlap"):
        prepare_polyhedral_translation(
            [box], [box + np.array([0.5, 0, 0])], (-1, 0, 0), max_step=2.0
        )
    with pytest.raises(ValueError, match="positive-volume"):
        prepare_polyhedral_translation(
            [box * (1, 1, 0)], [box + np.array([2, 0, 0])], (-1, 0, 0), max_step=2.0
        )
    with pytest.raises(ValueError, match="at least one"):
        prepare_polyhedral_translation([], [box], (-1, 0, 0), max_step=2.0)


@pytest.mark.parametrize("max_step", [0.0, -1.0, np.inf, np.nan])
def test_reject_invalid_interval(max_step: float) -> None:
    box = _box((0, 0, 0), (1, 1, 1))
    with pytest.raises(ValueError, match="max_step"):
        prepare_polyhedral_translation(
            [box], [box + np.array([1, 0, 0])], (-1, 0, 0), max_step=max_step
        )


def test_query_bounds_and_missing_one_sided_limits() -> None:
    box = _box((0, 0, 0), (1, 1, 1))
    prepared = prepare_polyhedral_translation(
        [box], [box + np.array([1, 0, 0])], (-1, 0, 0), max_step=2.0
    )
    profile = prepared.profile
    for step in (-1.0, 3.0, np.nan, np.inf):
        assert np.isnan(polyhedral_translation_volume(profile, step))
        assert np.isnan(polyhedral_translation_slope(profile, step))
    assert np.isnan(polyhedral_translation_slope(profile, 0.0, side="left"))
    assert np.isnan(polyhedral_translation_slope(profile, 2.0, side="right"))
    assert float(polyhedral_translation_volume(profile, 2.0)) == 0.0


@pytest.mark.parametrize("scale", [1e-110, 1e110])
def test_unrepresentable_volume_scale_is_rejected(scale: float) -> None:
    box = _box((0, 0, 0), (scale, scale, scale))
    with pytest.raises(ValueError, match=r"underflow|representable"):
        prepare_polyhedral_translation(
            [box],
            [box + np.array([scale, 0.0, 0.0])],
            (-scale, 0.0, 0.0),
            max_step=1.0,
        )
