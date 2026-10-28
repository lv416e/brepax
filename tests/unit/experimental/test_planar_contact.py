"""Closed-form overlap layers and rigid-motion laws at planar contact."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brepax.experimental.planar_contact import (
    planar_contact_volume_direction,
    prepare_planar_contact_patch,
)

SQUARE = np.array([[-1.0, -1], [1, -1], [1, 1], [-1, 1]])
BASIS = np.eye(3)[:, :2]


def test_opposite_tilts_are_positive_and_cannot_be_a_linear_jvp():
    patch = prepare_planar_contact_patch(SQUARE, origin=np.zeros(3), basis=BASIS)
    motions = jnp.array(
        [
            [0, 0, -1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, -1, 0, 0],
            [1, 2, 0, 0, 0, 3],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=jnp.float64,
    )
    evaluate = jax.jit(
        jax.vmap(
            lambda d: planar_contact_volume_direction(patch, d[:3], d[3:], jnp.zeros(3))
        )
    )
    np.testing.assert_allclose(evaluate(motions), [4, 0, 1, 1, 0, 0], atol=1e-14)


@pytest.mark.parametrize("epsilon", [0.1, 1e-6, 1e-12])
def test_small_triangle_caps_and_strips_do_not_cancel(epsilon):
    patch = prepare_planar_contact_patch(
        [[0, 0], [1, 0], [0, 1]], origin=np.zeros(3), basis=BASIS
    )
    translation = jnp.array([0.0, 0, -epsilon])
    # Integrate (epsilon-y)+ and (epsilon-x-y)+ directly over x+y <= 1.
    strip = planar_contact_volume_direction(
        patch, translation, jnp.array([1.0, 0, 0]), jnp.zeros(3)
    )
    corner = planar_contact_volume_direction(
        patch, translation, jnp.array([1.0, -1, 0]), jnp.zeros(3)
    )
    np.testing.assert_allclose(strip, epsilon**2 / 2 - epsilon**3 / 6, rtol=2e-14)
    np.testing.assert_allclose(corner, epsilon**3 / 6, rtol=2e-14)


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_rigid_frame_units_and_compensated_pivot(scale):
    vertices = np.array([[-1.0, -0.5], [1, -0.5], [0.7, 1], [-0.2, 1.2]])
    origin = np.array([0.2, -0.3, 0.1])
    pivot = np.array([0.7, 0.2, -0.6])
    linear = np.array([0.4, -0.3, -0.2])
    angular = np.array([0.7, -0.4, 0.2])
    patch = prepare_planar_contact_patch(vertices, origin=origin, basis=BASIS)
    expected = planar_contact_volume_direction(
        patch, jnp.asarray(linear), jnp.asarray(angular), jnp.asarray(pivot)
    )
    # A non-axis-aligned proper orthogonal matrix, with exactly known entries.
    rotation = np.array([[1, -2, 2], [2, 2, 1], [-2, 1, 2]]) / 3
    offset = np.array([2.0, -3, 5]) * scale
    moved = prepare_planar_contact_patch(
        vertices * scale,
        origin=rotation @ origin * scale + offset,
        basis=rotation @ BASIS,
    )
    moved_pivot = rotation @ pivot * scale + offset
    moved_linear = rotation @ linear * scale
    moved_angular = rotation @ angular
    result = jax.jit(planar_contact_volume_direction)(
        moved, moved_linear, moved_angular, moved_pivot
    )
    np.testing.assert_allclose(result / scale**3, expected, rtol=2e-14)
    shifted_pivot = moved_pivot + np.array([0.2, -0.3, 0.4]) * scale
    shifted_linear = moved_linear + np.cross(moved_angular, shifted_pivot - moved_pivot)
    equivalent = planar_contact_volume_direction(
        moved,
        jnp.asarray(shifted_linear),
        jnp.asarray(moved_angular),
        jnp.asarray(shifted_pivot),
    )
    np.testing.assert_allclose(equivalent, result, rtol=2e-14)


def test_positive_homogeneity_and_subadditivity():
    patch = prepare_planar_contact_patch(SQUARE, origin=np.zeros(3), basis=BASIS)
    directions = jnp.array(
        [[0.2, 0.3, -0.2, 0.4, -0.8, 0], [-0.1, 0, 0.3, 0.8, 0.2, 0]]
    )

    def response(d):
        return planar_contact_volume_direction(patch, d[:3], d[3:], jnp.zeros(3))

    individual = jax.vmap(response)(directions)
    np.testing.assert_allclose(
        jax.vmap(response)(directions * 3), individual * 3, atol=1e-14
    )
    assert response(directions.sum(axis=0)) <= individual.sum()


def test_small_positive_cap_does_not_underflow_before_dimensional_rescaling():
    patch = prepare_planar_contact_patch(
        [[-1, 0], [0, -1], [0, 0]], origin=np.zeros(3), basis=BASIS
    )
    result = planar_contact_volume_direction(
        patch,
        jnp.array([0.0, 0, -1e90]),
        jnp.array([-1e200, 1e200, 0]),
        jnp.zeros(3),
    )
    # A triangular cap with intercept 1e-110 and peak height 1e90.
    np.testing.assert_allclose(result, 1e-130 / 6, rtol=2e-14, atol=0)


def test_representable_large_response_avoids_intermediate_overflow():
    patch = prepare_planar_contact_patch(
        [[0, -1], [1, 1], [-1, 1]], origin=np.zeros(3), basis=BASIS
    )
    result = planar_contact_volume_direction(
        patch, jnp.zeros(3), jnp.array([-1e308, 0, 0]), jnp.zeros(3)
    )
    # Integrate 1e308*y over 0<=y<=1, with horizontal width 1+y.
    np.testing.assert_allclose(result / 1e308, 5 / 6, rtol=2e-14)


@pytest.mark.parametrize(
    "vertices",
    [
        [[0, 0], [1, 0]],
        [[0, 0], [1, 0], [2, 0]],
        SQUARE[::-1],
        [[0, 0], [2, 0], [0.5, 0.5], [2, 2], [0, 2]],
        [[0, 0], [2, 2e-14], [0, 2e-14], [1, 0]],
        np.array([[0, 0], [3, 1e-15], [1, 2e-15], [2, 0]])
        @ (np.array([[1.0, 1], [-1, 1]]) / np.sqrt(2)),
        np.concatenate([SQUARE, SQUARE]),
        [[0, 0], [np.nan, 0], [0, 1]],
    ],
)
def test_host_rejects_invalid_polygon_domain(vertices):
    with pytest.raises(ValueError):
        prepare_planar_contact_patch(vertices, origin=np.zeros(3), basis=BASIS)


def test_host_requires_orthonormal_frame_and_x64():
    with pytest.raises(ValueError, match="orthonormal"):
        prepare_planar_contact_patch(SQUARE, origin=np.zeros(3), basis=2 * BASIS)
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(ValueError, match="x64"):
            prepare_planar_contact_patch(SQUARE, origin=np.zeros(3), basis=BASIS)
    finally:
        jax.config.update("jax_enable_x64", True)


def test_thin_convex_patch_is_distinguished_from_a_self_crossing_sliver():
    patch = prepare_planar_contact_patch(
        [[0, 0], [1, 0], [1, 1e-14], [0, 1e-14]],
        origin=np.zeros(3),
        basis=BASIS,
    )
    result = planar_contact_volume_direction(
        patch, jnp.array([0.0, 0, -1]), jnp.zeros(3), jnp.zeros(3)
    )
    np.testing.assert_allclose(result, 1e-14, rtol=2e-14, atol=0)


def test_nonfinite_motion_is_not_reported_as_zero():
    patch = prepare_planar_contact_patch(SQUARE, origin=np.zeros(3), basis=BASIS)
    evaluate = jax.jit(planar_contact_volume_direction)
    for argument in range(3):
        values = [jnp.zeros(3), jnp.zeros(3), jnp.zeros(3)]
        values[argument] = jnp.array([0.0, 0, jnp.nan])
        assert np.isnan(evaluate(patch, *values))
