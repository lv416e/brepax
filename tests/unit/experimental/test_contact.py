"""Independent cap formulas and derivative laws for the contact experiment."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brepax.experimental.contact import (
    circle_rectangle_intersection_area,
    plate_with_hole_volume,
)


def _area(parameters):
    return circle_rectangle_intersection_area(
        parameters[:2], parameters[2], parameters[3:5]
    )


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_cap_area_and_all_boundary_fluxes(scale):
    # A unit circle centered half a radius beyond the right edge leaves a
    # sixty-degree half-angle cap. Other rectangle sides do not clip it.
    parameters = jnp.array([2.5, 0.2, 1.0, 2.0, 3.0]) * scale
    value, gradient = jax.jit(jax.value_and_grad(_area))(parameters)
    expected_area = np.pi / 3 - np.sqrt(3) / 4
    expected_gradient = np.array([-np.sqrt(3), 0, 2 * np.pi / 3, np.sqrt(3), 0])
    np.testing.assert_allclose(value / scale**2, expected_area, atol=1e-14)
    np.testing.assert_allclose(gradient / scale, expected_gradient, atol=3e-14)


@pytest.mark.parametrize(
    ("parameters", "area", "gradient"),
    [
        ([0, 0, 1, 2, 3], np.pi, [0, 0, 2 * np.pi, 0, 0]),
        ([3, 0, 1, 2, 3], 0, [0, 0, 0, 0, 0]),
        ([1, 0, 1, 2, 3], np.pi, [0, 0, 2 * np.pi, 0, 0]),
        ([2, 0, 1, 2, 3], np.pi / 2, [-2, 0, np.pi, 2, 0]),
        ([2, 3, 1, 2, 3], np.pi / 4, [-1, -1, np.pi / 2, 1, 1]),
        ([0, 0, 4, 2, 3], 24, [0, 0, 0, 12, 8]),
    ],
)
def test_containment_edge_contact_and_corner_quarter_disk(parameters, area, gradient):
    parameters = jnp.array(parameters, dtype=jnp.float64)
    actual, reverse = jax.value_and_grad(_area)(parameters)
    forward = jax.jacfwd(_area)(parameters)
    np.testing.assert_allclose(actual, area, atol=1e-14)
    np.testing.assert_allclose(reverse, gradient, atol=1e-14)
    np.testing.assert_allclose(forward, gradient, atol=1e-14)


def test_small_circle_chord_does_not_use_cancelling_quadratic_discriminant():
    radius = 1e-9
    parameters = jnp.array([2 + radius / 2, 0, radius, 2, 3])
    distance = float(parameters[0] - parameters[3])
    chord = 2 * np.sqrt((radius - distance) * (radius + distance))
    gradient = jax.grad(_area)(parameters)
    np.testing.assert_allclose(gradient[3], chord, rtol=2e-14)
    np.testing.assert_allclose(gradient[0], -chord, rtol=2e-14)


def test_near_external_contact_resolves_small_cap_and_its_derivative():
    delta = 1e-9
    parameters = jnp.array([3 - delta, 0, 1, 2, 3])
    # Use the actually represented penetration in the independent asymptotic.
    penetration = 3 - float(parameters[0])
    leading_area = 4 * np.sqrt(2) / 3 * penetration**1.5
    leading_gradient = 2 * np.sqrt(2 * penetration)
    value, gradient = jax.value_and_grad(_area)(parameters)
    np.testing.assert_allclose(value, leading_area, rtol=2e-8, atol=0)
    np.testing.assert_allclose(
        gradient[jnp.array([0, 2, 3])],
        leading_gradient * np.array([-1, 1, 1]),
        rtol=2e-8,
    )


def test_value_derivative_consistency_in_all_plate_parameters():
    def objective(p):
        return plate_with_hole_volume(p[:2], p[2], p[3:5], p[5])

    parameters = jnp.array([1.8, 0.8, 1.1, 2, 1.5, 2.2])
    gradient = jax.grad(objective)(parameters)
    errors = []
    for step in (1e-3, 1e-4):
        offsets = jnp.eye(6) * step
        plus = jax.vmap(objective)(parameters + offsets)
        minus = jax.vmap(objective)(parameters - offsets)
        errors.append(float(jnp.max(jnp.abs((plus - minus) / (2 * step) - gradient))))
    assert errors[1] < errors[0] / 50
    assert errors[1] < 2e-7


def test_breakout_volume_has_continuous_radius_derivative():
    def objective(radius):
        return plate_with_hole_volume(
            jnp.array([1.5, 0.2]), radius, jnp.array([3.0, 2.0]), jnp.array(2.0)
        )

    event = jnp.array(1.5)
    value, derivative = jax.jit(jax.value_and_grad(objective))(event)
    np.testing.assert_allclose(value, 48 - 2 * np.pi * 1.5**2, atol=1e-13)
    np.testing.assert_allclose(derivative, -6 * np.pi, atol=1e-13)
    steps = jnp.array([1e-4, 1e-6, 1e-8])
    outside_derivatives = jax.jit(jax.vmap(jax.grad(objective)))(event + steps)
    errors = jnp.abs(outside_derivatives - derivative)
    assert bool(jnp.all(errors[1:] < errors[:-1] / 9))
    assert float(errors[-1]) < 0.002


def test_reflections_and_axis_swap_transform_center_and_extent_gradients():
    parameters = jnp.array([1.8, 0.8, 1.1, 2, 1.5])
    value, gradient = jax.value_and_grad(_area)(parameters)
    reflected = parameters * jnp.array([-1, 1, 1, 1, 1])
    reflection_value, reflection_gradient = jax.value_and_grad(_area)(reflected)
    np.testing.assert_allclose(reflection_value, value, atol=2e-14)
    np.testing.assert_allclose(
        reflection_gradient, gradient * jnp.array([-1, 1, 1, 1, 1]), atol=2e-14
    )
    permutation = jnp.array([1, 0, 2, 4, 3])
    swapped_value, swapped_gradient = jax.value_and_grad(_area)(parameters[permutation])
    np.testing.assert_allclose(swapped_value, value, atol=2e-14)
    np.testing.assert_allclose(swapped_gradient, gradient[permutation], atol=2e-14)


@pytest.mark.parametrize(
    "index,value", [(2, 0), (2, -1), (3, 0), (4, np.inf), (0, np.nan)]
)
def test_invalid_geometry_returns_nan_under_jit(index, value):
    parameters = jnp.array([0, 0, 1, 2, 3], dtype=jnp.float64).at[index].set(value)
    area, gradient = jax.jit(jax.value_and_grad(_area))(parameters)
    assert bool(jnp.isnan(area))
    assert bool(jnp.all(jnp.isnan(gradient)))


def test_float32_preserves_dtype_and_resolves_a_partial_circle():
    parameters = jnp.array([2.5, 0.2, 1, 2, 3], dtype=jnp.float32)
    value, gradient = jax.jit(jax.value_and_grad(_area))(parameters)
    assert value.dtype == gradient.dtype == jnp.float32
    np.testing.assert_allclose(value, np.pi / 3 - np.sqrt(3) / 4, atol=1e-6)
    np.testing.assert_allclose(gradient[0], -np.sqrt(3), atol=1e-6)


def test_invalid_plate_height_returns_nan_value_and_derivative():
    def objective(height):
        return plate_with_hole_volume(
            jnp.zeros(2), jnp.array(1.0), jnp.array([2.0, 3.0]), height
        )

    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(objective)))(
        jnp.array([0.0, -1.0, jnp.nan])
    )
    assert bool(jnp.all(jnp.isnan(values)))
    assert bool(jnp.all(jnp.isnan(gradients)))
