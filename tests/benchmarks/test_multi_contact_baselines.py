"""Independent geometry and differentiation checks for sampled unions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.contact_transition.baselines import Temperature
from benchmarks.multi_contact.baselines import estimators, prepare_case, prepare_union
from benchmarks.multi_contact.fixtures import Box, fixtures
from benchmarks.multi_contact.sampling import (
    DIRECTIONAL_FD_STEP,
    measure_directional_differences,
)
from brepax.primitives import Box as BoxPrimitive


def test_union_distance_is_independent_of_an_internal_box_partition() -> None:
    split = (
        Box((-1.0, -1.0, -1.0), (0.0, 1.0, 1.0)),
        Box((0.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
    )
    union = prepare_union(split, cell_capacity=4, face_capacity=12)
    reference = BoxPrimitive(jnp.zeros(3), jnp.ones(3))
    points = np.vstack(
        (np.zeros((1, 3)), np.random.default_rng(7603).uniform(-2, 2, (100, 3)))
    )
    np.testing.assert_allclose(
        union.sdf(jnp.asarray(points)), reference.sdf(jnp.asarray(points)), atol=1e-14
    )
    assert float(union.volume()) == pytest.approx(8.0)
    # The ablation deliberately exposes the artificial seam being removed.
    assert (
        float(prepare_union(split, representation="cell_min").sdf(jnp.zeros(3))) == 0.0
    )
    assert float(union.sdf(jnp.zeros(3))) == -1.0


def test_sampled_volume_rejects_duplicate_material() -> None:
    cell = Box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="disjoint interiors"):
        prepare_union((cell, cell))


def test_method_c_keeps_its_disjoint_dispatch_at_exact_face_contact() -> None:
    geometry = prepare_case(fixtures(1)[0])
    methods = estimators(12)
    q = jnp.zeros(6)
    assert float(jax.jit(methods.classify)(q, geometry)) == 0.0
    gradient = jax.jit(jax.grad(methods.stratum))(q, geometry, 1.0, 1.0)
    np.testing.assert_array_equal(gradient, np.zeros(6))


def test_positive_sampled_differences_retain_nonlinear_opposite_motion_response() -> (
    None
):
    # Opposing slot walls make +x and -x responses symmetric. A single linear
    # gradient cannot produce two positive responses, but finite probes can.
    case = next(case for case in fixtures(1) if case.family == "slot2")
    geometry = prepare_case(case)
    function = jax.jit(estimators(12).sigmoid)
    base = float(function(jnp.zeros(6), geometry, 1.0, 0.5))
    directions = np.array([[1.0, 0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0, 0]])
    measured = measure_directional_differences(
        function,
        geometry,
        directions,
        base_value=base,
        temperature=Temperature(1.0, 0.5),
    )
    # Independent paired values establish the finite-step contract, including
    # subtraction of the smoothed baseline, which is nonzero at true contact.
    plus = float(
        function(jnp.array([DIRECTIONAL_FD_STEP, 0, 0, 0, 0, 0]), geometry, 1.0, 0.5)
    )
    minus = float(
        function(jnp.array([-DIRECTIONAL_FD_STEP, 0, 0, 0, 0, 0]), geometry, 1.0, 0.5)
    )
    expected = (np.array([plus, minus]) - base) / DIRECTIONAL_FD_STEP
    assert measured["directional_fd_error"] is None
    assert measured["directional_fd_forward_calls"] == 2
    np.testing.assert_allclose(measured["directional_fd_rates"], expected, atol=1e-13)
    assert np.all(expected > 0)


@pytest.mark.parametrize("method", ("sigmoid", "smoothing"))
def test_sampled_gradient_matches_its_own_forward_off_ties(method: str) -> None:
    geometry = prepare_case(fixtures(1)[0])
    function = getattr(estimators(12), method)
    forward = jax.jit(function)
    q = jnp.array([0.013, -0.017, -0.007, 0.01, -0.03, 0.02])
    gradient = np.asarray(jax.jit(jax.grad(function))(q, geometry, 0.5, 0.5))
    step = 1e-6
    finite_difference = np.asarray(
        [
            (
                float(forward(q + d, geometry, 0.5, 0.5))
                - float(forward(q - d, geometry, 0.5, 0.5))
            )
            / (2 * step)
            for d in np.eye(6) * step
        ]
    )
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-6, atol=2e-8)
