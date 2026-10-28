"""Independent geometry and ranking checks for face-contact comparisons."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.face_contact.baselines import ConvexPrism, rotation_matrix
from benchmarks.face_contact.evaluation import evaluate, select_direction
from brepax.primitives import Box


def test_prism_sdf_matches_independent_box_inside_and_outside() -> None:
    prism = ConvexPrism(
        jnp.array([[-1.0, -0.7], [1.0, -0.7], [1.0, 0.7], [-1.0, 0.7]]),
        jnp.array(0.0),
        jnp.array(1.0),
    )
    box = Box(jnp.array([0.0, 0.0, 0.5]), jnp.array([1.0, 0.7, 0.5]))
    points = jnp.asarray(np.random.default_rng(4103).uniform(-2, 2, (80, 3)))
    np.testing.assert_allclose(prism.sdf(points), box.sdf(points), atol=1e-13)
    assert float(prism.volume()) == pytest.approx(float(box.volume()), abs=1e-14)


def test_rigid_parameterization_has_physical_velocity_at_zero() -> None:
    point = jnp.array([0.8, -0.2, 0.4])
    pivot = jnp.array([-0.1, 0.3, 0.2])
    direction = jnp.array([0.2, 0.4, -0.3, -0.2, 0.6, 0.1])

    def path(t: jax.Array) -> jax.Array:
        return (
            pivot
            + rotation_matrix(t * direction[3:]) @ (point - pivot)
            + t * direction[:3]
        )

    velocity = jax.jit(jax.jacfwd(path))(jnp.array(0.0))
    expected = direction[:3] + jnp.cross(direction[3:], point - pivot)
    np.testing.assert_allclose(velocity, expected, atol=1e-14)


def test_roundoff_in_equal_candidate_scores_does_not_change_selection() -> None:
    epsilon = np.finfo(float).eps
    assert select_direction([0.3, 0.3 + epsilon, 0.1]) == 0
    assert select_direction([0.3, 0.31, 0.1]) == 1
    assert select_direction([0.0, 0.0, 0.0]) == 0
    with pytest.raises(ValueError, match="finite"):
        select_direction([0.3, np.nan])


def test_nonfinite_analytic_audit_row_prevents_a_successful_report() -> None:
    # A NaN in a zero-response audit direction used to be ignored by max().
    bad_row = {
        "group_id": "rectangle_00",
        "direction": "vx_positive",
        "value": np.nan,
        "reference": 0.0,
        "absolute_error": np.nan,
        "normalization": 1.0,
    }
    result = evaluate(
        {"records": [], "own_forward_generic_pose_checks": []},
        {"rows": [bad_row]},
        {"baselines": [], "rows": []},
        {"rows": []},
    )
    assert result["passed"] is False
    assert result["failed_cases"] == 1
