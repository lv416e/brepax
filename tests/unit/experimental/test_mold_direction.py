"""Tests for the mold direction optimizer."""

import jax
import jax.numpy as jnp
import pytest

from brepax.experimental.applications.mold_direction import (
    optimize_mold_direction,
    undercut_volume,
)
from brepax.primitives import Box

# -- Fixtures --


def _l_bracket_sdf():
    """L-bracket: box with a rectangular notch in the +x, +z corner.

    The notch breaks centrosymmetry, creating a clear directional
    preference for the mold pull direction.  Pulling toward -x/-z
    (away from the notch) yields lower undercut than pulling toward
    +x/+z (into the notch walls).
    """
    body = Box(
        center=jnp.array([0.0, 0.0, 0.0]),
        half_extents=jnp.array([1.5, 1.0, 1.0]),
    )
    notch = Box(
        center=jnp.array([1.0, 0.0, 0.5]),
        half_extents=jnp.array([0.6, 1.1, 0.6]),
    )

    def sdf(x):
        return jnp.maximum(body.sdf(x), -notch.sdf(x))

    lo = jnp.array([-2.5, -2.0, -2.0])
    hi = jnp.array([2.5, 2.0, 2.0])
    return sdf, lo, hi


# -- undercut_volume tests --


class TestUndercutVolume:
    """Tests for undercut_volume computation."""

    def test_returns_scalar(self):
        sdf, lo, hi = _l_bracket_sdf()
        vol = undercut_volume(
            sdf, jnp.array([0.0, 0.0, 1.0]), lo=lo, hi=hi, resolution=16
        )
        assert vol.shape == ()

    def test_positive_for_nontrivial_shape(self):
        sdf, lo, hi = _l_bracket_sdf()
        vol = undercut_volume(
            sdf, jnp.array([0.0, 0.0, 1.0]), lo=lo, hi=hi, resolution=16
        )
        assert float(vol) > 0.0

    def test_varies_with_direction(self):
        """Pulling into vs away from the notch should differ."""
        sdf, lo, hi = _l_bracket_sdf()
        res = 32

        # Pull into the notch (+x, +z) — notch walls oppose pull
        vol_into = float(
            undercut_volume(
                sdf,
                jnp.array([1.0, 0.0, 1.0]) / jnp.sqrt(2.0),
                lo=lo,
                hi=hi,
                resolution=res,
            )
        )
        # Pull away from the notch (-x, -z)
        vol_away = float(
            undercut_volume(
                sdf,
                jnp.array([-1.0, 0.0, -1.0]) / jnp.sqrt(2.0),
                lo=lo,
                hi=hi,
                resolution=res,
            )
        )
        assert abs(vol_into - vol_away) > 0.1

    def test_gradient_is_finite(self):
        """Gradient of undercut w.r.t. direction must be finite."""
        sdf, lo, hi = _l_bracket_sdf()
        grad = jax.grad(lambda d: undercut_volume(sdf, d, lo=lo, hi=hi, resolution=16))(
            jnp.array([1.0, 0.0, 0.0])
        )
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_wrt_direction_is_nonzero(self):
        """For a non-convex shape the gradient should not vanish."""
        sdf, lo, hi = _l_bracket_sdf()
        grad = jax.grad(lambda d: undercut_volume(sdf, d, lo=lo, hi=hi, resolution=24))(
            jnp.array([1.0, 0.0, 1.0])
        )
        assert float(jnp.linalg.norm(grad)) > 1e-4


# -- optimize_mold_direction tests --


class TestOptimizeMoldDirection:
    """Tests for the optimization loop."""

    def test_result_direction_is_unit_vector(self):
        sdf, lo, hi = _l_bracket_sdf()
        result = optimize_mold_direction(
            sdf,
            jnp.array([1.0, 1.0, 1.0]),
            lo=lo,
            hi=hi,
            resolution=16,
            steps=10,
        )
        norm = float(jnp.linalg.norm(result.direction))
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_loss_decreases(self):
        """Loss should not increase over the first several steps."""
        sdf, lo, hi = _l_bracket_sdf()
        result = optimize_mold_direction(
            sdf,
            jnp.array([1.0, 0.0, 1.0]),
            lo=lo,
            hi=hi,
            resolution=16,
            steps=30,
            lr=0.1,
        )
        assert len(result.losses) >= 2
        assert result.losses[-1] <= result.losses[0] + 1e-4

    def test_trajectory_shape(self):
        sdf, lo, hi = _l_bracket_sdf()
        result = optimize_mold_direction(
            sdf,
            jnp.array([1.0, 0.0, 0.0]),
            lo=lo,
            hi=hi,
            resolution=16,
            steps=5,
        )
        n_traj = result.trajectory.shape[0]
        assert n_traj >= 2
        assert result.trajectory.shape[1] == 3

    def test_convergence_flag(self):
        """With enough steps, should converge."""
        sdf, lo, hi = _l_bracket_sdf()
        result = optimize_mold_direction(
            sdf,
            jnp.array([-1.0, 0.0, -1.0]),
            lo=lo,
            hi=hi,
            resolution=16,
            steps=200,
            lr=0.1,
            tol=1e-4,
        )
        assert result.converged
