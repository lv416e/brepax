"""Unit tests for center of mass and moment of inertia."""

import jax
import jax.numpy as jnp

from brepax.metrics.inertia import center_of_mass, moment_of_inertia
from brepax.primitives import Box, Sphere


class TestCenterOfMass:
    """Tests for center_of_mass."""

    def test_centered_sphere(self) -> None:
        """Sphere at origin has CoM at origin."""
        s = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        com = center_of_mass(s.sdf, lo=lo, hi=hi, resolution=32)
        assert jnp.allclose(com, jnp.zeros(3), atol=0.1)

    def test_offset_sphere(self) -> None:
        """Sphere at (2,3,1) has CoM at (2,3,1)."""
        c = jnp.array([2.0, 3.0, 1.0])
        s = Sphere(center=c, radius=jnp.array(1.0))
        lo, hi = jnp.array([-1.0] * 3), jnp.array([5.0] * 3)
        com = center_of_mass(s.sdf, lo=lo, hi=hi, resolution=32)
        assert jnp.allclose(com, c, atol=0.15), f"com={com}"

    def test_centered_box(self) -> None:
        """Box at origin has CoM at origin."""
        b = Box(center=jnp.zeros(3), half_extents=jnp.array([2.0, 1.5, 1.0]))
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)
        com = center_of_mass(b.sdf, lo=lo, hi=hi, resolution=32)
        assert jnp.allclose(com, jnp.zeros(3), atol=0.1)

    def test_differentiable(self) -> None:
        """jax.grad of CoM w.r.t. sphere center is finite."""
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)

        def com_x(c: jnp.ndarray) -> jnp.ndarray:
            s = Sphere(center=c, radius=jnp.array(1.0))
            return center_of_mass(s.sdf, lo=lo, hi=hi, resolution=16)[0]

        grad = jax.grad(com_x)(jnp.array([1.0, 0.0, 0.0]))
        assert jnp.all(jnp.isfinite(grad))

    def test_jit_compatible(self) -> None:
        """center_of_mass works under jax.jit."""
        s = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)

        @jax.jit
        def compute() -> jnp.ndarray:
            return center_of_mass(s.sdf, lo=lo, hi=hi, resolution=16)

        com = compute()
        assert com.shape == (3,)
        assert jnp.all(jnp.isfinite(com))


class TestMomentOfInertia:
    """Tests for moment_of_inertia."""

    def test_sphere_diagonal(self) -> None:
        """Sphere inertia: I = 2/5 * V * r^2 on diagonal (approximate)."""
        r = 2.0
        s = Sphere(center=jnp.zeros(3), radius=jnp.array(r))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        inertia = moment_of_inertia(s.sdf, lo=lo, hi=hi, resolution=48)
        vol = 4.0 / 3.0 * jnp.pi * r**3
        expected = 2.0 / 5.0 * vol * r**2
        # Sigmoid bleeding amplifies at r^2; tolerance is wider than volume
        assert jnp.isclose(inertia[0, 0], expected, rtol=0.25), (
            f"I_xx={float(inertia[0, 0]):.2f}, expected={float(expected):.2f}"
        )

    def test_sphere_symmetric(self) -> None:
        """Sphere inertia tensor is approximately diagonal and symmetric."""
        s = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        inertia = moment_of_inertia(s.sdf, lo=lo, hi=hi, resolution=32)
        assert jnp.allclose(inertia[0, 0], inertia[1, 1], rtol=0.05)
        assert jnp.allclose(inertia[0, 0], inertia[2, 2], rtol=0.05)
        assert jnp.allclose(inertia[0, 1], 0.0, atol=0.5)

    def test_box_analytical(self) -> None:
        """Box inertia: I_xx = V/12 * (b^2 + c^2) (approximate)."""
        he = jnp.array([2.0, 1.5, 1.0])
        b = Box(center=jnp.zeros(3), half_extents=he)
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        inertia = moment_of_inertia(b.sdf, lo=lo, hi=hi, resolution=48)
        vol = 8.0 * he[0] * he[1] * he[2]
        _, b2, c2 = (2 * he[0]) ** 2, (2 * he[1]) ** 2, (2 * he[2]) ** 2
        expected_xx = vol / 12.0 * (b2 + c2)
        assert jnp.isclose(inertia[0, 0], expected_xx, rtol=0.30), (
            f"I_xx={float(inertia[0, 0]):.1f}, expected={float(expected_xx):.1f}"
        )

    def test_differentiable(self) -> None:
        """jax.grad of inertia w.r.t. box half_extents is finite."""
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)

        def i_xx(he: jnp.ndarray) -> jnp.ndarray:
            b = Box(center=jnp.zeros(3), half_extents=he)
            return moment_of_inertia(b.sdf, lo=lo, hi=hi, resolution=16)[0, 0]

        grad = jax.grad(i_xx)(jnp.array([2.0, 1.5, 1.0]))
        assert jnp.all(jnp.isfinite(grad))
