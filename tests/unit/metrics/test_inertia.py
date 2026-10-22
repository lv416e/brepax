"""Unit tests for center of mass and moment of inertia."""

import jax
import jax.numpy as jnp
import pytest

from brepax.brep.csg_eval import make_grid_3d
from brepax.metrics.inertia import (
    center_of_mass,
    integrate_sdf_center_of_mass,
    integrate_sdf_moment_of_inertia,
    moment_of_inertia,
)
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

    def test_sphere_units(self) -> None:
        """Co-scaled domains preserve the analytic centroid in length units."""
        expected = jnp.array([0.3, -0.4, 0.5])

        def compute(scale: jnp.ndarray) -> jnp.ndarray:
            c = scale * expected
            sphere = Sphere(center=c, radius=0.8 * scale)
            return (
                center_of_mass(
                    sphere.sdf, lo=c - 2 * scale, hi=c + 2 * scale, resolution=24
                )
                / scale
            )

        actual = jax.jit(jax.vmap(compute))(jnp.array([1e-9, 1.0, 1e3]))
        assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-12)

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
        """Sphere inertia: I = 2/5 * V * r^2 on diagonal."""
        r = 2.0
        s = Sphere(center=jnp.zeros(3), radius=jnp.array(r))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        inertia = moment_of_inertia(s.sdf, lo=lo, hi=hi, resolution=48)
        vol = 4.0 / 3.0 * jnp.pi * r**3
        expected = 2.0 / 5.0 * vol * r**2
        assert jnp.isclose(inertia[0, 0], expected, rtol=0.05), (
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
        """Box inertia: I_xx = V/12 * (b^2 + c^2)."""
        he = jnp.array([2.0, 1.5, 1.0])
        b = Box(center=jnp.zeros(3), half_extents=he)
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        inertia = moment_of_inertia(b.sdf, lo=lo, hi=hi, resolution=48)
        vol = 8.0 * he[0] * he[1] * he[2]
        _, b2, c2 = (2 * he[0]) ** 2, (2 * he[1]) ** 2, (2 * he[2]) ** 2
        expected_xx = vol / 12.0 * (b2 + c2)
        assert jnp.isclose(inertia[0, 0], expected_xx, rtol=0.05), (
            f"I_xx={float(inertia[0, 0]):.1f}, expected={float(expected_xx):.1f}"
        )


@pytest.mark.parametrize("richardson", [False, True])
def test_public_mass_metrics_similarity(richardson: bool) -> None:
    """Co-transform the SDF and grid; only grid-preserving rotations are exact."""
    rotation = jnp.roll(jnp.eye(3), 1, axis=0)
    shift = jnp.array([0.3, -0.4, 0.5])
    centers = jnp.array([[-0.6, -0.3, 0.1], [0.65, 0.2, 0.4]])
    radii = jnp.array([0.7, 0.45])

    def compute(
        scale: jnp.ndarray, rotate: jnp.ndarray, offset: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        transformed = scale * (centers @ rotate.T) + offset
        a = Sphere(center=transformed[0], radius=scale * radii[0])
        b = Sphere(center=transformed[1], radius=scale * radii[1])

        def sdf(points: jnp.ndarray) -> jnp.ndarray:
            return jnp.minimum(a.sdf(points), b.sdf(points))

        lo, hi = offset - 2 * scale, offset + 2 * scale
        com = center_of_mass(sdf, lo=lo, hi=hi, resolution=24)
        inertia = moment_of_inertia(
            sdf, lo=lo, hi=hi, resolution=24, richardson=richardson
        )
        return com, inertia

    com, inertia = compute(jnp.array(1.0), jnp.eye(3), jnp.zeros(3))
    scales = jnp.array([1e-9, 1.0, 1e3])
    actual_com, actual_inertia = jax.jit(
        jax.vmap(lambda s: compute(s, rotation, s * shift))
    )(scales)
    assert jnp.allclose(
        actual_com / scales[:, None], rotation @ com + shift, rtol=1e-11, atol=1e-12
    )
    assert jnp.allclose(
        actual_inertia / scales[:, None, None] ** 5,
        rotation @ inertia @ rotation.T,
        rtol=1e-11,
        atol=1e-12,
    )


def test_sampled_mass_derivatives() -> None:
    """AD differentiates the sampled sigmoid moments, checked by central FD."""
    lo, hi = -jnp.ones(3), jnp.ones(3)
    grid, _ = make_grid_3d(lo, hi, 2)
    sdf = jnp.linspace(-0.7, 0.9, 8).reshape(2, 2, 2)
    direction = jnp.cos(jnp.arange(8.0)).reshape(sdf.shape)
    probe = jnp.array([0.2, -0.3, 0.7])

    def objective(values: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        com = integrate_sdf_center_of_mass(
            values, scale * grid, scale * lo, scale * hi, 2
        )
        inertia = integrate_sdf_moment_of_inertia(
            values, scale * grid, scale * lo, scale * hi, 2
        )
        return jnp.stack([jnp.dot(probe, com) / scale, jnp.trace(inertia) / scale**5])

    jacobian = jax.jit(jax.jacrev(objective))
    reference = jacobian(sdf, jnp.array(1.0))
    for scale in [1e-9, 1.0, 1e3]:
        actual = jacobian(scale * sdf, jnp.array(scale))
        assert jnp.allclose(scale * actual, reference, rtol=1e-12, atol=1e-12)
    derivative = jnp.sum(reference * direction, axis=(1, 2, 3))
    for step in [1e-3, 1e-4, 1e-5]:
        finite_difference = (
            objective(sdf + step * direction, jnp.array(1.0))
            - objective(sdf - step * direction, jnp.array(1.0))
        ) / (2 * step)
        assert jnp.allclose(derivative, finite_difference, rtol=2e-6, atol=1e-10)


def test_zero_sampled_mass() -> None:
    """A zero-mass centroid is undefined; inertia about a finite point is zero."""
    lo, hi = -jnp.ones(3), jnp.ones(3)
    grid, _ = make_grid_3d(lo, hi, 2)
    sdf = jnp.full((2, 2, 2), 1e6)
    com = integrate_sdf_center_of_mass(sdf, grid, lo, hi, 2)
    inertia = integrate_sdf_moment_of_inertia(sdf, grid, lo, hi, 2)
    assert jnp.all(jnp.isnan(com))
    assert jnp.all(jnp.isnan(inertia))
    about_origin = integrate_sdf_moment_of_inertia(sdf, grid, lo, hi, 2, jnp.zeros(3))
    assert jnp.array_equal(about_origin, jnp.zeros((3, 3)))


@pytest.mark.parametrize("dtype, distance", [(jnp.float32, 60.0), (jnp.float64, 400.0)])
def test_small_positive_sampled_mass(dtype: type, distance: float) -> None:
    """Positive tail weights retain finite, accurate centroid derivatives."""
    lo, hi = -jnp.ones(3, dtype=dtype), jnp.ones(3, dtype=dtype)
    grid, _ = make_grid_3d(lo, hi, 2)
    sdf = distance + jnp.arange(8, dtype=dtype).reshape(2, 2, 2) / 10

    def objective(values: jnp.ndarray) -> jnp.ndarray:
        com = integrate_sdf_center_of_mass(values, grid, lo, hi, 2)
        inertia = integrate_sdf_moment_of_inertia(values, grid, lo, hi, 2)
        # Normalize inertia for an accuracy check at this small mass.
        return jnp.stack([jnp.sum(com), jnp.trace(inertia) * jnp.exp(distance)])

    direction = jnp.cos(jnp.arange(8, dtype=dtype)).reshape(sdf.shape)
    derivative = jnp.sum(
        jax.jit(jax.jacrev(objective))(sdf) * direction, axis=(1, 2, 3)
    )
    step = 0.02 if dtype == jnp.float32 else 1e-3
    finite_difference = (
        objective(sdf + step * direction) - objective(sdf - step * direction)
    ) / (2 * step)
    assert jnp.allclose(derivative, finite_difference, rtol=0.005, atol=1e-5)
