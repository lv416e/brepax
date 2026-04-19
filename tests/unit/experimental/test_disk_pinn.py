"""Tests for the unit-disk Poisson PINN solver."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from brepax.experimental.physics.poisson_pinn import (
    DiskPoissonPINN,
    disk_analytical,
    disk_pinn_loss,
    disk_sdf,
    evaluate_disk_pinn,
    sample_disk_boundary,
    sample_disk_interior,
    train_disk_pinn,
)


class TestDiskSdf:
    """Verify SDF geometry properties."""

    def test_origin_is_inside(self) -> None:
        pts = jnp.array([[0.0, 0.0]])
        assert float(disk_sdf(pts)[0]) == pytest.approx(-1.0)

    def test_boundary_is_zero(self) -> None:
        theta = jnp.linspace(0, 2 * jnp.pi, 100)
        pts = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
        vals = disk_sdf(pts)
        assert jnp.allclose(vals, 0.0, atol=1e-6)

    def test_exterior_is_positive(self) -> None:
        pts = jnp.array([[2.0, 0.0], [0.0, 1.5]])
        assert jnp.all(disk_sdf(pts) > 0)


class TestAnalytical:
    """Verify the analytical solution satisfies the PDE."""

    def test_max_at_origin(self) -> None:
        pts = jnp.array([[0.0, 0.0]])
        assert float(disk_analytical(pts)[0]) == pytest.approx(0.25)

    def test_zero_on_boundary(self) -> None:
        theta = jnp.linspace(0, 2 * jnp.pi, 100)
        pts = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
        vals = disk_analytical(pts)
        assert jnp.allclose(vals, 0.0, atol=1e-6)


class TestSampling:
    """Verify collocation point samplers produce valid geometry."""

    def test_interior_points_inside(self) -> None:
        key = jax.random.PRNGKey(0)
        pts = sample_disk_interior(200, key)
        sdf_vals = disk_sdf(pts)
        assert jnp.all(sdf_vals < 0)

    def test_boundary_points_on_circle(self) -> None:
        key = jax.random.PRNGKey(1)
        pts = sample_disk_boundary(100, key)
        r = jnp.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        assert jnp.allclose(r, 1.0, atol=1e-6)


class TestLoss:
    """Verify loss function computes and gradients flow."""

    def test_loss_is_scalar(self) -> None:
        key = jax.random.PRNGKey(2)
        k1, k2, k3 = jax.random.split(key, 3)
        model = DiskPoissonPINN(width=16, depth=2, key=k1)
        interior = sample_disk_interior(50, k2)
        boundary = sample_disk_boundary(20, k3)
        loss = disk_pinn_loss(model, interior, boundary)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_gradients_are_finite(self) -> None:
        key = jax.random.PRNGKey(3)
        k1, k2, k3 = jax.random.split(key, 3)
        model = DiskPoissonPINN(width=16, depth=2, key=k1)
        interior = sample_disk_interior(50, k2)
        boundary = sample_disk_boundary(20, k3)
        grads = eqx.filter_grad(disk_pinn_loss)(model, interior, boundary)
        leaves = jax.tree.leaves(grads)
        for leaf in leaves:
            assert jnp.all(jnp.isfinite(leaf))


@pytest.mark.slow
class TestTraining:
    """End-to-end training convergence test."""

    def test_disk_pinn_converges(self) -> None:
        """Train on unit disk and verify L2 error < 5%."""
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)

        model = DiskPoissonPINN(width=32, depth=3, key=k1)
        interior = sample_disk_interior(500, k2)
        boundary = sample_disk_boundary(200, k3)

        trained = train_disk_pinn(
            model,
            interior,
            boundary,
            source=1.0,
            bc_weight=100.0,
            n_steps=5000,
            lr=1e-3,
        )

        metrics = evaluate_disk_pinn(trained, n_eval=50)
        # L2 error relative to analytical max (0.25)
        relative_l2 = metrics["l2_error"] / 0.25
        assert relative_l2 < 0.05, (
            f"Relative L2 error {relative_l2:.4f} exceeds 5% threshold. "
            f"Metrics: {metrics}"
        )
