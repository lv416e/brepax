"""Unit tests for the Disk primitive."""

import equinox as eqx
import jax.numpy as jnp


class TestDisk:
    """Tests for Disk SDF evaluation."""

    def test_sdf_at_center_is_negative_radius(self) -> None:
        from brepax.primitives import Disk

        disk = Disk(center=jnp.array([0.0, 0.0]), radius=jnp.array(1.0))
        result = disk.sdf(jnp.array([0.0, 0.0]))
        assert jnp.isclose(result, -1.0)

    def test_sdf_on_boundary_is_zero(self) -> None:
        from brepax.primitives import Disk

        disk = Disk(center=jnp.array([0.0, 0.0]), radius=jnp.array(1.0))
        result = disk.sdf(jnp.array([1.0, 0.0]))
        assert jnp.isclose(result, 0.0, atol=1e-7)

    def test_sdf_outside_is_positive(self) -> None:
        from brepax.primitives import Disk

        disk = Disk(center=jnp.array([0.0, 0.0]), radius=jnp.array(1.0))
        result = disk.sdf(jnp.array([2.0, 0.0]))
        assert result > 0.0

    def test_sdf_is_jit_compatible(self) -> None:
        from brepax.primitives import Disk

        disk = Disk(center=jnp.array([0.0, 0.0]), radius=jnp.array(1.0))
        # Equinox modules require eqx.filter_jit for JAX tracing
        jitted = eqx.filter_jit(disk.sdf)
        result = jitted(jnp.array([0.0, 0.0]))
        assert jnp.isclose(result, -1.0)

    def test_sdf_is_vmap_compatible(self) -> None:
        from brepax.primitives import Disk

        disk = Disk(center=jnp.array([0.0, 0.0]), radius=jnp.array(1.0))
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        # Equinox bound methods carry self as a pytree leaf
        result = eqx.filter_vmap(disk.sdf)(points)
        assert result.shape == (3,)

    def test_sdf_is_differentiable(self) -> None:
        from brepax.primitives import Disk

        disk = Disk(center=jnp.array([0.0, 0.0]), radius=jnp.array(1.0))
        grad_fn = eqx.filter_grad(lambda x: disk.sdf(x).sum())
        grad = grad_fn(jnp.array([2.0, 0.0]))
        assert grad.shape == (2,)
