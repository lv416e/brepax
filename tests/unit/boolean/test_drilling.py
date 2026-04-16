"""Cylinder + Plane drilling demonstration.

Phase 1 midpoint milestone: differentiable Boolean subtraction of a
cylinder through a half-space, representing a drilling operation.
Design variable is the cylinder radius; objective is the resulting volume.
"""

import jax
import jax.numpy as jnp

from brepax.boolean import subtract_volume
from brepax.primitives import Cylinder, Sphere


class TestDrillingDemo:
    """Cylinder + Plane drilling: subtract cylinder from half-space."""

    def _make_block_and_hole(
        self,
        hole_radius: float = 0.5,
    ) -> tuple[Sphere, Cylinder]:
        """Create a sphere (block proxy) with a cylinder hole through it.

        Uses a sphere as a finite volume proxy for the block.
        The cylinder passes through the sphere along the z-axis.
        """
        # Sphere as a finite "block" (radius 2)
        block = Sphere(
            center=jnp.array([0.0, 0.0, 0.0]),
            radius=jnp.array(2.0),
        )
        # Cylinder hole along z-axis
        hole = Cylinder(
            point=jnp.array([0.0, 0.0, 0.0]),
            axis=jnp.array([0.0, 0.0, 1.0]),
            radius=jnp.array(hole_radius),
        )
        return block, hole

    def test_subtract_reduces_volume(self) -> None:
        """Subtracting a cylinder from a sphere reduces volume."""
        block, hole = self._make_block_and_hole(0.5)

        # Volume of sphere alone
        sphere_vol = (4.0 / 3.0) * jnp.pi * 2.0**3

        # Volume after drilling
        drilled_vol = subtract_volume(block, hole, resolution=48)

        assert drilled_vol < sphere_vol
        assert drilled_vol > 0.0
        print(
            f"\n  Sphere vol: {float(sphere_vol):.2f}, "
            f"drilled vol: {float(drilled_vol):.2f}, "
            f"removed: {float(sphere_vol - drilled_vol):.2f}"
        )

    def test_larger_hole_removes_more(self) -> None:
        """Larger cylinder radius removes more volume."""
        block_s, hole_s = self._make_block_and_hole(0.3)
        block_l, hole_l = self._make_block_and_hole(0.8)

        vol_small = subtract_volume(block_s, hole_s, resolution=48)
        vol_large = subtract_volume(block_l, hole_l, resolution=48)

        assert vol_large < vol_small
        print(
            f"\n  Small hole (r=0.3): {float(vol_small):.2f}, "
            f"large hole (r=0.8): {float(vol_large):.2f}"
        )

    def test_gradient_wrt_hole_radius(self) -> None:
        """Larger hole radius means less volume after drilling."""
        block = Sphere(
            center=jnp.array([0.0, 0.0, 0.0]),
            radius=jnp.array(2.0),
        )

        def vol_fn(hole_radius):
            hole = Cylinder(
                point=jnp.array([0.0, 0.0, 0.0]),
                axis=jnp.array([0.0, 0.0, 1.0]),
                radius=hole_radius,
            )
            return subtract_volume(block, hole, resolution=48)

        grad = jax.grad(vol_fn)(jnp.array(0.5))
        assert jnp.isfinite(grad)
        # Increasing hole radius should decrease drilled volume
        assert grad < 0.0
        print(f"\n  d(drilled_vol)/d(hole_radius) = {float(grad):.4f} (expected < 0)")

    def test_gradient_wrt_block_radius(self) -> None:
        """Larger block radius means more volume after drilling."""
        hole = Cylinder(
            point=jnp.array([0.0, 0.0, 0.0]),
            axis=jnp.array([0.0, 0.0, 1.0]),
            radius=jnp.array(0.5),
        )

        def vol_fn(block_radius):
            block = Sphere(
                center=jnp.array([0.0, 0.0, 0.0]),
                radius=block_radius,
            )
            return subtract_volume(block, hole, resolution=48)

        grad = jax.grad(vol_fn)(jnp.array(2.0))
        assert jnp.isfinite(grad)
        # Increasing block radius should increase drilled volume
        assert grad > 0.0
        print(f"\n  d(drilled_vol)/d(block_radius) = {float(grad):.4f} (expected > 0)")

    def test_loss_gradient_direction(self) -> None:
        """Loss gradient points in the correct direction for optimization."""
        block = Sphere(
            center=jnp.array([0.0, 0.0, 0.0]),
            radius=jnp.array(2.0),
        )
        sphere_vol = (4.0 / 3.0) * jnp.pi * 2.0**3
        target_vol = 0.8 * sphere_vol

        def loss(hole_radius):
            hole = Cylinder(
                point=jnp.array([0.0, 0.0, 0.0]),
                axis=jnp.array([0.0, 0.0, 1.0]),
                radius=hole_radius,
            )
            vol = subtract_volume(block, hole, resolution=48)
            return (vol - target_vol) ** 2

        grad = jax.grad(loss)(jnp.array(0.3))
        assert jnp.isfinite(grad)
        # Current vol (~32.5) > target (26.8), so increasing radius
        # decreases volume toward target, meaning d(loss)/d(r) < 0
        assert grad < 0.0, f"Expected negative loss gradient, got {float(grad):.4f}"
        print(f"\n  d(loss)/d(hole_radius) = {float(grad):.4f} (expected < 0, correct)")
