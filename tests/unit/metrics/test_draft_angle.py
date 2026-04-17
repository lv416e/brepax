"""Unit tests for draft angle violation metric."""

import jax
import jax.numpy as jnp

from brepax.metrics.draft_angle import draft_angle_violation
from brepax.primitives import Box, Sphere


class TestDraftAngleViolation:
    """Tests for the draft_angle_violation function."""

    def test_box_side_walls_violate(self) -> None:
        """Box side walls (0 draft) violate any positive min_angle."""
        box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        d = jnp.array([0.0, 0.0, 1.0])
        violation = draft_angle_violation(
            box.sdf, d, jnp.radians(5.0), lo=lo, hi=hi, resolution=64
        )
        # 4 side walls each 2x2 = total area 16; top+bottom have 90 draft
        total_area = 24.0
        side_area = 16.0
        assert float(violation) > side_area * 0.5, (
            f"violation={float(violation):.2f}, expected >{side_area * 0.5:.1f}"
        )
        assert float(violation) < total_area, (
            f"violation={float(violation):.2f} should be less than total {total_area}"
        )

    def test_box_top_bottom_no_violation(self) -> None:
        """Box top/bottom faces (90 draft) should not violate."""
        box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        d = jnp.array([0.0, 0.0, 1.0])
        # min_angle=80: only surfaces with < 80 draft violate
        violation = draft_angle_violation(
            box.sdf, d, jnp.radians(80.0), lo=lo, hi=hi, resolution=64
        )
        total_area = 24.0
        # All 4 side walls (0 draft) + edges near top/bottom violate
        # but top/bottom faces themselves (90 draft) should not
        assert float(violation) < total_area, (
            f"violation={float(violation):.2f} should be less than total"
        )

    def test_zero_min_angle_no_violation(self) -> None:
        """Zero min_angle produces near-zero violation."""
        box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        d = jnp.array([0.0, 0.0, 1.0])
        violation = draft_angle_violation(box.sdf, d, 0.0, lo=lo, hi=hi, resolution=64)
        total_area = 24.0
        assert float(violation) < total_area * 0.3, (
            f"violation={float(violation):.2f} should be small for zero min_angle"
        )

    def test_increasing_min_angle_increases_violation(self) -> None:
        """Higher min_angle threshold produces more violation."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.5))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        d = jnp.array([0.0, 0.0, 1.0])
        v_small = draft_angle_violation(
            sphere.sdf, d, jnp.radians(10.0), lo=lo, hi=hi, resolution=48
        )
        v_large = draft_angle_violation(
            sphere.sdf, d, jnp.radians(45.0), lo=lo, hi=hi, resolution=48
        )
        assert float(v_large) > float(v_small), (
            f"45 violation ({float(v_large):.2f}) should exceed "
            f"10 violation ({float(v_small):.2f})"
        )

    def test_differentiable_wrt_min_angle(self) -> None:
        """jax.grad w.r.t. min_angle is non-negative."""
        box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        d = jnp.array([0.0, 0.0, 1.0])

        def loss(angle: jnp.ndarray) -> jnp.ndarray:
            return draft_angle_violation(box.sdf, d, angle, lo=lo, hi=hi, resolution=48)

        grad_angle = jax.grad(loss)(jnp.radians(5.0))
        assert jnp.isfinite(grad_angle), f"Non-finite gradient: {grad_angle}"
        assert float(grad_angle) >= 0.0, (
            f"Expected non-negative gradient, got {float(grad_angle):.4f}"
        )

    def test_differentiable_wrt_direction(self) -> None:
        """jax.grad w.r.t. mold direction is finite."""
        box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)

        def loss(d: jnp.ndarray) -> jnp.ndarray:
            return draft_angle_violation(
                box.sdf, d, jnp.radians(5.0), lo=lo, hi=hi, resolution=48
            )

        grad_d = jax.grad(loss)(jnp.array([0.0, 0.0, 1.0]))
        assert jnp.all(jnp.isfinite(grad_d)), f"Non-finite gradient: {grad_d}"

    def test_differentiable_wrt_shape(self) -> None:
        """jax.grad w.r.t. shape parameters works."""
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)
        d = jnp.array([0.0, 0.0, 1.0])

        def loss(he: jnp.ndarray) -> jnp.ndarray:
            box = Box(center=jnp.zeros(3), half_extents=he)
            return draft_angle_violation(
                box.sdf, d, jnp.radians(5.0), lo=lo, hi=hi, resolution=48
            )

        grad_he = jax.grad(loss)(jnp.array([2.0, 1.5, 1.0]))
        assert jnp.all(jnp.isfinite(grad_he)), f"Non-finite gradient: {grad_he}"

    def test_jit_compatible(self) -> None:
        """draft_angle_violation works under jax.jit."""
        box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)

        @jax.jit
        def compute(d: jnp.ndarray) -> jnp.ndarray:
            return draft_angle_violation(
                box.sdf, d, jnp.radians(5.0), lo=lo, hi=hi, resolution=32
            )

        result = compute(jnp.array([0.0, 0.0, 1.0]))
        assert jnp.isfinite(result)
        assert float(result) > 0.0

    def test_sphere_equator_violates(self) -> None:
        """Sphere equator (0 draft) violates; poles (90 draft) do not."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.5))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        d = jnp.array([0.0, 0.0, 1.0])
        violation = draft_angle_violation(
            sphere.sdf, d, jnp.radians(30.0), lo=lo, hi=hi, resolution=64
        )
        total_area = 4.0 * jnp.pi * 1.5**2
        # Equatorial band with draft < 30 should be a fraction of total
        assert float(violation) > 0.0
        assert float(violation) < float(total_area), (
            f"violation={float(violation):.2f} should be partial, not total"
        )
