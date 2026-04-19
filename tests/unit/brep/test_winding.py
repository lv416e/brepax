"""Tests for generalized winding number computation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from brepax.brep.winding import triangle_solid_angle, winding_number


class TestTriangleSolidAngle:
    """Tests for Van Oosterom & Strackee solid angle formula."""

    def test_unit_triangle_from_origin(self) -> None:
        """Solid angle of a unit triangle at the origin is positive."""
        v0 = jnp.array([1.0, 0.0, 0.0])
        v1 = jnp.array([0.0, 1.0, 0.0])
        v2 = jnp.array([0.0, 0.0, 1.0])
        omega = triangle_solid_angle(jnp.zeros(3), v0, v1, v2)
        assert float(omega) > 0.0

    def test_solid_angle_sign_flips_with_winding(self) -> None:
        """Swapping two vertices negates the solid angle."""
        p = jnp.zeros(3)
        v0 = jnp.array([1.0, 0.0, 0.0])
        v1 = jnp.array([0.0, 1.0, 0.0])
        v2 = jnp.array([0.0, 0.0, 1.0])
        omega_fwd = triangle_solid_angle(p, v0, v1, v2)
        omega_rev = triangle_solid_angle(p, v0, v2, v1)
        assert float(omega_fwd) == pytest.approx(-float(omega_rev), abs=1e-6)

    def test_far_point_small_angle(self) -> None:
        """Point far from a triangle sees a small solid angle."""
        far = jnp.array([100.0, 100.0, 100.0])
        v0 = jnp.array([1.0, 0.0, 0.0])
        v1 = jnp.array([0.0, 1.0, 0.0])
        v2 = jnp.array([0.0, 0.0, 1.0])
        omega = triangle_solid_angle(far, v0, v1, v2)
        assert abs(float(omega)) < 0.001


class TestWindingNumber:
    """Tests for generalized winding number."""

    @staticmethod
    def _unit_cube_triangles() -> jnp.ndarray:
        """12 triangles forming a unit cube [0,1]^3."""
        verts = jnp.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=jnp.float32,
        )
        # 6 faces x 2 triangles, outward normals
        faces = jnp.array(
            [
                [0, 2, 1],
                [0, 3, 2],  # bottom (-z)
                [4, 5, 6],
                [4, 6, 7],  # top (+z)
                [0, 1, 5],
                [0, 5, 4],  # front (-y)
                [2, 3, 7],
                [2, 7, 6],  # back (+y)
                [0, 4, 7],
                [0, 7, 3],  # left (-x)
                [1, 2, 6],
                [1, 6, 5],  # right (+x)
            ]
        )
        return verts[faces]

    def test_inside_point_winding_one(self) -> None:
        """Point inside a closed cube has winding number ~1."""
        tris = self._unit_cube_triangles()
        inside = jnp.array([0.5, 0.5, 0.5])
        w = winding_number(inside, tris)
        assert float(w) == pytest.approx(1.0, abs=0.01)

    def test_outside_point_winding_zero(self) -> None:
        """Point outside a closed cube has winding number ~0."""
        tris = self._unit_cube_triangles()
        outside = jnp.array([5.0, 5.0, 5.0])
        w = winding_number(outside, tris)
        assert float(w) == pytest.approx(0.0, abs=0.01)

    def test_differentiable(self) -> None:
        """Winding number is differentiable w.r.t. query point."""
        tris = self._unit_cube_triangles()

        def wn_at(p: jnp.ndarray) -> jnp.ndarray:
            return winding_number(p, tris)

        grad = jax.grad(wn_at)(jnp.array([0.5, 0.5, 0.5]))
        assert jnp.all(jnp.isfinite(grad))
