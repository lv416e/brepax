"""Trim-aware signed blend SDF composition (ADR-0018)."""

from __future__ import annotations

import itertools
import math

import jax
import jax.numpy as jnp
import pytest

from brepax.brep.trim_sdf import trim_aware_sdf


def _square_trim_on_plane() -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Plane z=0 trimmed to the unit square ``[-0.5, 0.5]^2``.

    Returns ``(polygon_uv, polygon_mask, polyline_3d, polyline_mask)``
    matching the signature of ``trim_aware_sdf``.  UV here coincides
    with ``(x, y)`` because the plane's canonical frame is the
    coordinate frame.
    """
    polygon_uv = jnp.array(
        [
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [-0.5, 0.5],
        ]
    )
    mask = jnp.ones(4)
    polyline_3d = jnp.concatenate([polygon_uv, jnp.zeros((4, 1))], axis=-1)
    return polygon_uv, mask, polyline_3d, mask


def _evaluate_at(query: jnp.ndarray) -> jnp.ndarray:
    """Trim-aware SDF value at a query point for the square-trimmed plane."""
    polygon_uv, poly_mask, polyline_3d, line_mask = _square_trim_on_plane()
    d_s = query[2]  # plane z=0, normal +z
    foot_uv = query[:2]  # orthogonal projection is (x, y, 0)
    return trim_aware_sdf(
        query,
        d_s,
        foot_uv,
        polygon_uv,
        poly_mask,
        polyline_3d,
        line_mask,
    )


class TestFourRegimes:
    """ADR-0018 regime table: in/out trim x in/out half-space."""

    def test_inside_trim_below_plane_is_negative(self) -> None:
        d = _evaluate_at(jnp.array([0.0, 0.0, -0.5]))
        # chi_T ~ 1, so d_T ~ d_S = -0.5
        assert jnp.isclose(d, -0.5, atol=1e-3)

    def test_inside_trim_above_plane_is_positive(self) -> None:
        d = _evaluate_at(jnp.array([0.0, 0.0, 0.5]))
        # chi_T ~ 1, so d_T ~ d_S = +0.5
        assert jnp.isclose(d, 0.5, atol=1e-3)

    def test_outside_trim_below_plane_is_positive(self) -> None:
        # Key phantom-elimination case: d_S = -0.5 (below plane) but
        # chi_T ~ 0 (outside trim) -> d_T should be +d_partial, not -0.5.
        d = _evaluate_at(jnp.array([1.0, 1.0, -0.5]))
        assert float(d) > 0.0, (
            f"phantom not eliminated: d_T={float(d)} but query is outside trim"
        )
        # Nearest loop vertex is (0.5, 0.5, 0); distance = sqrt(0.5 + 0.25) ~= 0.866.
        assert jnp.isclose(d, math.sqrt(0.75), atol=1e-3)

    def test_outside_trim_above_plane_is_positive(self) -> None:
        d = _evaluate_at(jnp.array([1.0, 1.0, 0.5]))
        assert jnp.isclose(d, math.sqrt(0.75), atol=1e-3)


class TestContinuityAcrossTrim:
    """The signed blend must remain continuous as chi_T transitions."""

    def test_transition_across_trim_boundary(self) -> None:
        # Sweep x from well-inside to well-outside along y=0, z=-0.1.
        xs = [0.0, 0.4, 0.5, 0.6, 1.0]
        values = [float(_evaluate_at(jnp.array([x, 0.0, -0.1]))) for x in xs]
        # Monotone increase: deep inside ~= d_s = -0.1, deep outside = d_partial > 0.
        for a, b in itertools.pairwise(values):
            assert b > a - 1e-6, f"non-monotone at transition: {values}"
        # Endpoints land in the expected regimes.
        assert values[0] < 0.0  # deep inside trim
        assert values[-1] > 0.0  # deep outside trim

    def test_on_trim_boundary_midpoint(self) -> None:
        # On the trim boundary in 3D: chi_T ~ 0.5, d_partial ~ 0.
        # d_S = -0.1 here; d_T should land between d_S and 0.
        d = _evaluate_at(jnp.array([0.5, 0.0, -0.1]))
        assert -0.1 < float(d) < 0.1


class TestGradient:
    """jax.grad through each input must stay finite."""

    def test_grad_through_query_finite(self) -> None:
        def loss(p: jnp.ndarray) -> jnp.ndarray:
            return _evaluate_at(p)

        g = jax.grad(loss)(jnp.array([0.3, 0.3, -0.2]))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_query_finite_outside_trim(self) -> None:
        # Phantom-regime query: grad must still be finite.
        def loss(p: jnp.ndarray) -> jnp.ndarray:
            return _evaluate_at(p)

        g = jax.grad(loss)(jnp.array([1.0, 1.0, -0.5]))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_polygon_finite(self) -> None:
        polygon_uv, poly_mask, polyline_3d, line_mask = _square_trim_on_plane()
        query = jnp.array([0.3, 0.3, -0.2])

        def loss(pg: jnp.ndarray) -> jnp.ndarray:
            return trim_aware_sdf(
                query,
                query[2],
                query[:2],
                pg,
                poly_mask,
                polyline_3d,
                line_mask,
            )

        g = jax.grad(loss)(polygon_uv)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_polyline_finite(self) -> None:
        polygon_uv, poly_mask, polyline_3d, line_mask = _square_trim_on_plane()
        query = jnp.array([1.0, 1.0, -0.5])

        def loss(pl: jnp.ndarray) -> jnp.ndarray:
            return trim_aware_sdf(
                query,
                query[2],
                query[:2],
                polygon_uv,
                poly_mask,
                pl,
                line_mask,
            )

        g = jax.grad(loss)(polyline_3d)
        assert jnp.all(jnp.isfinite(g))


class TestSharpness:
    """Higher sharpness narrows the trim transition zone."""

    @pytest.mark.parametrize("sharp", [50.0, 200.0, 1000.0])
    def test_far_inside_trim_converges_to_d_s(self, sharp: float) -> None:
        # Well inside the trim square, d_T must converge to d_s regardless
        # of sharpness.
        polygon_uv, poly_mask, polyline_3d, line_mask = _square_trim_on_plane()
        query = jnp.array([0.0, 0.0, -0.3])
        d = trim_aware_sdf(
            query,
            query[2],
            query[:2],
            polygon_uv,
            poly_mask,
            polyline_3d,
            line_mask,
            sharpness=sharp,
        )
        assert jnp.isclose(d, -0.3, atol=1e-2)

    @pytest.mark.parametrize("sharp", [50.0, 200.0, 1000.0])
    def test_far_outside_trim_converges_to_d_partial(self, sharp: float) -> None:
        polygon_uv, poly_mask, polyline_3d, line_mask = _square_trim_on_plane()
        query = jnp.array([2.0, 2.0, -0.3])
        d = trim_aware_sdf(
            query,
            query[2],
            query[:2],
            polygon_uv,
            poly_mask,
            polyline_3d,
            line_mask,
            sharpness=sharp,
        )
        # Nearest vertex is (0.5, 0.5, 0); distance = sqrt(1.5^2 + 1.5^2 + 0.3^2).
        expected = math.sqrt(1.5**2 + 1.5**2 + 0.3**2)
        assert jnp.isclose(d, expected, atol=1e-2)
