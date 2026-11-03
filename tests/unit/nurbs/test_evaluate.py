"""Unit tests for B-spline surface evaluation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from OCP.Geom import Geom_BSplineSurface
from OCP.gp import gp_Pnt
from OCP.TColgp import TColgp_Array2OfPnt
from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal

from brepax.nurbs.evaluate import (
    bspline_basis,
    evaluate_surface,
    evaluate_surface_derivs,
)


def _build_bspline_surface(
    *,
    n_u: int,
    n_v: int,
    degree_u: int,
    degree_v: int,
    u_knots: list[float],
    u_mults: list[int],
    v_knots: list[float],
    v_mults: list[int],
    seed: int = 0,
) -> tuple[Geom_BSplineSurface, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build a clamped, non-rational BSpline surface and its flat-knot arrays.

    Returns ``(occt_surface, control_points, knots_u, knots_v)`` so a
    test can evaluate the same surface through both
    :func:`~brepax.nurbs.evaluate.evaluate_surface` and OCCT's own
    ``Value`` as an independent oracle.
    """
    rng = np.random.default_rng(seed)
    poles = TColgp_Array2OfPnt(1, n_u, 1, n_v)
    control_points = np.zeros((n_u, n_v, 3))
    for i in range(1, n_u + 1):
        for j in range(1, n_v + 1):
            x, y, z = float(i), float(j), float(rng.normal())
            poles.SetValue(i, j, gp_Pnt(x, y, z))
            control_points[i - 1, j - 1] = [x, y, z]

    u_array = TColStd_Array1OfReal(1, len(u_knots))
    u_mult_array = TColStd_Array1OfInteger(1, len(u_mults))
    for idx, (knot, mult) in enumerate(zip(u_knots, u_mults, strict=True), start=1):
        u_array.SetValue(idx, knot)
        u_mult_array.SetValue(idx, mult)
    v_array = TColStd_Array1OfReal(1, len(v_knots))
    v_mult_array = TColStd_Array1OfInteger(1, len(v_mults))
    for idx, (knot, mult) in enumerate(zip(v_knots, v_mults, strict=True), start=1):
        v_array.SetValue(idx, knot)
        v_mult_array.SetValue(idx, mult)

    surface = Geom_BSplineSurface(
        poles, u_array, v_array, u_mult_array, v_mult_array, degree_u, degree_v
    )
    knots_u = jnp.array(
        [k for k, m in zip(u_knots, u_mults, strict=True) for _ in range(m)]
    )
    knots_v = jnp.array(
        [k for k, m in zip(v_knots, v_mults, strict=True) for _ in range(m)]
    )
    return surface, jnp.array(control_points), knots_u, knots_v


class TestBsplineBasis:
    """Tests for Cox-de Boor basis function evaluation."""

    def test_partition_of_unity(self) -> None:
        """Basis functions sum to 1 at any interior parameter value."""
        knots = jnp.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
        for t_val in [0.1, 0.25, 0.5, 0.75, 0.99]:
            vals = bspline_basis(jnp.array(t_val), knots, degree=3, n_basis=5)
            assert jnp.isclose(jnp.sum(vals), 1.0, atol=1e-10), (
                f"t={t_val}: sum={float(jnp.sum(vals))}"
            )

    def test_endpoints_clamped(self) -> None:
        """First basis = 1 at t=0, last basis = 1 at t=1 for clamped knots."""
        knots = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        vals_start = bspline_basis(jnp.array(0.0), knots, degree=3, n_basis=4)
        assert jnp.isclose(vals_start[0], 1.0)
        assert jnp.allclose(vals_start[1:], 0.0, atol=1e-10)

        vals_end = bspline_basis(jnp.array(1.0), knots, degree=3, n_basis=4)
        assert jnp.isclose(vals_end[-1], 1.0)
        assert jnp.allclose(vals_end[:-1], 0.0, atol=1e-10)

    def test_degree_one_linear(self) -> None:
        """Degree 1 with uniform knots gives linear interpolation."""
        knots = jnp.array([0.0, 0.0, 0.5, 1.0, 1.0])
        vals = bspline_basis(jnp.array(0.25), knots, degree=1, n_basis=3)
        assert jnp.isclose(vals[0], 0.5, atol=1e-6)
        assert jnp.isclose(vals[1], 0.5, atol=1e-6)
        assert jnp.isclose(vals[2], 0.0, atol=1e-6)

    def test_non_negative(self) -> None:
        """Basis functions are non-negative."""
        knots = jnp.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
        for t_val in [0.0, 0.1, 0.5, 0.9, 1.0]:
            vals = bspline_basis(jnp.array(t_val), knots, degree=3, n_basis=5)
            assert jnp.all(vals >= -1e-10), f"t={t_val}: {vals}"


def _right_closed_only_basis(
    t: jnp.ndarray, knots: jnp.ndarray, degree: int, n_basis: int
) -> jnp.ndarray:
    """The basis recurrence before the left-endpoint extension.

    Kept only as a frozen reference to confirm the extension changes
    nothing inside the domain; not a second implementation to maintain.
    """
    n_spans = n_basis + degree
    left = knots[:n_spans]
    right = knots[1 : n_spans + 1]
    basis = jnp.where((left <= t) & (t < right), 1.0, 0.0)
    basis = basis.at[n_basis - 1].set(
        jnp.where(t >= knots[n_basis], 1.0, basis[n_basis - 1])
    )
    for d in range(1, degree + 1):
        size = n_spans - d
        denom_l = knots[d : d + size] - knots[:size]
        safe_l = jnp.where(denom_l > 0, denom_l, 1.0)
        alpha_l = jnp.where(denom_l > 0, (t - knots[:size]) / safe_l, 0.0)
        denom_r = knots[d + 1 : d + 1 + size] - knots[1 : 1 + size]
        safe_r = jnp.where(denom_r > 0, denom_r, 1.0)
        alpha_r = jnp.where(
            denom_r > 0, (knots[d + 1 : d + 1 + size] - t) / safe_r, 0.0
        )
        basis = alpha_l * basis[:size] + alpha_r * basis[1 : size + 1]
    return basis


class TestBsplineBasisDomainExtension:
    """Tests for the out-of-domain polynomial extension of ``bspline_basis``.

    OCCT's own mesher can place a trimmed-face mesh node at a parameter
    slightly outside the knot domain -- measured up to ~3.3e-4 on a
    real NIST part (STC-10, face 139) -- and
    ``Geom_BSplineSurface::Value`` still evaluates it there by
    continuing the nearest end span's polynomial piece, not by
    clamping or refusing. ``bspline_basis`` must match that, since a
    plain zero basis outside the domain collapsed the evaluated point
    to the origin.
    """

    @pytest.mark.parametrize(
        "knots,degree,n_basis",
        [
            (jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), 3, 4),
            (jnp.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]), 3, 5),
        ],
    )
    def test_in_domain_bit_identical_to_right_closed_only(
        self, knots: jnp.ndarray, degree: int, n_basis: int
    ) -> None:
        """Adding the left-endpoint extension changes nothing in-domain."""
        for t_val in [0.0, 0.001, 0.25, 0.5, 0.75, 0.999, 1.0]:
            t = jnp.array(t_val)
            got = bspline_basis(t, knots, degree, n_basis)
            want = _right_closed_only_basis(t, knots, degree, n_basis)
            assert jnp.array_equal(got, want), f"t={t_val}: {got} != {want}"

    def test_out_of_domain_matches_occt_value_both_ends_u_and_v(self) -> None:
        """Out-of-domain (u, v) matches ``Geom_BSplineSurface.Value`` directly.

        Degree 3 x 2 on a non-uniform, off-origin u domain, modeled on
        the NIST STC-10 face that reported this defect.
        """
        surface, control_points, knots_u, knots_v = _build_bspline_surface(
            n_u=4,
            n_v=3,
            degree_u=3,
            degree_v=2,
            u_knots=[0.00507966, 0.026184],
            u_mults=[4, 4],
            v_knots=[0.0, 1.0],
            v_mults=[3, 3],
        )
        u_lower, u_upper = 0.00507966, 0.026184
        domain_width_u = u_upper - u_lower
        u_mid, v_mid = 0.015, 0.5
        cases = [
            (u_lower - 3.3e-4, v_mid),  # u undershoot (matches face 139)
            (u_upper + 3.3e-4, v_mid),  # u overshoot
            (u_mid, -3.3e-4),  # v undershoot
            (u_mid, 1.0 + 3.3e-4),  # v overshoot
            (u_lower - 0.1 * domain_width_u, v_mid),  # larger u undershoot
            (u_mid, -0.25),  # larger v undershoot
        ]
        for u_val, v_val in cases:
            point = evaluate_surface(
                control_points,
                knots_u,
                knots_v,
                3,
                2,
                jnp.array(u_val),
                jnp.array(v_val),
            )
            oracle = surface.Value(u_val, v_val)
            occt = np.array([oracle.X(), oracle.Y(), oracle.Z()])
            np.testing.assert_allclose(np.asarray(point), occt, atol=1e-9, rtol=1e-9)

    @pytest.mark.parametrize(
        "knots,degree",
        [
            ([0.0, 1.0, 2.0, 3.0, 4.0], 1),
            ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2),
            ([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0], 3),
        ],
    )
    def test_matches_scipy_extrapolation_for_unclamped_knots(
        self, knots: list[float], degree: int
    ) -> None:
        """Outside the domain, each basis function continues its end piece.

        ``scipy.interpolate.BSpline`` with ``extrapolate=True`` evaluates
        past ``[t[k], t[n]]`` with the first or last polynomial piece; with
        unclamped knots the spans outside the domain have nonzero width,
        which is the case that must not activate a second span.
        """
        from scipy.interpolate import BSpline

        n_basis = len(knots) - degree - 1
        lo, hi = knots[degree], knots[n_basis]
        width = hi - lo
        samples = [lo - 0.4 * width, lo - 1e-3, lo, hi, hi + 1e-3, hi + 0.4 * width]
        for t in samples:
            actual = bspline_basis(
                jnp.array(t), jnp.array(knots), degree=degree, n_basis=n_basis
            )
            expected = [
                BSpline(knots, np.eye(n_basis)[i], degree, extrapolate=True)(t)
                for i in range(n_basis)
            ]
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_jit_vmap_grad_work_outside_domain(self) -> None:
        """Out-of-domain evaluation stays ``jit``-, ``vmap``-, ``grad``-compatible."""
        _surface, control_points, knots_u, knots_v = _build_bspline_surface(
            n_u=4,
            n_v=4,
            degree_u=3,
            degree_v=3,
            u_knots=[0.0, 1.0],
            u_mults=[4, 4],
            v_knots=[0.0, 1.0],
            v_mults=[4, 4],
        )

        @jax.jit
        def eval_at(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
            return evaluate_surface(control_points, knots_u, knots_v, 3, 3, u, v)

        point = eval_at(jnp.array(1.2), jnp.array(-0.1))
        assert jnp.all(jnp.isfinite(point))

        batched = jax.vmap(eval_at)(
            jnp.array([-0.2, 0.5, 1.2]), jnp.array([0.5, -0.1, 1.3])
        )
        assert batched.shape == (3, 3)
        assert jnp.all(jnp.isfinite(batched))

        grad_u = jax.grad(lambda u: jnp.sum(eval_at(u, jnp.array(0.5))))(jnp.array(1.2))
        assert jnp.all(jnp.isfinite(grad_u))


class TestEvaluateSurface:
    """Tests for B-spline surface evaluation."""

    def test_flat_patch_corners(self) -> None:
        """Bilinear patch evaluates to control points at corners."""
        pts = jnp.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            ]
        )
        knots = jnp.array([0.0, 0.0, 1.0, 1.0])
        p00 = evaluate_surface(pts, knots, knots, 1, 1, jnp.array(0.0), jnp.array(0.0))
        p11 = evaluate_surface(pts, knots, knots, 1, 1, jnp.array(1.0), jnp.array(1.0))
        assert jnp.allclose(p00, jnp.array([0.0, 0.0, 0.0]), atol=1e-10)
        assert jnp.allclose(p11, jnp.array([1.0, 1.0, 0.0]), atol=1e-10)

    def test_flat_patch_midpoint(self) -> None:
        """Bilinear patch midpoint is average of corners."""
        pts = jnp.array(
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
            ]
        )
        knots = jnp.array([0.0, 0.0, 1.0, 1.0])
        mid = evaluate_surface(pts, knots, knots, 1, 1, jnp.array(0.5), jnp.array(0.5))
        assert jnp.allclose(mid, jnp.array([1.0, 1.0, 0.0]), atol=1e-10)

    def test_curved_patch_midpoint(self) -> None:
        """Cubic patch with raised center control point lifts the surface."""
        pts = jnp.zeros((4, 4, 3))
        for i in range(4):
            for j in range(4):
                pts = pts.at[i, j, 0].set(i / 3.0)
                pts = pts.at[i, j, 1].set(j / 3.0)
        pts = pts.at[1, 1, 2].set(1.0)
        pts = pts.at[1, 2, 2].set(1.0)
        pts = pts.at[2, 1, 2].set(1.0)
        pts = pts.at[2, 2, 2].set(1.0)

        knots = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        mid = evaluate_surface(pts, knots, knots, 3, 3, jnp.array(0.5), jnp.array(0.5))
        assert float(mid[2]) > 0.3, f"Expected lifted z, got {float(mid[2]):.4f}"

    def test_jit_compatible(self) -> None:
        """Surface evaluation works under jax.jit."""
        pts = jnp.zeros((4, 4, 3))
        knots = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

        @jax.jit
        def eval_at(u, v):
            return evaluate_surface(pts, knots, knots, 3, 3, u, v)

        result = eval_at(jnp.array(0.5), jnp.array(0.5))
        assert result.shape == (3,)
        assert jnp.all(jnp.isfinite(result))


class TestEvaluateSurfaceDerivs:
    """Tests for surface derivative computation."""

    def test_flat_patch_zero_z_derivative(self) -> None:
        """Flat xy-plane patch has zero z-derivative."""
        pts = jnp.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            ]
        )
        knots = jnp.array([0.0, 0.0, 1.0, 1.0])
        _, du, dv = evaluate_surface_derivs(
            pts, knots, knots, 1, 1, jnp.array(0.5), jnp.array(0.5)
        )
        assert jnp.isclose(du[2], 0.0, atol=1e-6)
        assert jnp.isclose(dv[2], 0.0, atol=1e-6)

    def test_bilinear_patch_tangents(self) -> None:
        """Bilinear patch tangent vectors are constant."""
        pts = jnp.array(
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
            ]
        )
        knots = jnp.array([0.0, 0.0, 1.0, 1.0])
        _, du, dv = evaluate_surface_derivs(
            pts, knots, knots, 1, 1, jnp.array(0.3), jnp.array(0.7)
        )
        assert jnp.allclose(du, jnp.array([0.0, 2.0, 0.0]), atol=1e-4)
        assert jnp.allclose(dv, jnp.array([2.0, 0.0, 0.0]), atol=1e-4)
