"""3D gradient accuracy benchmarks for stratum-aware Boolean operations.

Extends the 2D benchmark (test_gradient_accuracy.py) to 3D with:
- Sphere-sphere STE accuracy vs analytical ground truth
- Cylinder-sphere STE accuracy vs OCCT GProp + FD ground truth
- Optimization convergence with Adam on non-degenerate objectives
- Cross-stratum stall validation (ADR-0011)

Key findings:
- STE interior gradient: < 1% (sphere-sphere), 3-9% (cyl-sphere) at res=128
- Near-tangent error increases as sigmoid kernel width > feature size
- Method B (TOI) does not address STE accuracy (stratum-internal problem)
- Optimization converges despite near-tangent gradient error
"""

import jax
import jax.numpy as jnp
import pytest

from brepax.analytical.sphere_sphere import (
    _intersection_volume,
    sphere_sphere_union_volume,
)
from brepax.boolean.stratum import (
    intersect_volume_stratum,
    subtract_volume_stratum,
    union_volume_stratum,
)
from brepax.primitives.cylinder import Cylinder
from brepax.primitives.sphere import Sphere


def _relative_error(approx, exact):
    """Relative error, safe for near-zero exact values."""
    return jnp.where(
        jnp.abs(exact) > 1e-12,
        jnp.abs(approx - exact) / jnp.abs(exact),
        jnp.abs(approx - exact),
    )


# ---------------------------------------------------------------------------
# Sphere-sphere STE gradient accuracy (analytical ground truth)
# ---------------------------------------------------------------------------

_SPHERE_C1 = jnp.array([0.0, 0.0, 0.0])
_SPHERE_R = jnp.array(1.0)

# Interior configs: well inside intersecting stratum
_INTERIOR_D_VALUES = [0.5, 1.0, 1.5]

# Near-tangent configs: approaching external tangent at d = r1 + r2 = 2.0
_NEAR_TANGENT_D_VALUES = [1.8, 1.9, 1.95]


def _ste_intersect_grad_r1(d_val, resolution):
    """STE gradient d(V_intersect)/d(r1) via stratum dispatch."""
    c2 = jnp.array([d_val, 0.0, 0.0])
    s2 = Sphere(center=c2, radius=_SPHERE_R)

    def vol_fn(radius):
        s1 = Sphere(center=_SPHERE_C1, radius=radius)
        return intersect_volume_stratum(s1, s2, resolution=resolution)

    return jax.grad(vol_fn)(_SPHERE_R)


def _analytical_intersect_grad_r1(d_val):
    """Analytical d(V_intersect)/d(r1) from closed-form formula."""
    return jax.grad(_intersection_volume, argnums=1)(
        jnp.array(d_val), _SPHERE_R, _SPHERE_R
    )


@pytest.mark.slow
@pytest.mark.parametrize("d", _INTERIOR_D_VALUES)
@pytest.mark.parametrize("resolution", [64, 128])
def test_sphere_sphere_interior_gradient(d, resolution):
    """STE gradient accuracy in intersecting stratum interior."""
    exact = _analytical_intersect_grad_r1(d)
    approx = _ste_intersect_grad_r1(d, resolution)
    rel_err = float(_relative_error(approx, exact))

    print(
        f"\n  d={d}, res={resolution}, "
        f"exact={float(exact):.6f}, approx={float(approx):.6f}, "
        f"rel_err={rel_err:.4f}"
    )

    if resolution >= 128:
        assert rel_err < 0.05, (
            f"Interior gradient error {rel_err:.4f} exceeds 5% at res={resolution}"
        )


@pytest.mark.slow
@pytest.mark.parametrize("d", _NEAR_TANGENT_D_VALUES)
@pytest.mark.parametrize("resolution", [64, 128])
def test_sphere_sphere_near_tangent_gradient(d, resolution):
    """STE gradient accuracy near external tangent (documented limitation).

    Near-tangent error is large because the sigmoid kernel width exceeds
    the intersection feature size. This test documents the behavior
    without asserting tight accuracy bounds.
    """
    exact = _analytical_intersect_grad_r1(d)
    approx = _ste_intersect_grad_r1(d, resolution)
    rel_err = float(_relative_error(approx, exact))

    print(
        f"\n  d={d}, res={resolution}, "
        f"exact={float(exact):.6f}, approx={float(approx):.6f}, "
        f"rel_err={rel_err:.4f}"
    )

    assert jnp.isfinite(approx), f"Non-finite gradient at d={d}, res={resolution}"
    # Sign must be correct (positive for d(V_int)/d(r))
    assert float(approx) >= 0, f"Wrong sign: approx={float(approx):.6f} at d={d}"


@pytest.mark.slow
def test_sphere_sphere_resolution_convergence():
    """Verify STE gradient error decreases with resolution."""
    d = 1.0
    exact = _analytical_intersect_grad_r1(d)

    errors = {}
    for res in [32, 64, 128]:
        approx = _ste_intersect_grad_r1(d, res)
        errors[res] = float(_relative_error(approx, exact))

    print(
        f"\n  Convergence: res=32 {errors[32]:.4f}, "
        f"res=64 {errors[64]:.4f}, res=128 {errors[128]:.4f}"
    )

    assert errors[128] < errors[32], (
        f"No convergence: err@32={errors[32]:.4f}, err@128={errors[128]:.4f}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("op", ["union", "intersect", "subtract"])
def test_sphere_sphere_all_operations(op):
    """STE gradient accuracy across all Boolean operations at interior."""
    d = 1.0
    c2 = jnp.array([d, 0.0, 0.0])
    s2 = Sphere(center=c2, radius=_SPHERE_R)

    op_fn = {
        "union": union_volume_stratum,
        "intersect": intersect_volume_stratum,
        "subtract": subtract_volume_stratum,
    }

    def vol_fn(radius):
        s1 = Sphere(center=_SPHERE_C1, radius=radius)
        return op_fn[op](s1, s2, resolution=128)

    approx = float(jax.grad(vol_fn)(_SPHERE_R))

    # Analytical ground truth
    grad_int = float(
        jax.grad(_intersection_volume, argnums=1)(jnp.array(d), _SPHERE_R, _SPHERE_R)
    )
    grad_vol_a = float(4.0 * jnp.pi * _SPHERE_R**2)  # d(4/3*pi*r^3)/dr
    exact_map = {
        "union": grad_vol_a - grad_int,  # d(V_a + V_b - V_int)/d(r1), d(V_b)/d(r1)=0
        "intersect": grad_int,
        "subtract": grad_vol_a - grad_int,
    }
    exact = exact_map[op]
    rel_err = abs(approx - exact) / max(abs(exact), 1e-12)

    print(f"\n  op={op}, exact={exact:.6f}, approx={approx:.6f}, rel_err={rel_err:.4f}")

    assert rel_err < 0.05, (
        f"{op} gradient error {rel_err:.4f} exceeds 5% at d=1.0, res=128"
    )


# ---------------------------------------------------------------------------
# Cylinder-sphere STE gradient accuracy (OCCT GProp + FD ground truth)
# ---------------------------------------------------------------------------

_CYL_BREPAX = Cylinder(
    point=jnp.array([0.8, 0.0, 0.0]),
    axis=jnp.array([0.0, 0.0, 1.0]),
    radius=jnp.array(0.5),
)

_CYL_R_VALUES = [0.8, 1.0, 1.2, 1.5]


def _occt_intersect_volume(r_sphere):
    """OCCT GProp intersection volume (non-differentiable ground truth)."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCP.GProp import GProp_GProps

    sphere = BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, 0), r_sphere).Shape()
    ax = gp_Ax2(gp_Pnt(0.8, 0, -2), gp_Dir(0, 0, 1))
    cyl = BRepPrimAPI_MakeCylinder(ax, 0.5, 4.0).Shape()
    common = BRepAlgoAPI_Common(sphere, cyl)
    if not common.IsDone():
        return 0.0
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(common.Shape(), props)
    return props.Mass()


def _fd_gradient(r_sphere, eps=1e-4):
    """FD gradient of intersection volume w.r.t. sphere radius."""
    vp = _occt_intersect_volume(r_sphere + eps)
    vm = _occt_intersect_volume(r_sphere - eps)
    return (vp - vm) / (2 * eps)


@pytest.mark.slow
@pytest.mark.parametrize("r_sphere", _CYL_R_VALUES)
def test_cylinder_sphere_gradient(r_sphere):
    """STE gradient accuracy for cylinder-sphere intersection."""
    r = jnp.array(r_sphere)

    def vol_fn(radius):
        s = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=radius)
        return intersect_volume_stratum(s, _CYL_BREPAX, resolution=128)

    approx = float(jax.grad(vol_fn)(r))
    exact = _fd_gradient(r_sphere)
    rel_err = abs(approx - exact) / max(abs(exact), 1e-12)

    print(
        f"\n  r_sphere={r_sphere}, exact(FD)={exact:.6f}, "
        f"approx(STE)={approx:.6f}, rel_err={rel_err:.4f}"
    )

    assert rel_err < 0.15, (
        f"Cylinder-sphere gradient error {rel_err:.4f} exceeds 15% "
        f"at r={r_sphere}, res=128"
    )


# ---------------------------------------------------------------------------
# Optimization convergence
# ---------------------------------------------------------------------------


def _adam_step(g, m, v, step, lr, b1=0.9, b2=0.999, eps=1e-8):
    """Single Adam update step."""
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g**2
    m_hat = m / (1 - b1 ** (step + 1))
    v_hat = v / (1 - b2 ** (step + 1))
    update = lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return m, v, update


@pytest.mark.slow
def test_optimization_sphere_sphere():
    """Maximize V_intersect from near-tangent d=1.9 using Adam.

    Validates that near-tangent gradient error does not prevent
    convergence to the correct optimum (d=0, V=V_sphere).
    """
    s1 = Sphere(center=_SPHERE_C1, radius=_SPHERE_R)
    d = jnp.array(1.9)
    m_state = jnp.array(0.0)
    v_state = jnp.array(0.0)

    def neg_obj(d_val):
        s2 = Sphere(center=jnp.array([d_val, 0.0, 0.0]), radius=_SPHERE_R)
        return -intersect_volume_stratum(s1, s2, resolution=64)

    for step in range(150):
        lr = 0.05 * (0.5 * (1 + jnp.cos(jnp.pi * step / 150)))
        lr = float(jnp.maximum(lr, 0.001))

        g = jax.grad(neg_obj)(d)
        m_state, v_state, update = _adam_step(g, m_state, v_state, step, lr)
        d = d - update

    v_target = float(_intersection_volume(jnp.array(0.0), _SPHERE_R, _SPHERE_R))
    v_final = float(-neg_obj(d))

    print(
        f"\n  Final d={float(d):.6f}, V={v_final:.4f}, "
        f"V_target={v_target:.4f}, ratio={v_final / v_target:.4f}"
    )

    assert abs(float(d)) < 0.1, f"Did not converge to d=0: d={float(d):.4f}"
    assert v_final / v_target > 0.95, (
        f"Achieved only {v_final / v_target:.2%} of target volume"
    )


@pytest.mark.slow
def test_optimization_sphere_target_volume():
    """Minimize (V_union - 6.0)^2 by adjusting r1.

    Non-degenerate objective with a unique optimum in the intersecting
    stratum. Validates practical gradient-based optimization.
    """
    c2 = jnp.array([1.0, 0.0, 0.0])
    s2 = Sphere(center=c2, radius=_SPHERE_R)
    v_target = 6.0

    # Find analytical optimum via grid search
    best_r1 = 0.5
    best_err = 999.0
    for r_test in jnp.linspace(0.3, 2.5, 500):
        v = float(sphere_sphere_union_volume(_SPHERE_C1, r_test, c2, _SPHERE_R))
        err = abs(v - v_target)
        if err < best_err:
            best_err = err
            best_r1 = float(r_test)

    # Optimize with Adam
    r1 = jnp.array(0.5)
    m_state = jnp.array(0.0)
    v_state = jnp.array(0.0)

    def obj(radius):
        s1 = Sphere(center=_SPHERE_C1, radius=radius)
        return (union_volume_stratum(s1, s2, resolution=64) - v_target) ** 2

    for step in range(150):
        lr = 0.02 * (0.5 * (1 + jnp.cos(jnp.pi * step / 150)))
        lr = float(jnp.maximum(lr, 0.0005))

        g = jax.grad(obj)(r1)
        m_state, v_state, update = _adam_step(g, m_state, v_state, step, lr)
        r1 = jnp.maximum(r1 - update, 0.01)

    v_final = float(sphere_sphere_union_volume(_SPHERE_C1, r1, c2, _SPHERE_R))
    r1_err = abs(float(r1) - best_r1) / best_r1

    print(
        f"\n  Final r1={float(r1):.4f} (opt={best_r1:.4f}), "
        f"V={v_final:.4f} (target={v_target}), r1_err={r1_err:.4f}"
    )

    assert r1_err < 0.05, (
        f"r1 error {r1_err:.4f} exceeds 5%: got {float(r1):.4f}, "
        f"expected ~{best_r1:.4f}"
    )


@pytest.mark.slow
def test_optimization_cylinder_sphere():
    """Minimize (V_intersect - 1.5)^2 for cylinder-sphere by adjusting r_sphere.

    Validates optimization convergence on a non-sphere primitive pair.
    """
    v_target = 1.5

    # Find OCCT optimum
    best_r = 0.5
    best_err = 999.0
    for r_test in [x * 0.01 for x in range(50, 250)]:
        v = _occt_intersect_volume(r_test)
        err = abs(v - v_target)
        if err < best_err:
            best_err = err
            best_r = r_test

    # Optimize with Adam
    r = jnp.array(0.6)
    m_state = jnp.array(0.0)
    v_state = jnp.array(0.0)

    def obj(radius):
        s = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=radius)
        return (intersect_volume_stratum(s, _CYL_BREPAX, resolution=64) - v_target) ** 2

    for step in range(150):
        lr = 0.02 * (0.5 * (1 + jnp.cos(jnp.pi * step / 150)))
        lr = float(jnp.maximum(lr, 0.0005))

        g = jax.grad(obj)(r)
        m_state, v_state, update = _adam_step(g, m_state, v_state, step, lr)
        r = jnp.maximum(r - update, 0.01)

    v_final = _occt_intersect_volume(float(r))
    r_err = abs(float(r) - best_r) / best_r

    print(
        f"\n  Final r={float(r):.4f} (opt={best_r:.4f}), "
        f"V(OCCT)={v_final:.4f} (target={v_target}), r_err={r_err:.4f}"
    )

    assert r_err < 0.05, (
        f"r error {r_err:.4f} exceeds 5%: got {float(r):.4f}, expected ~{best_r:.4f}"
    )


# ---------------------------------------------------------------------------
# Cross-stratum validation (ADR-0011)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cross_stratum_disjoint_stall():
    """Disjoint stratum gives zero intersect gradient (ADR-0011).

    Method C correctly returns zero gradient for stratum-invariant
    directions. Crossing from disjoint to intersecting requires
    Method A (smoothing) for a non-zero exploration signal.
    """
    s1 = Sphere(center=_SPHERE_C1, radius=_SPHERE_R)

    def vol_fn(d_val):
        s2_ = Sphere(center=jnp.array([d_val, 0.0, 0.0]), radius=_SPHERE_R)
        return intersect_volume_stratum(s1, s2_, resolution=64)

    g = jax.grad(vol_fn)(jnp.array(2.5))

    print(f"\n  Disjoint intersect gradient: {float(g):.10f}")

    assert float(jnp.abs(g)) < 1e-6, (
        f"Expected zero gradient in disjoint stratum, got {float(g):.6f}"
    )
