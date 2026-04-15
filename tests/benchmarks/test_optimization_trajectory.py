"""Gate criterion 2: optimization trajectory analysis.

Tests optimization convergence for both within-stratum and
cross-stratum scenarios, revealing the complementary strengths
of Method (A) smoothing and Method (C) stratum-aware gradients.

Key finding: Method (C) provides exact gradients within each stratum
but zero gradient signal for design variables that are stratum-invariant
(e.g., center positions in the disjoint stratum). Method (A) provides
biased but globally non-zero gradient signal via its smoothing kernel.
"""

import jax
import jax.numpy as jnp

from brepax.analytical.disk_disk import disk_disk_union_area
from brepax.boolean import union_area
from brepax.primitives import Disk

# Fixed parameters
C1 = jnp.array([0.0, 0.0])
R1 = jnp.array(1.0)
R2 = jnp.array(1.0)
C2_Y = 0.0


def _optimize(loss_fn, init_x, *, lr=0.05, max_steps=300):
    """Gradient descent with trajectory recording."""
    x = init_x
    trajectory = [float(x)]
    losses = [float(loss_fn(x))]
    grad_fn = jax.grad(loss_fn)

    for _ in range(max_steps):
        g = grad_fn(x)
        x = x - lr * g
        trajectory.append(float(x))
        loss = float(loss_fn(x))
        losses.append(loss)
        if loss < 1e-8:
            break

    return x, trajectory, losses


# ---- Scenario 1: within-stratum optimization (Method C excels) ----


class TestWithinStratumOptimization:
    """Optimization within the intersecting stratum where Method (C) is exact."""

    def test_method_c_converges_within_stratum(self) -> None:
        """Starting inside intersecting stratum, optimize r1 toward target area."""
        target_area = float(
            disk_disk_union_area(
                C1,
                jnp.array(1.5),
                jnp.array([1.0, 0.0]),
                R2,
            )
        )

        def loss_c(r1):
            a = Disk(center=C1, radius=r1)
            b = Disk(center=jnp.array([1.0, 0.0]), radius=R2)
            area = union_area(a, b, method="stratum")
            return (area - target_area) ** 2

        def loss_a(r1):
            a = Disk(center=C1, radius=r1)
            b = Disk(center=jnp.array([1.0, 0.0]), radius=R2)
            area = union_area(a, b, method="smoothing", k=0.1, beta=0.1, resolution=128)
            return (area - target_area) ** 2

        final_c, traj_c, losses_c = _optimize(
            loss_c, jnp.array(0.8), lr=0.01, max_steps=200
        )
        final_a, traj_a, losses_a = _optimize(
            loss_a, jnp.array(0.8), lr=0.01, max_steps=200
        )

        print("\n  === Within-stratum (intersecting), optimize r1 ===")
        print(f"  Target area: {target_area:.6f}")
        print(
            f"  Method (C): final_r1={float(final_c):.6f}, "
            f"final_loss={losses_c[-1]:.2e}, steps={len(traj_c) - 1}"
        )
        print(
            f"  Method (A): final_r1={float(final_a):.6f}, "
            f"final_loss={losses_a[-1]:.2e}, steps={len(traj_a) - 1}"
        )

        # Both methods converge within stratum; verify Method (C) reaches target
        assert losses_c[-1] < 1e-6, (
            f"Method (C) did not converge: loss={losses_c[-1]:.2e}"
        )
        assert losses_a[-1] < 1e-6, (
            f"Method (A) did not converge: loss={losses_a[-1]:.2e}"
        )


# ---- Scenario 2: cross-stratum optimization (reveals fundamental tradeoff) ----


class TestCrossStratumOptimization:
    """Cross-stratum optimization revealing the smoothing vs exact gradient tradeoff."""

    def test_cross_stratum_gradient_analysis(self) -> None:
        """Zero-gradient phenomenon in stratum-aware cross-boundary optimization.

        In the disjoint stratum, area = pi*r1^2 + pi*r2^2 is independent
        of center positions. Method (C)'s exact gradient correctly returns
        zero for d(area)/d(c2_x), meaning gradient descent cannot cross
        the stratum boundary by moving centers.

        Method (A) smoothing provides a weak but non-zero gradient signal
        because the smooth-min kernel extends beyond the exact boundary.
        """
        target_area = float(
            disk_disk_union_area(
                C1,
                R1,
                jnp.array([1.5, C2_Y]),
                R2,
            )
        )

        def loss_c(c2_x):
            c2 = jnp.array([c2_x, C2_Y])
            a = Disk(center=C1, radius=R1)
            b = Disk(center=c2, radius=R2)
            return (union_area(a, b, method="stratum") - target_area) ** 2

        def loss_a(c2_x):
            c2 = jnp.array([c2_x, C2_Y])
            a = Disk(center=C1, radius=R1)
            b = Disk(center=c2, radius=R2)
            return (
                union_area(a, b, method="smoothing", k=0.1, beta=0.1, resolution=128)
                - target_area
            ) ** 2

        init_x = jnp.array(3.0)  # disjoint stratum

        # Method (C): gradient is zero in disjoint stratum for center vars
        grad_c = jax.grad(loss_c)(init_x)
        # Method (A): smoothing provides weak gradient signal
        grad_a = jax.grad(loss_a)(init_x)

        print("\n  === Cross-stratum gradient at c2_x=3.0 (disjoint) ===")
        print(f"  Method (C) gradient: {float(grad_c):.8f} (expected: ~0)")
        print(f"  Method (A) gradient: {float(grad_a):.8f} (expected: non-zero)")

        # Method (C) gradient is exactly zero (area independent of c2_x in disjoint)
        assert jnp.isclose(grad_c, 0.0, atol=1e-10), (
            f"Expected zero gradient in disjoint stratum, got {float(grad_c)}"
        )
        # Method (A) gradient is non-zero due to smoothing
        assert jnp.abs(grad_a) > 1e-6, (
            f"Expected non-zero smoothing gradient, got {float(grad_a)}"
        )

        # Run optimization with both methods
        final_c, _traj_c, _losses_c = _optimize(loss_c, init_x, lr=0.1, max_steps=100)
        final_a, _traj_a, _losses_a = _optimize(loss_a, init_x, lr=0.1, max_steps=100)

        print("\n  === Cross-stratum optimization (disjoint → intersecting) ===")
        print(
            f"  Method (C): final_x={float(final_c):.4f}, "
            f"moved={abs(float(final_c) - 3.0):.4f}"
        )
        print(
            f"  Method (A): final_x={float(final_a):.4f}, "
            f"moved={abs(float(final_a) - 3.0):.4f}"
        )
        print("  Method (A) moved further due to smoothing gradient signal")

        # Verify Method (C) is stuck (zero gradient → no movement)
        assert jnp.isclose(final_c, init_x, atol=1e-6)
        # Verify Method (A) moved (smoothing gradient provides weak signal)
        assert jnp.abs(final_a - init_x) > 1e-4
