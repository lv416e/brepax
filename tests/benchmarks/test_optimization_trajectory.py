"""Fixed-budget disk optimization checked against closed-form geometry.

Method C uses analytical gradients for disjoint disks and a grid estimator
for intersecting disks. A nonzero surrogate gradient is distinct from
movement, boundary crossing, and improvement of the exact geometric loss.
These two fixtures compare equal update counts, not equal runtime or a
general ranking of the methods.
"""

import jax
import jax.numpy as jnp

from brepax.analytical.disk_disk import disk_disk_union_area
from brepax.boolean import union_area
from brepax.primitives import Disk

C1 = jnp.array([0.0, 0.0])
R1 = jnp.array(1.0)
R2 = jnp.array(1.0)
C2_Y = 0.0


def _optimize(loss_fn, init_x, *, lr, max_steps):
    """Record every iterate and surrogate loss under a fixed update budget."""
    x = init_x
    trajectory = [float(x)]
    losses = []
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    for _ in range(max_steps):
        loss, gradient = value_and_grad(x)
        losses.append(float(loss))
        x = x - lr * gradient
        trajectory.append(float(x))
    losses.append(float(loss_fn(x)))

    return x, jnp.array(trajectory), jnp.array(losses)


class TestWithinStratumOptimization:
    """Compare surrogate fitting with true geometric error inside one stratum."""

    def test_radius_optimization_reduces_exact_error(self) -> None:
        """Both methods improve exact area at a common, fixed update budget."""
        target_radius = 1.5
        c2 = jnp.array([1.0, 0.0])

        def exact_area(radius):
            return disk_disk_union_area(C1, radius, c2, R2)

        target_area = exact_area(jnp.array(target_radius))

        def loss_c(r1):
            a = Disk(center=C1, radius=r1)
            b = Disk(center=c2, radius=R2)
            # The stratum area dispatcher currently uses resolution 128.
            return (union_area(a, b, method="stratum") - target_area) ** 2

        def loss_a(r1):
            a = Disk(center=C1, radius=r1)
            b = Disk(center=c2, radius=R2)
            area = union_area(a, b, method="smoothing", k=0.1, beta=0.1, resolution=128)
            return (area - target_area) ** 2

        steps = 200
        for method, loss_fn in [("C", loss_c), ("A", loss_a)]:
            final, trajectory, losses = _optimize(
                loss_fn, jnp.array(0.8), lr=0.01, max_steps=steps
            )
            true_areas = jax.vmap(exact_area)(trajectory)
            true_losses = (true_areas - target_area) ** 2
            parameter_error = abs(float(final) - target_radius)
            print(
                f"\nMethod {method}: updates={len(trajectory) - 1}, "
                f"final_radius={float(final):.6f}, radius_error={parameter_error:.6f}, "
                f"surrogate_loss={float(losses[-1]):.3e}, "
                f"true_loss={float(true_losses[-1]):.3e}, "
                f"radius_error_below_0.01={parameter_error < 0.01}"
            )

            assert len(trajectory) == len(losses) == steps + 1
            # Strictly intersecting throughout: |r1-r2| < d < r1+r2, d=1.
            assert jnp.all((jnp.abs(trajectory - R2) < 1.0) & (trajectory + R2 > 1.0))
            assert jnp.all(jnp.isfinite(losses))
            assert losses[-1] < 1e-4
            # A shared geometric criterion, independent of each surrogate's bias.
            assert true_losses[-1] < 0.01 * true_losses[0]
            assert parameter_error < 0.05


class TestCrossStratumOptimization:
    """Characterize the disjoint plateau without claiming a boundary crossing."""

    def test_disjoint_center_optimization_does_not_cross_boundary(self) -> None:
        """Method C stays fixed; smoothing moves but neither crosses in 100 steps."""

        def exact_area(c2_x):
            return disk_disk_union_area(C1, R1, jnp.array([c2_x, C2_Y]), R2)

        target_area = exact_area(jnp.array(1.5))

        def loss_c(c2_x):
            a = Disk(center=C1, radius=R1)
            b = Disk(center=jnp.array([c2_x, C2_Y]), radius=R2)
            return (union_area(a, b, method="stratum") - target_area) ** 2

        def loss_a(c2_x):
            a = Disk(center=C1, radius=R1)
            b = Disk(center=jnp.array([c2_x, C2_Y]), radius=R2)
            area = union_area(a, b, method="smoothing", k=0.1, beta=0.1, resolution=128)
            return (area - target_area) ** 2

        init_x = jnp.array(3.0)
        grad_c = jax.grad(loss_c)(init_x)
        grad_a = jax.grad(loss_a)(init_x)
        # The exact area is constant for disjoint disks. Zero signal is correct.
        assert jax.grad(exact_area)(init_x) == 0.0
        assert grad_c == 0.0
        assert 0.0 < grad_a < jnp.inf

        steps = 100
        for method, loss_fn, initial_gradient in [
            ("C", loss_c, grad_c),
            ("A", loss_a, grad_a),
        ]:
            final, trajectory, losses = _optimize(
                loss_fn, init_x, lr=0.1, max_steps=steps
            )
            true_losses = (jax.vmap(exact_area)(trajectory) - target_area) ** 2
            # This analytic event test examines every recorded iterate.
            crossed = bool(jnp.any(jnp.abs(trajectory) < R1 + R2))
            print(
                f"\nMethod {method}: initial_gradient={float(initial_gradient):.8f}, "
                f"updates={len(trajectory) - 1}, final_x={float(final):.6f}, "
                f"moved={float(init_x - final):.6f}, crossed={crossed}, "
                f"surrogate_loss={float(losses[0]):.3e}->{float(losses[-1]):.3e}, "
                f"true_loss={float(true_losses[0]):.3e}->{float(true_losses[-1]):.3e}"
            )

            assert len(trajectory) == len(losses) == steps + 1
            assert jnp.all(jnp.isfinite(losses))
            assert not crossed
            assert jnp.allclose(true_losses, true_losses[0], rtol=1e-12, atol=1e-12)
            assert true_losses[-1] > 0.0
            if method == "C":
                assert jnp.all(trajectory == init_x)
            else:
                assert final < init_x
                assert losses[-1] < losses[0]
