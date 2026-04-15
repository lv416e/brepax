"""Stratum-aware Boolean operations with exact per-stratum gradients.

Uses jax.custom_vjp to provide analytically correct gradients within
each topological stratum. No smoothing or temperature parameters --
the forward pass evaluates the exact area formula, and the backward
pass dispatches to the correct gradient based on the stratum label.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.primitives._base import Primitive


def union_area_stratum(
    a: Primitive,
    b: Primitive,
) -> Float[Array, ""]:
    """Compute union area with stratum-aware exact gradients.

    The forward pass computes the exact analytical union area.
    The backward pass uses jax.custom_vjp to select the correct
    gradient formula based on which topological stratum the
    configuration belongs to.

    Args:
        a: First primitive (must be Disk).
        b: Second primitive (must be Disk).

    Returns:
        Exact union area as a differentiable scalar.
    """
    p_a = a.parameters()
    p_b = b.parameters()
    return _union_area_stratum_impl(
        p_a["center"],
        p_a["radius"],
        p_b["center"],
        p_b["radius"],
    )


@jax.custom_vjp
def _union_area_stratum_impl(
    c1: Float[Array, "2"],
    r1: Float[Array, ""],
    c2: Float[Array, "2"],
    r2: Float[Array, ""],
) -> Float[Array, ""]:
    """Core implementation with custom_vjp for stratum-aware gradients."""
    return _union_area_forward(c1, r1, c2, r2)


def _union_area_forward(
    c1: Float[Array, "2"],
    r1: Float[Array, ""],
    c2: Float[Array, "2"],
    r2: Float[Array, ""],
) -> Float[Array, ""]:
    """Exact union area using the analytical formula."""
    d = jnp.linalg.norm(c1 - c2)
    area1 = jnp.pi * r1**2
    area2 = jnp.pi * r2**2
    intersection = _safe_intersection_area(d, r1, r2)
    return area1 + area2 - intersection


def _safe_intersection_area(
    d: Float[Array, ""],
    r1: Float[Array, ""],
    r2: Float[Array, ""],
) -> Float[Array, ""]:
    """NaN-safe intersection area for use inside custom_vjp forward."""
    # See docs/explanation/jax_where_gradient_pitfall.md and ADR-0004.
    safe_d = jnp.maximum(d, 1e-10)
    eps = 1e-7
    cos_alpha = jnp.clip(
        (safe_d**2 + r1**2 - r2**2) / (2.0 * safe_d * r1),
        -1.0 + eps,
        1.0 - eps,
    )
    cos_beta = jnp.clip(
        (safe_d**2 + r2**2 - r1**2) / (2.0 * safe_d * r2),
        -1.0 + eps,
        1.0 - eps,
    )
    alpha = jnp.arccos(cos_alpha)
    beta = jnp.arccos(cos_beta)
    lens = r1**2 * (alpha - jnp.sin(2.0 * alpha) / 2.0) + r2**2 * (
        beta - jnp.sin(2.0 * beta) / 2.0
    )

    disjoint = d >= r1 + r2
    contained = d <= jnp.abs(r1 - r2)
    contained_area = jnp.pi * jnp.minimum(r1, r2) ** 2

    return jnp.where(disjoint, 0.0, jnp.where(contained, contained_area, lens))


def _stratum_label(
    d: Float[Array, ""],
    r1: Float[Array, ""],
    r2: Float[Array, ""],
) -> Float[Array, ""]:
    """Classify stratum: 0=disjoint, 1=intersecting, 2=contained."""
    return jnp.where(d >= r1 + r2, 0.0, jnp.where(d <= jnp.abs(r1 - r2), 2.0, 1.0))


# --- Per-stratum gradient functions ---
# Each computes d(union_area)/d(c1, r1, c2, r2) for one stratum.


def _grad_disjoint(c1, r1, c2, r2):
    """Gradient when disks are disjoint: area = pi*r1^2 + pi*r2^2."""
    return (
        jnp.zeros(2),  # d/dc1
        2.0 * jnp.pi * r1,  # d/dr1
        jnp.zeros(2),  # d/dc2
        2.0 * jnp.pi * r2,  # d/dr2
    )


def _grad_contained(c1, r1, c2, r2):
    """Gradient when one disk contains the other: area = pi*max(r1,r2)^2."""
    r_max = jnp.maximum(r1, r2)
    is_r1_bigger = r1 >= r2
    return (
        jnp.zeros(2),
        jnp.where(is_r1_bigger, 2.0 * jnp.pi * r_max, 0.0),
        jnp.zeros(2),
        jnp.where(is_r1_bigger, 0.0, 2.0 * jnp.pi * r_max),
    )


def _grad_intersecting(c1, r1, c2, r2):
    """Gradient for the intersecting stratum via autodiff of the lens formula."""
    # Use JAX autodiff on the analytical formula within this stratum.
    # The custom_vjp ensures only this branch's gradient is used.
    primals = (c1, r1, c2, r2)
    _, vjp_fn = jax.vjp(_union_area_forward, *primals)
    gc1, gr1, gc2, gr2 = vjp_fn(jnp.ones(()))
    return (gc1, gr1, gc2, gr2)


# --- custom_vjp plumbing ---


def _union_area_stratum_fwd(c1, r1, c2, r2):
    """Forward pass: compute primal + save residuals for backward."""
    primal = _union_area_forward(c1, r1, c2, r2)
    d = jnp.linalg.norm(c1 - c2)
    label = _stratum_label(d, r1, r2)
    residuals = (c1, r1, c2, r2, label)
    return primal, residuals


def _union_area_stratum_bwd(residuals, g_bar):
    """Backward pass: dispatch gradient based on stratum label."""
    c1, r1, c2, r2, label = residuals

    # Compute gradient for each stratum
    gc1_d, gr1_d, gc2_d, gr2_d = _grad_disjoint(c1, r1, c2, r2)
    gc1_i, gr1_i, gc2_i, gr2_i = _grad_intersecting(c1, r1, c2, r2)
    gc1_c, gr1_c, gc2_c, gr2_c = _grad_contained(c1, r1, c2, r2)

    # Select based on label (0=disjoint, 1=intersecting, 2=contained)
    def select(d_val, i_val, c_val):
        return jnp.where(
            label == 0.0,
            d_val,
            jnp.where(label == 2.0, c_val, i_val),
        )

    return (
        g_bar * select(gc1_d, gc1_i, gc1_c),
        g_bar * select(gr1_d, gr1_i, gr1_c),
        g_bar * select(gc2_d, gc2_i, gc2_c),
        g_bar * select(gr2_d, gr2_i, gr2_c),
    )


_union_area_stratum_impl.defvjp(_union_area_stratum_fwd, _union_area_stratum_bwd)
