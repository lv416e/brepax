"""B-spline surface evaluation via Cox-de Boor basis functions.

Implements the standard Cox-de Boor recursion for B-spline basis
functions, followed by tensor-product evaluation for surfaces.
All operations use ``jnp`` for JIT and gradient compatibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def bspline_basis(
    t: Float[Array, ""],
    knots: Float[Array, ...],
    degree: int,
    n_basis: int,
) -> Float[Array, ...]:
    """Evaluate all B-spline basis functions at parameter ``t``.

    Uses the Cox-de Boor recursion, vectorized per degree level.

    Args:
        t: Parameter value (scalar).
        knots: Knot vector of length ``n_basis + degree + 1``.
        degree: Polynomial degree.
        n_basis: Number of basis functions (= number of control points
            along this parametric direction).

    Returns:
        Basis function values, shape ``(n_basis,)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> knots = jnp.array([0., 0., 0., 0., 1., 1., 1., 1.])
        >>> vals = bspline_basis(jnp.array(0.5), knots, degree=3, n_basis=4)
    """
    n_spans = n_basis + degree

    left = knots[:n_spans]
    right = knots[1 : n_spans + 1]
    basis = jnp.where((left <= t) & (t < right), 1.0, 0.0)
    # Clamped right endpoint: last non-degenerate span is closed on right
    basis = basis.at[n_basis - 1].set(
        jnp.where(t >= knots[n_basis], 1.0, basis[n_basis - 1])
    )

    for d in range(1, degree + 1):
        size = n_spans - d
        denom_l = knots[d : d + size] - knots[:size]
        # Safe division: avoid 0/0 NaN in backward pass
        safe_l = jnp.where(denom_l > 0, denom_l, 1.0)
        alpha_l = jnp.where(denom_l > 0, (t - knots[:size]) / safe_l, 0.0)

        denom_r = knots[d + 1 : d + 1 + size] - knots[1 : 1 + size]
        safe_r = jnp.where(denom_r > 0, denom_r, 1.0)
        alpha_r = jnp.where(
            denom_r > 0, (knots[d + 1 : d + 1 + size] - t) / safe_r, 0.0
        )

        basis = alpha_l * basis[:size] + alpha_r * basis[1 : size + 1]

    return basis


def evaluate_surface(
    control_points: Float[Array, "nu nv 3"],
    knots_u: Float[Array, ...],
    knots_v: Float[Array, ...],
    degree_u: int,
    degree_v: int,
    u: Float[Array, ""],
    v: Float[Array, ""],
) -> Float[Array, 3]:
    """Evaluate a B-spline surface at parameters ``(u, v)``.

    Computes ``S(u, v) = N_u^T P N_v`` via tensor-product basis.

    Args:
        control_points: Control point grid, shape ``(n_u, n_v, 3)``.
        knots_u: Knot vector in u-direction.
        knots_v: Knot vector in v-direction.
        degree_u: Polynomial degree in u.
        degree_v: Polynomial degree in v.
        u: Parameter in u-direction (scalar).
        v: Parameter in v-direction (scalar).

    Returns:
        Surface point, shape ``(3,)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> pts = jnp.zeros((4, 4, 3))
        >>> k = jnp.array([0., 0., 0., 0., 1., 1., 1., 1.])
        >>> pt = evaluate_surface(pts, k, k, 3, 3,
        ...     jnp.array(0.5), jnp.array(0.5))
    """
    n_u = control_points.shape[0]
    n_v = control_points.shape[1]
    basis_u = bspline_basis(u, knots_u, degree_u, n_u)
    basis_v = bspline_basis(v, knots_v, degree_v, n_v)
    return jnp.einsum("i,ijk,j->k", basis_u, control_points, basis_v)


def evaluate_surface_derivs(
    control_points: Float[Array, "nu nv 3"],
    knots_u: Float[Array, ...],
    knots_v: Float[Array, ...],
    degree_u: int,
    degree_v: int,
    u: Float[Array, ""],
    v: Float[Array, ""],
) -> tuple[Float[Array, 3], Float[Array, 3], Float[Array, 3]]:
    """Evaluate surface point and partial derivatives at ``(u, v)``.

    Computes ``S(u,v)``, ``dS/du``, and ``dS/dv`` via ``jax.grad``
    applied to each coordinate of :func:`evaluate_surface`.

    Args:
        control_points: Control point grid, shape ``(n_u, n_v, 3)``.
        knots_u: Knot vector in u-direction.
        knots_v: Knot vector in v-direction.
        degree_u: Polynomial degree in u.
        degree_v: Polynomial degree in v.
        u: Parameter in u-direction (scalar).
        v: Parameter in v-direction (scalar).

    Returns:
        Tuple of ``(point, du, dv)`` each with shape ``(3,)``.
    """

    def _eval_component(
        u_val: Float[Array, ""], v_val: Float[Array, ""], idx: int
    ) -> Float[Array, ""]:
        return evaluate_surface(
            control_points, knots_u, knots_v, degree_u, degree_v, u_val, v_val
        )[idx]

    point = evaluate_surface(control_points, knots_u, knots_v, degree_u, degree_v, u, v)

    du = jnp.array([jax.grad(_eval_component, argnums=0)(u, v, i) for i in range(3)])
    dv = jnp.array([jax.grad(_eval_component, argnums=1)(u, v, i) for i in range(3)])

    return point, du, dv


__all__ = [
    "bspline_basis",
    "evaluate_surface",
    "evaluate_surface_derivs",
]
