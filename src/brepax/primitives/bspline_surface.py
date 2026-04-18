"""B-spline surface primitive with differentiable SDF.

Wraps the NURBS evaluation and closest-point projection pipeline
as a :class:`~brepax.primitives.Primitive` so that existing Boolean
operations and metrics work with NURBS surfaces automatically.

The surface is unbounded (a single open patch, not a closed solid),
so ``volume()`` returns infinity and Boolean operations fall back to
grid-based evaluation, matching the pattern used by
:class:`~brepax.primitives.Plane` and
:class:`~brepax.primitives.Cylinder`.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.nurbs.evaluate import evaluate_surface
from brepax.nurbs.sdf import bspline_sdf
from brepax.primitives._base import Primitive


class BSplineSurface(Primitive):
    """B-spline surface defined by a control point grid and knot vectors.

    The SDF is computed via closest-point projection onto the surface.
    Control points are differentiable design variables: ``jax.grad``
    flows through the unrolled Newton projection.

    Attributes:
        control_points: Control point grid, shape ``(n_u, n_v, 3)``.
        knots_u: Knot vector in u-direction.
        knots_v: Knot vector in v-direction.
        degree_u: Polynomial degree in u.
        degree_v: Polynomial degree in v.

    Examples:
        >>> import jax.numpy as jnp
        >>> pts = jnp.array([[[0,0,0],[1,0,0]],
        ...                  [[0,1,0],[1,1,0]]], dtype=float)
        >>> knots = jnp.array([0., 0., 1., 1.])
        >>> surf = BSplineSurface(
        ...     control_points=pts, knots_u=knots, knots_v=knots,
        ...     degree_u=1, degree_v=1,
        ... )
        >>> d = surf.sdf(jnp.array([0.5, 0.5, 1.0]))
    """

    control_points: Float[Array, "nu nv 3"]
    knots_u: Array = eqx.field()
    knots_v: Array = eqx.field()
    degree_u: int = eqx.field(static=True)
    degree_v: int = eqx.field(static=True)
    weights: Array | None = eqx.field(default=None)
    param_u_range: tuple[float, float] | None = eqx.field(default=None, static=True)
    param_v_range: tuple[float, float] | None = eqx.field(default=None, static=True)
    trim_polygon: Array | None = eqx.field(default=None)
    trim_mask: Array | None = eqx.field(default=None)

    def sdf(self, x: Float[Array, "... 3"]) -> Float[Array, "..."]:
        """Signed distance from query points to the B-spline surface.

        For batched inputs, each point is projected independently via
        Newton iteration.
        """
        shape = x.shape[:-1]
        flat = x.reshape(-1, 3)

        # Coarse grid sampling: compute once, reuse for all query points
        from brepax.nurbs.projection import _COARSE_GRID

        u_lo = self.knots_u[self.degree_u]
        u_hi = self.knots_u[-self.degree_u - 1]
        v_lo = self.knots_v[self.degree_v]
        v_hi = self.knots_v[-self.degree_v - 1]
        us = jnp.linspace(u_lo, u_hi, _COARSE_GRID)
        vs = jnp.linspace(v_lo, v_hi, _COARSE_GRID)
        u_grid, v_grid = jnp.meshgrid(us, vs, indexing="ij")
        u_flat_g = u_grid.ravel()
        v_flat_g = v_grid.ravel()

        def _eval_sample(u: Array, v: Array) -> Array:
            return evaluate_surface(
                self.control_points,
                self.knots_u,
                self.knots_v,
                self.degree_u,
                self.degree_v,
                u,
                v,
                self.weights,
            )

        samples = jax.vmap(_eval_sample)(u_flat_g, v_flat_g)

        def _single_sdf(q: Array) -> Array:
            # Find closest coarse sample (stop_gradient: argmin is non-diff)
            dists = jnp.sum((samples - q) ** 2, axis=-1)
            best = jnp.argmin(dists)
            u0 = jax.lax.stop_gradient(u_flat_g[best])
            v0 = jax.lax.stop_gradient(v_flat_g[best])
            return bspline_sdf(
                q,
                self.control_points,
                self.knots_u,
                self.knots_v,
                self.degree_u,
                self.degree_v,
                u0=u0,
                v0=v0,
                weights=self.weights,
                param_u_range=self.param_u_range,
                param_v_range=self.param_v_range,
            )

        result = jax.vmap(_single_sdf)(flat)
        return result.reshape(shape)

    def parameters(self) -> dict[str, Array]:
        """Return differentiable design parameters."""
        params: dict[str, Array] = {
            "control_points": self.control_points,
            "knots_u": self.knots_u,
            "knots_v": self.knots_v,
        }
        if self.weights is not None:
            params["weights"] = self.weights
        return params
