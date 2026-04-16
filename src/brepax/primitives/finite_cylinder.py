"""3D finite cylinder primitive defined by center, axis, radius, and height."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.primitives._base import Primitive


class FiniteCylinder(Primitive):
    """A finite cylinder (capped) defined by center, axis, radius, and height.

    The cylinder is centered at `center` and extends `height/2` in each
    direction along `axis`.

    Attributes:
        center: Center of the cylinder (3,).
        axis: Unit direction vector of the axis (3,). Must be normalized.
        radius: Cylinder radius (must be positive).
        height: Total height along the axis (must be positive).
    """

    center: Float[Array, "3"]
    axis: Float[Array, "3"]
    radius: Float[Array, ""]
    height: Float[Array, ""]

    def sdf(self, x: Float[Array, "... 3"]) -> Float[Array, "..."]:
        """Signed distance from query points to the finite cylinder surface."""
        v = x - self.center
        # Axial distance from center
        h = jnp.sum(v * self.axis, axis=-1)
        # Perpendicular distance from axis
        perp_sq = jnp.sum(v * v, axis=-1) - h**2
        perp = jnp.sqrt(jnp.maximum(perp_sq, 1e-20))

        # Signed distances to the two constraints
        d_radial = perp - self.radius
        d_axial = jnp.abs(h) - self.height / 2.0

        # SDF of intersection of infinite cylinder and slab.
        # Use sqrt with eps to avoid NaN gradient at [0, 0].
        dr = jnp.maximum(d_radial, 0.0)
        da = jnp.maximum(d_axial, 0.0)
        outside = jnp.sqrt(dr**2 + da**2 + 1e-20)
        # Subtract the eps contribution when truly outside
        outside = jnp.where((dr > 0) | (da > 0), outside, 0.0)
        inside = jnp.minimum(jnp.maximum(d_radial, d_axial), 0.0)
        return outside + inside

    def parameters(self) -> dict[str, Array]:
        """Return differentiable design parameters."""
        return {
            "center": self.center,
            "axis": self.axis,
            "radius": self.radius,
            "height": self.height,
        }

    def volume(self) -> Float[Array, ""]:
        """Finite cylinder volume: pi * r^2 * h."""
        return jnp.pi * self.radius**2 * self.height
