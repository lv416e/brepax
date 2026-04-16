"""3D torus primitive defined by center, axis, major radius, and minor radius."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.primitives._base import Primitive


class Torus(Primitive):
    """A torus defined by center, axis, major radius, and minor radius.

    Attributes:
        center: Center of the torus (3,).
        axis: Unit normal to the torus plane (3,). Must be normalized.
        major_radius: Distance from center to tube center.
        minor_radius: Tube radius.
    """

    center: Float[Array, "3"]
    axis: Float[Array, "3"]
    major_radius: Float[Array, ""]
    minor_radius: Float[Array, ""]

    def sdf(self, x: Float[Array, "... 3"]) -> Float[Array, "..."]:
        """Signed distance from query points to the torus surface."""
        v = x - self.center
        # Component along axis (height above torus plane)
        h = jnp.sum(v * self.axis, axis=-1)
        # Component in the torus plane (distance from center axis)
        v_plane_sq = jnp.sum(v * v, axis=-1) - h**2
        v_plane = jnp.sqrt(jnp.maximum(v_plane_sq, 1e-20))
        # Distance from the tube center ring
        q = jnp.sqrt((v_plane - self.major_radius) ** 2 + h**2)
        return q - self.minor_radius

    def parameters(self) -> dict[str, Array]:
        """Return differentiable design parameters."""
        return {
            "center": self.center,
            "major_radius": self.major_radius,
            "minor_radius": self.minor_radius,
        }

    def volume(self) -> Float[Array, ""]:
        """Torus volume: 2 * pi^2 * R * r^2."""
        return 2.0 * jnp.pi**2 * self.major_radius * self.minor_radius**2
