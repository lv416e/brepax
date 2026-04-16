"""3D infinite cone primitive defined by apex, axis, and half-angle."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.primitives._base import Primitive


class Cone(Primitive):
    """An infinite cone defined by apex position, axis direction, and half-angle.

    The cone extends infinitely from the apex along the axis direction.
    The SDF is positive outside, negative inside.

    Attributes:
        apex: Apex point of the cone (3,).
        axis: Unit direction vector from apex (3,). Must be normalized.
        angle: Half-angle in radians (0, pi/2).
    """

    apex: Float[Array, "3"]
    axis: Float[Array, "3"]
    angle: Float[Array, ""]

    def sdf(self, x: Float[Array, "... 3"]) -> Float[Array, "..."]:
        """Signed distance from query points to the cone surface."""
        v = x - self.apex
        # Distance along axis
        h = jnp.sum(v * self.axis, axis=-1)
        # Perpendicular distance from axis
        perp = jnp.sqrt(jnp.maximum(jnp.sum(v * v, axis=-1) - h**2, 1e-20))
        sin_a = jnp.sin(self.angle)
        cos_a = jnp.cos(self.angle)
        # Signed distance: positive outside cone, negative inside
        return jnp.where(
            h >= 0,
            perp * cos_a - h * sin_a,
            jnp.sqrt(perp**2 + h**2),
        )

    def parameters(self) -> dict[str, Array]:
        """Return differentiable design parameters."""
        return {"apex": self.apex, "axis": self.axis, "angle": self.angle}
