"""3D axis-aligned box primitive defined by center and half-extents."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.primitives._base import Primitive


class Box(Primitive):
    """An axis-aligned box defined by center and half-extents.

    Attributes:
        center: Center of the box (3,).
        half_extents: Half-size in each dimension (3,).
    """

    center: Float[Array, "3"]
    half_extents: Float[Array, "3"]

    def sdf(self, x: Float[Array, "... 3"]) -> Float[Array, "..."]:
        """Signed distance from query points to the box surface."""
        q = jnp.abs(x - self.center) - self.half_extents
        # Outside: Euclidean distance to nearest surface point
        outside = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1)
        # Inside: negative of distance to nearest face
        inside = jnp.minimum(jnp.max(q, axis=-1), 0.0)
        return outside + inside  # type: ignore[no-any-return]

    def parameters(self) -> dict[str, Array]:
        """Return differentiable design parameters."""
        return {"center": self.center, "half_extents": self.half_extents}

    def volume(self) -> Float[Array, ""]:
        """Box volume: 8 * hx * hy * hz."""
        return 8.0 * jnp.prod(self.half_extents)
