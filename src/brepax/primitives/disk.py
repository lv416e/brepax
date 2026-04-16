"""2D disk primitive defined by center and radius."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.primitives._base import Primitive


class Disk(Primitive):
    """A 2D disk defined by center and radius.

    Attributes:
        center: Center coordinates (2,).
        radius: Scalar radius (must be positive).
    """

    center: Float[Array, "2"]
    radius: Float[Array, ""]

    def sdf(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        """Signed distance from query points to the disk boundary."""
        return jnp.linalg.norm(x - self.center, axis=-1) - self.radius  # type: ignore[no-any-return]

    def parameters(self) -> dict[str, Array]:
        """Return differentiable design parameters."""
        return {"center": self.center, "radius": self.radius}

    def volume(self) -> Float[Array, ""]:
        """Disk area: pi * r^2."""
        return jnp.pi * self.radius**2
