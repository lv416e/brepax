"""3D sphere primitive defined by center and radius."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.primitives._base import Primitive


class Sphere(Primitive):
    """A 3D sphere defined by center and radius.

    Attributes:
        center: Center coordinates (3,).
        radius: Scalar radius (must be positive).
    """

    center: Float[Array, "3"]
    radius: Float[Array, ""]

    def sdf(self, x: Float[Array, "... 3"]) -> Float[Array, "..."]:
        """Signed distance from query points to the sphere surface."""
        return jnp.linalg.norm(x - self.center, axis=-1) - self.radius  # type: ignore[no-any-return]

    def parameters(self) -> dict[str, Array]:
        """Return differentiable design parameters."""
        return {"center": self.center, "radius": self.radius}

    def volume(self) -> Float[Array, ""]:
        """Sphere volume: 4/3 * pi * r^3."""
        return (4.0 / 3.0) * jnp.pi * self.radius**3
