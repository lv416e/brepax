"""3D plane primitive defined by normal vector and offset."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.primitives._base import Primitive


class Plane(Primitive):
    """An infinite plane (half-space boundary) defined by normal and offset.

    The SDF is positive on the side the normal points toward (outside),
    and negative on the opposite side (inside).

    Attributes:
        normal: Unit normal vector (3,). Must be normalized.
        offset: Signed distance from origin to the plane along the normal.
    """

    normal: Float[Array, "3"]
    offset: Float[Array, ""]

    def sdf(self, x: Float[Array, "... 3"]) -> Float[Array, "..."]:
        """Signed distance from query points to the plane."""
        return jnp.sum(x * self.normal, axis=-1) - self.offset

    def parameters(self) -> dict[str, Array]:
        """Return differentiable design parameters."""
        return {"normal": self.normal, "offset": self.offset}
