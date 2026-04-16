"""3D infinite cylinder primitive defined by axis, point on axis, and radius."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.primitives._base import Primitive


class Cylinder(Primitive):
    """An infinite cylinder defined by a point on the axis, axis direction, and radius.

    The SDF measures perpendicular distance from the axis line minus the radius.
    The cylinder extends infinitely along the axis direction.

    Attributes:
        point: A point on the cylinder axis (3,).
        axis: Unit direction vector of the axis (3,). Must be normalized.
        radius: Scalar radius (must be positive).
    """

    point: Float[Array, "3"]
    axis: Float[Array, "3"]
    radius: Float[Array, ""]

    def sdf(self, x: Float[Array, "... 3"]) -> Float[Array, "..."]:
        """Signed distance from query points to the cylinder surface.

        Computes the perpendicular distance from each query point to
        the cylinder axis, then subtracts the radius.
        """
        # Vector from axis point to query
        v = x - self.point
        # Project onto axis to get parallel component
        proj_len = jnp.sum(v * self.axis, axis=-1, keepdims=True)
        # Perpendicular component
        perp = v - proj_len * self.axis
        perp_dist = jnp.linalg.norm(perp, axis=-1)
        return perp_dist - self.radius  # type: ignore[no-any-return]

    def parameters(self) -> dict[str, Array]:
        """Return differentiable design parameters."""
        return {"point": self.point, "axis": self.axis, "radius": self.radius}
