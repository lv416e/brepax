"""Abstract base class for geometric primitives."""

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Float


class Primitive(eqx.Module):
    """Base class for all geometric primitives.

    Each primitive exposes an SDF, gradient w.r.t. spatial coordinates,
    and a list of design parameters that gradients can flow into.
    """

    @abstractmethod
    def sdf(self, x: Float[Array, "... dim"]) -> Float[Array, "..."]:
        """Evaluate the signed distance function at query points."""
        raise NotImplementedError

    @abstractmethod
    def parameters(self) -> dict[str, Array]:
        """Return differentiable design parameters."""
        raise NotImplementedError

    def volume(self) -> Float[Array, ""]:
        """Analytical volume of this primitive.

        Returns the finite volume for bounded primitives (Sphere, Box, etc.).
        Unbounded primitives (Cylinder, Plane, Cone) return inf.
        Override in subclasses with known analytical formulas.
        Differentiable via jax.grad for gradient computation.
        """
        # Default: unbounded primitive has infinite volume
        import jax.numpy as jnp

        return jnp.array(jnp.inf)
