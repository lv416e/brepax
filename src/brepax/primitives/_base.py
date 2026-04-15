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
