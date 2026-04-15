"""Boolean operations on geometric primitives.

Provides a unified API for computing Boolean operations using different
differentiation strategies. Method dispatch is handled internally;
callers use the same `union_area()` interface regardless of method.
"""

from typing import Literal

from jaxtyping import Array, Float

from brepax.boolean.smoothing import union_area_smoothing
from brepax.primitives._base import Primitive

BooleanMethod = Literal["smoothing", "toi", "stratum"]


def union_area(
    a: Primitive,
    b: Primitive,
    *,
    method: BooleanMethod = "stratum",
    **kwargs: float | int | tuple[tuple[float, float], tuple[float, float]] | None,
) -> Float[Array, ""]:
    """Compute the union area of two primitives.

    Dispatches to the appropriate backend based on method selection.
    All methods return a differentiable scalar area estimate.

    Args:
        a: First primitive.
        b: Second primitive.
        method: Differentiation strategy. One of:
            - "smoothing": Smooth-min SDF + sigmoid soft indicator.
              kwargs: k, beta, resolution, domain.
            - "toi": Time-of-impact boundary correction (not yet implemented).
            - "stratum": Stratum-aware tracking (not yet implemented).
        **kwargs: Method-specific parameters passed to the backend.

    Returns:
        Union area as a differentiable scalar.

    Raises:
        ValueError: If method is unknown.
        NotImplementedError: If method is not yet implemented.
    """
    if method == "smoothing":
        return union_area_smoothing(a, b, **kwargs)  # type: ignore[arg-type]
    elif method == "toi":
        raise NotImplementedError("TOI method not yet implemented")
    elif method == "stratum":
        raise NotImplementedError("stratum method not yet implemented")
    else:
        msg = f"unknown method: {method}"
        raise ValueError(msg)
