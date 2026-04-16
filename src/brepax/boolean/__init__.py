"""Boolean operations on geometric primitives.

Provides a unified API for computing Boolean operations using different
differentiation strategies. Method dispatch is handled internally.
"""

from typing import Literal

from jaxtyping import Array, Float

from brepax.boolean.smoothing import union_area_smoothing
from brepax.boolean.stratum import (
    intersect_volume_stratum,
    subtract_volume_stratum,
    union_area_stratum,
    union_volume_stratum,
)
from brepax.primitives._base import Primitive

BooleanMethod = Literal["smoothing", "toi", "stratum"]


def union_area(
    a: Primitive,
    b: Primitive,
    *,
    method: BooleanMethod = "stratum",
    **kwargs: float | int | tuple[tuple[float, float], tuple[float, float]] | None,
) -> Float[Array, ""]:
    """Compute the union area of two 2D primitives."""
    if method == "smoothing":
        return union_area_smoothing(a, b, **kwargs)  # type: ignore[arg-type]
    elif method == "toi":
        raise NotImplementedError("TOI method not yet implemented")
    elif method == "stratum":
        return union_area_stratum(a, b)
    else:
        msg = f"unknown method: {method}"
        raise ValueError(msg)


def union_volume(
    a: Primitive,
    b: Primitive,
    *,
    method: BooleanMethod = "stratum",
    **kwargs: float | int | None,
) -> Float[Array, ""]:
    """Compute the union volume of two 3D primitives."""
    if method == "stratum":
        return union_volume_stratum(a, b, **kwargs)  # type: ignore[arg-type]
    elif method == "toi":
        raise NotImplementedError("TOI method not yet implemented")
    elif method == "smoothing":
        raise NotImplementedError("3D smoothing not yet implemented")
    else:
        msg = f"unknown method: {method}"
        raise ValueError(msg)


def subtract_volume(
    a: Primitive,
    b: Primitive,
    *,
    method: BooleanMethod = "stratum",
    **kwargs: float | int | None,
) -> Float[Array, ""]:
    """Compute volume of a minus b (subtract b from a)."""
    if method == "stratum":
        return subtract_volume_stratum(a, b, **kwargs)  # type: ignore[arg-type]
    else:
        msg = f"subtract not yet implemented for method: {method}"
        raise NotImplementedError(msg)


def intersect_volume(
    a: Primitive,
    b: Primitive,
    *,
    method: BooleanMethod = "stratum",
    **kwargs: float | int | None,
) -> Float[Array, ""]:
    """Compute intersection volume of a and b."""
    if method == "stratum":
        return intersect_volume_stratum(a, b, **kwargs)  # type: ignore[arg-type]
    else:
        msg = f"intersect not yet implemented for method: {method}"
        raise NotImplementedError(msg)
