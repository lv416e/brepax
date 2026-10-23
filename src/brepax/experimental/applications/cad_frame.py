"""Fixed working coordinates for dimension-driven CAD experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax._occt.backend import (
    BRepBuilderAPI_Transform,
    BRepCheck_Analyzer,
    gp_Trsf,
)
from brepax._occt.types import TopoDS_Shape


@dataclass(frozen=True)
class CadFrame:
    """Fixed host metadata with ``x_external = origin + length_scale * x_work``.

    ``length_scale`` is external length units per working length unit. Choose
    it once from a reference design size, before constructing CAD, and keep it
    fixed throughout editing. For a 40 mm reference expressed in metres, use
    ``length_scale=0.001`` to work at size 40. ``origin`` is in external units.
    Directions/rotations are unchanged by this positive uniform scale; any
    rigid rotation belongs to the supplied geometry model.

    Array conversions support JIT, vmap and differentiation when this frame is
    captured as fixed metadata. A degree-k scalar scales by s**k and its
    gradient with respect to external lengths by s**(k-1). The frame itself
    is not a differentiable model. Prefer x64 for CAD accuracy.

    Normalize design dimensions and dimensional meshing/smoothing tolerances
    BEFORE OCCT construction. This does not infer STEP units, repair existing
    B-Reps, or guarantee CAD accuracy at arbitrary scales. Output transforms
    retain OCCT's tolerance semantics; they do not tighten tolerances.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> frame = CadFrame(0.001, (1.0, 2.0, 3.0))
        >>> def sphere_volume(radius):
        ...     r = frame.working_lengths(radius)
        ...     return frame.external_measure(4 * jnp.pi * r**3 / 3, 3)
        >>> derivative = jax.jit(jax.grad(sphere_volume))(jnp.array(0.003))
    """

    length_scale: float
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.length_scale) or self.length_scale <= 0:
            raise ValueError("length_scale must be finite and positive")
        if len(self.origin) != 3 or not all(math.isfinite(x) for x in self.origin):
            raise ValueError("origin must contain three finite coordinates")
        # Own immutable copies even if a host caller supplied a mutable sequence.
        object.__setattr__(self, "length_scale", float(self.length_scale))
        object.__setattr__(self, "origin", tuple(float(x) for x in self.origin))

    def working_lengths(self, lengths: Float[Array, ...]) -> Float[Array, ...]:
        """Convert external lengths (including deflections) to working units."""
        return lengths / self.length_scale

    def external_measure(
        self, value: Float[Array, ...], length_power: Literal[1, 2, 3]
    ) -> Float[Array, ...]:
        """Restore a length, area or volume; JAX propagates the unit chain rule."""
        if length_power not in (1, 2, 3):
            raise ValueError("length_power must be 1, 2 or 3")
        return value * self.length_scale**length_power

    def external_points(self, points: Float[Array, "... 3"]) -> Float[Array, "... 3"]:
        """Restore point positions; evaluate scalar metrics before this step."""
        return points * self.length_scale + jnp.asarray(self.origin)

    def restore_shape(self, working_shape: TopoDS_Shape) -> TopoDS_Shape:
        """Copy a working B-Rep into external coordinates on the host.

        Does not copy triangulations. Raises ValueError for null input and
        RuntimeError if OCCT returns an invalid result. This is an output
        conversion, not a traced operation or a mesh-to-CAD reconstruction.
        """
        if working_shape.IsNull():
            raise ValueError("Cannot restore a null CAD shape")
        scale = self.length_scale
        x, y, z = self.origin
        transform = gp_Trsf()
        transform.SetValues(scale, 0.0, 0.0, x, 0.0, scale, 0.0, y, 0.0, 0.0, scale, z)
        shape = BRepBuilderAPI_Transform(working_shape, transform, True, False).Shape()
        if shape.IsNull() or not BRepCheck_Analyzer(shape).IsValid():
            raise RuntimeError("Invalid CAD shape after restoring external coordinates")
        return shape
