"""Matched sampled Boolean estimators for contact-transition experiments.

C uses the unchanged private dispatcher to make its normally implicit domain
explicit. Its backward is a surrogate of a hard sampled forward. The sigmoid
baseline differentiates exactly the integrand C uses in intersecting cases;
it is not a different gradient algorithm in that stratum. Smooth intersection
is an experimental extension of Method A, whose public API only supports 2D
union. Neither experimental adapter changes the library's supported API.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.boolean.stratum import (
    _boolean_measure_with_dispatch,
    _detect_stratum_generic,
    _make_grid_nd,
)
from brepax.primitives import Box, Disk, Sphere
from brepax.primitives._base import Primitive
from brepax.primitives.finite_cylinder import FiniteCylinder

Measure = Callable[[Array, float, float], Array]
Pair = Callable[[Array], tuple[Primitive, Primitive]]
ContactKind = Literal["sphere", "cylinder", "box"]


@dataclass(frozen=True)
class SampledFamily:
    """Same domain, quadrature, and physical length scale for three estimators."""

    stratum: Measure
    sigmoid: Measure
    smoothing: Measure
    classify: Callable[[Array], Array]
    cell_width: float
    grid_points: int
    domain: tuple[tuple[float, ...], tuple[float, ...]]


@dataclass(frozen=True)
class Temperature:
    """Dimensionless smoothing lengths, multiplied by geometric-mean spacing."""

    k_over_h: float
    beta_over_h: float


TEMPERATURES = (
    Temperature(0.25, 0.25),
    Temperature(0.5, 0.5),
    Temperature(1.0, 1.0),
    Temperature(0.25, 0.5),
    Temperature(0.5, 0.25),
)


class Rectangle(Primitive):
    """Host-prepared stock for the 2D area-times-height comparison only."""

    center: Float[Array, 2]
    half_extents: Float[Array, 2]

    def sdf(self, x: Float[Array, "... 2"]) -> Float[Array, ...]:
        q = jnp.abs(x - self.center) - self.half_extents
        positive = jnp.maximum(q, 0)
        # Match Box's finite-norm regularization, but retain physical units.
        eps2 = (1e-10 * jnp.min(self.half_extents)) ** 2
        outside = jnp.sqrt(jnp.sum(positive**2, axis=-1) + eps2)
        return jnp.where(jnp.max(q, axis=-1) > 0, outside, 0) + jnp.minimum(
            jnp.max(q, axis=-1), 0
        )

    def volume(self) -> Float[Array, ""]:
        """The dispatcher's measure is 2D area for this benchmark primitive."""
        return 4 * jnp.prod(self.half_extents)

    def parameters(self) -> dict[str, Array]:
        return {"center": self.center, "half_extents": self.half_extents}


class RigidPrimitive(Primitive):
    """Evaluate transformed geometry on a freshly sampled, fixed world grid.

    This adapter only supports explicit-domain benchmarks: its metadata does
    not supply rotated bounds for the production auto-domain estimator.
    """

    primitive: Primitive
    rotation: Float[Array, "3 3"]
    shift: Float[Array, 3]

    def sdf(self, x: Float[Array, "... 3"]) -> Float[Array, ...]:
        return self.primitive.sdf((x - self.shift) @ self.rotation)

    def volume(self) -> Float[Array, ""]:
        return self.primitive.volume()

    def parameters(self) -> dict[str, Array]:
        return self.primitive.parameters()


def sampled_family(
    pair: Pair,
    resolution: int,
    domain: tuple[tuple[float, ...], tuple[float, ...]],
    *,
    multiplier: float = 1.0,
    operation: Literal["intersect", "subtract"] = "intersect",
) -> SampledFamily:
    """Build pure measures f(parameters, k/h, beta/h) on one frozen grid.

    Endpoint sampling matches current Method C, including its quadrature bias.
    Bounds are fixed during differentiation. Temperature factors must be
    positive. ``multiplier`` is a fixed extrusion height for 2D experiments.
    """
    if resolution < 2 or multiplier <= 0:
        raise ValueError("Resolution >= 2 and a positive multiplier are required")
    lo, hi = map(jnp.asarray, domain)
    if lo.ndim != 1 or lo.shape != hi.shape or not bool(jnp.all(hi > lo)):
        raise ValueError("Domain must have ordered, equally sized coordinate bounds")
    grid, cell = _make_grid_nd(lo, hi, resolution)
    width = cell ** (1 / lo.shape[0])

    def stratum(parameters: Array, unused_k: float, unused_beta: float) -> Array:
        a, b = pair(parameters)
        return multiplier * _boolean_measure_with_dispatch(
            a, b, operation, resolution, lo, hi
        )

    def operands(parameters: Array) -> tuple[Array, Array]:
        a, b = pair(parameters)
        first, second = a.sdf(grid), b.sdf(grid)
        return first, -second if operation == "subtract" else second

    def sigmoid(parameters: Array, unused_k: float, beta_factor: float) -> Array:
        first, second = operands(parameters)
        sdf = jnp.maximum(first, second)
        return multiplier * jnp.sum(jax.nn.sigmoid(-sdf / (width * beta_factor))) * cell

    def smoothing(parameters: Array, k_factor: float, beta_factor: float) -> Array:
        first, second = operands(parameters)
        k = width * k_factor
        sdf = k * jnp.logaddexp(first / k, second / k)
        return multiplier * jnp.sum(jax.nn.sigmoid(-sdf / (width * beta_factor))) * cell

    def classify(parameters: Array) -> Array:
        return _detect_stratum_generic(*pair(parameters), grid)

    return SampledFamily(
        stratum,
        sigmoid,
        smoothing,
        classify,
        float(width),
        resolution ** len(lo),
        domain,
    )


def contact_family(
    kind: ContactKind,
    resolution: int,
    *,
    scale: float = 1.0,
    rotation: Array | None = None,
    shift: Array | None = None,
    cubic_domain: bool = False,
) -> SampledFamily:
    """Equal unit shapes, physical penetration delta=2*scale-center_distance.

    Cylinders have equal fixed height 2*scale and parallel z axes. Their side
    contact has positive axial overlap; cap events and axis misalignment are
    different contact geometries. Cubic domains support the rotation controls.
    """
    if kind not in ("sphere", "cylinder", "box") or scale <= 0:
        raise ValueError("Use sphere/cylinder/box and a positive scale")
    origin = jnp.zeros(3) if shift is None else jnp.asarray(shift)
    matrix = jnp.eye(3) if rotation is None else jnp.asarray(rotation)

    def pair(delta: Array) -> tuple[Primitive, Primitive]:
        distance = 2 * scale - delta
        center = jnp.array([-distance / 2, 0.0, 0.0])
        radius = jnp.asarray(scale)
        if kind == "sphere":
            a: Primitive = Sphere(center, radius)
            b: Primitive = Sphere(-center, radius)
        elif kind == "cylinder":
            axis, height = jnp.array([0.0, 0.0, 1.0]), jnp.asarray(2 * scale)
            a = FiniteCylinder(center, axis, radius, height)
            b = FiniteCylinder(-center, axis, radius, height)
        else:
            extents = jnp.ones(3) * scale
            a, b = Box(center, extents), Box(-center, extents)
        if rotation is not None or shift is not None:
            a, b = RigidPrimitive(a, matrix, origin), RigidPrimitive(b, matrix, origin)
        return a, b

    limits = jnp.asarray((2.7, 2.7, 2.7) if cubic_domain else (2.6, 1.6, 1.6)) * scale
    domain = (tuple(map(float, origin - limits)), tuple(map(float, origin + limits)))
    return sampled_family(pair, resolution, domain)


def plate_family(
    half_extents: tuple[float, float],
    height: float,
    resolution: int,
    domain: tuple[tuple[float, float], tuple[float, float]],
    *,
    remaining_volume: bool = False,
) -> SampledFamily:
    """Compare on 2D samples times height; parameters are [cx, cy, radius].

    The fixed plate is centered at the origin. This reduction gives every
    method the same extrusion structure as the boundary-integral candidate.
    """
    extents = jnp.asarray(half_extents)
    if not bool(jnp.all(extents > 0)):
        raise ValueError("Positive half extents are required")
    stock = Rectangle(jnp.zeros(2), extents)

    def pair(parameters: Array) -> tuple[Primitive, Primitive]:
        return stock, Disk(parameters[:2], parameters[2])

    return sampled_family(
        pair,
        resolution,
        domain,
        multiplier=height,
        operation="subtract" if remaining_volume else "intersect",
    )


def analytic_contact(kind: ContactKind) -> Measure:
    """Ordinary AD through analytic point/line/face reference primals.

    At box contact JAX's chosen max derivative is a convention, not a unique
    classical derivative. The cylinder inactive branch is safe without moving
    positive contact offsets or clipping them to a fixed epsilon.
    """

    def measure(delta: Array, unused_k: float, unused_beta: float) -> Array:
        if kind == "sphere":
            positive = jnp.maximum(delta, 0.0)
            return jnp.pi * positive**2 / 2 - jnp.pi * positive**3 / 12
        if kind == "box":
            return 4 * jnp.maximum(delta, 0.0)
        safe = jnp.where(delta > 0, delta, 0.5)
        distance = 2 - safe
        lens = 2 * jnp.arccos(distance / 2) - distance / 2 * jnp.sqrt(safe * (4 - safe))
        return jnp.where(delta > 0, 2 * lens, 0.0)

    return measure
