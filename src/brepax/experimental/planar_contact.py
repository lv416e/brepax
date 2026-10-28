"""One-sided volume response at a known isolated opposing planar contact.

This is a nonlinear function of a supplied rigid-motion direction, not a
custom JVP of Boolean volume or a detector of contact in arbitrary geometry.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from numpy.typing import ArrayLike


class PlanarContactPatch(eqx.Module):
    """Convex contact polygon in an oriented orthonormal plane frame.

    Create with :func:`prepare_planar_contact_patch` outside JAX tracing.
    ``vertices`` are counterclockwise local coordinates without a repeated
    closing vertex. ``origin + basis @ vertex`` gives a world point; the cross
    product of the basis columns points outward from the fixed solid toward
    the moving solid. Geometry uses a single length unit.

    The record contains no solids and cannot certify contact or isolation.
    Keep it fixed when querying motions. Polygon size is static under JIT;
    vmap over directions does not require padding or changing this size.
    """

    vertices: Float[Array, "N 2"]
    origin: Float[Array, "3"]
    basis: Float[Array, "3 2"]


def prepare_planar_contact_patch(
    vertices: ArrayLike, *, origin: ArrayLike, basis: ArrayLike
) -> PlanarContactPatch:
    """Validate a convex local polygon and its world frame on the host.

    Requires JAX x64. Raises ``ValueError`` for nonfinite/malformed inputs,
    repeated edges, nonpositive area, nonconvexity, or a nonorthonormal frame.
    Polygon half-plane checks reject negative orientations, including those
    within roundoff: accepting an uncertain negative sign can admit a thin
    self-crossing polygon. This conservative check may reject nearly collinear
    input after a coordinate transform. Frame checks use 64 float64 epsilons
    as a dimensionless roundoff allowance.
    This checks represented geometry, not the existence of a physical contact.
    No vertex is snapped or removed. Near-degenerate or extremely large local
    coordinates can lose precision and should be recentered by the caller.
    """
    if jnp.asarray(0.0).dtype != np.dtype("float64"):
        raise ValueError("Planar contact preparation requires JAX x64.")
    points = np.asarray(vertices, dtype=np.float64)
    position = np.asarray(origin, dtype=np.float64)
    axes = np.asarray(basis, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("Expected at least three polygon vertices with shape (N, 2).")
    if position.shape != (3,) or axes.shape != (3, 2):
        raise ValueError("Expected origin (3,) and basis (3, 2).")
    if not all(np.all(np.isfinite(item)) for item in (points, position, axes)):
        raise ValueError("Patch coordinates and frame must be finite.")
    roundoff = 64 * np.finfo(np.float64).eps
    if not np.allclose(axes.T @ axes, np.eye(2), rtol=0, atol=roundoff):
        raise ValueError("Patch basis columns must be orthonormal.")
    relative = points - points[0]
    extent = float(np.max(np.abs(relative)))
    if not np.isfinite(extent) or extent == 0:
        raise ValueError("Polygon must have a finite positive extent.")
    normalized = relative / extent
    edges = np.roll(normalized, -1, axis=0) - normalized
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Polygon must not repeat vertices, including the closing one.")
    offsets = normalized[None, :, :] - normalized[:, None, :]
    first_term = edges[:, None, 0] * offsets[:, :, 1]
    second_term = edges[:, None, 1] * offsets[:, :, 0]
    sides = first_term - second_term
    area_twice = np.sum(
        normalized[:, 0] * np.roll(normalized[:, 1], -1)
        - normalized[:, 1] * np.roll(normalized[:, 0], -1)
    )
    if area_twice <= 0 or np.any(sides < 0):
        raise ValueError("Polygon must be convex, counterclockwise, and have area.")
    return PlanarContactPatch(
        jnp.asarray(points), jnp.asarray(position), jnp.asarray(axes)
    )


def _positive_triangle_mean(speeds: Float[Array, "T 3"]) -> Float[Array, " T"]:
    """Mean positive affine speed from its three nodal values."""
    scale = jnp.max(jnp.abs(speeds), axis=-1)
    safe_scale = jnp.where(scale > 0, scale, 1.0)
    low, middle, high = jnp.moveaxis(
        jnp.sort(speeds / safe_scale[:, None], axis=-1), -1, 0
    )
    # For one positive vertex, the positive region is a similar triangle.
    raw_high = jnp.max(speeds, axis=-1)
    first_ratio = high / jnp.where(high > low, high - low, 1.0)
    second_ratio = high / jnp.where(high > middle, high - middle, 1.0)
    # Multiply the dimensional speed before the small area fractions: cubing
    # a normalized positive value could underflow before rescaling recovers it.
    one = raw_high * first_ratio * second_ratio / 3
    # With two positive vertices, subtracting a negative cap from the whole
    # integral loses tiny positive strips. This expansion has positive terms.
    negative = -low
    ratio = middle / jnp.where(high > 0, high, 1.0)
    numerator = negative * (ratio**2 + ratio + 1) + middle * (ratio + 1)
    positive_fraction = high / jnp.where(negative + high > 0, negative + high, 1.0)
    dimensionless_mean = numerator / (
        3 * jnp.where(negative + middle > 0, negative + middle, 1.0)
    )
    two = raw_high * positive_fraction * dimensionless_mean
    mean = jnp.where(
        high <= 0,
        0.0,
        jnp.where(
            low >= 0,
            scale * ((low + middle + high) / 3),
            jnp.where(middle <= 0, one, two),
        ),
    )
    return jnp.asarray(mean)


def planar_contact_volume_direction(
    patch: PlanarContactPatch,
    linear_velocity: Float[Array, "3"],
    angular_velocity: Float[Array, "3"],
    pivot: Float[Array, "3"],
) -> Float[Array, ""]:
    """Positive one-sided derivative of intersection volume at planar contact.

    The initial interiors must be disjoint, with one isolated opposing contact
    polygon and positive solid thickness on both sides. For the moving body's
    path ``pivot + exp(t*[angular_velocity]_cross) @ (x-pivot)
    + t*linear_velocity``, this returns the integral of
    ``max(-normal dot (linear_velocity + angular_velocity cross (x-pivot)), 0)``.
    It measures the limit as t decreases to zero from above. A positive gap,
    existing overlap, or an omitted positive-area contact changes that target.

    Supports ``jit`` and ``vmap`` over supplied finite directions with a fixed
    patch. This map is nonlinear in direction; it is not an ordinary JVP of
    volume. Differentiating this function would differentiate the response
    model, not the underlying Boolean volume. No derivative of this response
    map is part of the contract. Invalid motion components return NaN.

    With dimensionless t, linear velocity uses length units, angular velocity
    radians, and the result volume units. Proper rigid transformations must
    also transform the frame, pivot, and both velocities. Scaling all lengths
    and linear velocity by s>0 scales the result by s**3. Changing pivot to p'
    requires v' = v + angular_velocity cross (p'-pivot) for the same
    instantaneous motion direction (finite paths can differ at second order).
    For rigid bodies, union and fixed-minus-moving volume responses have the
    opposite sign. Finite edits still require an independent volume check.

    Uses exact affine integration over a triangle fan in float64. Very small
    caps, extreme scales, or cancellation in the represented velocity field
    can lose relative accuracy; no arbitrary-precision guarantee is made.

    Example::

        jax.config.update("jax_enable_x64", True)
        patch = prepare_planar_contact_patch(
            [[-1, -1], [1, -1], [1, 1], [-1, 1]],
            origin=[0, 0, 0], basis=[[1, 0], [0, 1], [0, 0]],
        )
        # Opposite tilts both create overlap: each response is 1.
        value = jax.jit(planar_contact_volume_direction)(
            patch, jnp.zeros(3), jnp.array([1.0, 0, 0]), jnp.zeros(3)
        )
    """
    normal = jnp.cross(patch.basis[:, 0], patch.basis[:, 1])
    at_origin = linear_velocity + jnp.cross(angular_velocity, patch.origin - pivot)
    slope = patch.basis.T @ jnp.cross(angular_velocity, normal)
    speeds = -jnp.dot(normal, at_origin) + patch.vertices @ slope
    left = patch.vertices[1:-1] - patch.vertices[0]
    right = patch.vertices[2:] - patch.vertices[0]
    areas = (left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]) / 2
    nodal = jnp.stack(
        (jnp.broadcast_to(speeds[0], speeds[1:-1].shape), speeds[1:-1], speeds[2:]),
        axis=-1,
    )
    result = jnp.sum(areas * _positive_triangle_mean(nodal))
    valid = (
        jnp.all(jnp.isfinite(linear_velocity))
        & jnp.all(jnp.isfinite(angular_velocity))
        & jnp.all(jnp.isfinite(pivot))
    )
    return jnp.where(valid, result, jnp.nan)
