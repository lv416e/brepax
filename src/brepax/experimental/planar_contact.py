"""One-sided volume responses from supplied opposing planar contact patches.

This is a nonlinear function of a supplied rigid-motion direction, not a
custom JVP of Boolean volume or a detector of contact in arbitrary geometry.
"""

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float
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


class PlanarContactSet(eqx.Module):
    """Packed contact triangles with a fixed capacity and padding mask.

    Create with :func:`prepare_planar_contact_set` outside JAX tracing.
    Each active row stores one triangle in a patch's local plane coordinates,
    together with that plane's origin and oriented orthonormal basis. Keeping
    local coordinates avoids constructing large world-space triangle vertices.
    The leading dimension is static under JIT; inactive rows contribute zero.

    This record carries supplied geometry, not proof that its patches form a
    complete, nonoverlapping physical contact set. Keep it fixed when querying
    motion directions. Its arrays are not a differentiable CAD construction.
    """

    vertices: Float[Array, "T 3 2"]
    origins: Float[Array, "T 3"]
    bases: Float[Array, "T 3 2"]
    active: Bool[Array, " T"]


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


def prepare_planar_contact_set(
    patches: Sequence[PlanarContactPatch], *, capacity: int | None = None
) -> PlanarContactSet:
    """Validate and pack supplied patches into masked triangle arrays.

    Requires JAX x64 and validates each patch using
    :func:`prepare_planar_contact_patch`. ``capacity`` counts triangles, not
    patches. Its default is the sum of ``len(vertices)-2`` over the patches;
    a larger capacity pads with inactive rows. An empty input is allowed and
    has zero default capacity. Raises ``ValueError`` for a negative/noninteger
    capacity, insufficient capacity, or an invalid patch.

    Patch interiors must be disjoint as surface regions and the collection
    must include every positive-area contact relevant to the requested volume
    response. These geometric conditions are the caller's responsibility:
    this function neither detects overlaps nor removes duplicates. Repeating
    a patch counts its contribution twice. Edge-sharing partitions are valid.
    All patches must use one length unit and outward normals from the same
    fixed body toward the same moving body.
    """
    if jnp.asarray(0.0).dtype != np.dtype("float64"):
        raise ValueError("Planar contact preparation requires JAX x64.")
    validated = tuple(
        prepare_planar_contact_patch(
            patch.vertices, origin=patch.origin, basis=patch.basis
        )
        for patch in patches
    )
    count = sum(patch.vertices.shape[0] - 2 for patch in validated)
    if capacity is None:
        capacity = count
    if isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)):
        raise ValueError("Triangle capacity must be a nonnegative integer.")
    if capacity < 0:
        raise ValueError("Triangle capacity must be a nonnegative integer.")
    if capacity < count:
        raise ValueError(f"Triangle capacity {capacity} is smaller than {count}.")
    vertices = np.zeros((capacity, 3, 2))
    origins = np.zeros((capacity, 3))
    bases = np.zeros((capacity, 3, 2))
    offset = 0
    for patch in validated:
        points = np.asarray(patch.vertices)
        size = len(points) - 2
        rows = slice(offset, offset + size)
        vertices[rows] = np.stack(
            (np.broadcast_to(points[0], (size, 2)), points[1:-1], points[2:]),
            axis=1,
        )
        origins[rows] = np.asarray(patch.origin)
        bases[rows] = np.asarray(patch.basis)
        offset += size
    return PlanarContactSet(
        jnp.asarray(vertices),
        jnp.asarray(origins),
        jnp.asarray(bases),
        jnp.arange(capacity) < count,
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


def planar_contact_set_volume_direction(
    contacts: PlanarContactSet,
    linear_velocity: Float[Array, "3"],
    angular_velocity: Float[Array, "3"],
    pivot: Float[Array, "3"],
) -> Float[Array, ""]:
    """Sum positive one-sided overlap responses over supplied contact patches.

    Every active triangle integrates the positive normal approach speed for
    the common rigid motion specified by ``linear_velocity``,
    ``angular_velocity`` and ``pivot``. For two finite polyhedral bodies with
    initially disjoint interiors, this is the intersection-volume derivative
    when the supplied patches cover all positive-area opposing contacts once
    and the solids have positive thickness there. Contacts may share edges.
    Omitted or duplicated patches invalidate the full-body interpretation.
    Gapped, edge-only and point-only polyhedral contacts contribute no first
    order volume, but can contribute to a finite edit. A zero response is not
    a collision-free certificate; independently check every finite edit.

    Supports ``jit`` and ``vmap`` over directions and equally padded sets.
    Empty sets return zero for finite motions. Nonfinite motion or pivot
    components return NaN, including for an empty set. Uses the same stable
    affine triangle integration as :func:`planar_contact_volume_direction`,
    with the same floating-point limits. Accumulation over many triangles can
    introduce additional summation error.

    Units, rigid covariance, cubic length scaling and compensated-pivot laws
    match :func:`planar_contact_volume_direction`. This is a nonlinear map of
    the supplied direction, not an ordinary JVP. No derivative of this map or
    of the packed geometry is part of the contract.

    Example::

        unit_square = [[0, 0], [1, 0], [1, 1], [0, 1]]
        floor = prepare_planar_contact_patch(
            unit_square, origin=[0, 0, 0], basis=[[1, 0], [0, 1], [0, 0]]
        )
        wall = prepare_planar_contact_patch(
            unit_square, origin=[0, 0, 0], basis=[[0, 0], [1, 0], [0, 1]]
        )
        contacts = prepare_planar_contact_set((floor, wall), capacity=8)
        # Into both walls: each unit-area patch contributes one.
        response = planar_contact_set_volume_direction(
            contacts, jnp.array([-1.0, 0, -1]), jnp.zeros(3), jnp.zeros(3)
        )  # 2.0
    """
    normals = jnp.cross(contacts.bases[:, :, 0], contacts.bases[:, :, 1])
    at_origins = linear_velocity + jnp.cross(angular_velocity, contacts.origins - pivot)
    slopes = jnp.einsum(
        "tji,tj->ti", contacts.bases, jnp.cross(angular_velocity, normals)
    )
    speeds = -jnp.sum(normals * at_origins, axis=-1)[:, None] + jnp.einsum(
        "tni,ti->tn", contacts.vertices, slopes
    )
    left = contacts.vertices[:, 1] - contacts.vertices[:, 0]
    right = contacts.vertices[:, 2] - contacts.vertices[:, 0]
    areas = (left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]) / 2
    contributions = areas * _positive_triangle_mean(speeds)
    result = jnp.sum(jnp.where(contacts.active, contributions, 0.0))
    valid = (
        jnp.all(jnp.isfinite(linear_velocity))
        & jnp.all(jnp.isfinite(angular_velocity))
        & jnp.all(jnp.isfinite(pivot))
    )
    return jnp.where(valid, result, jnp.nan)
