"""Closed-form solutions for two-disk Boolean operations.

Provides analytically differentiable union area, stratum classification,
and boundary distance for pairs of 2D disks. Used as ground truth for
validating numerical gradient methods.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float, Integer


def disk_disk_union_area(
    c1: Float[Array, "2"],
    r1: Float[Array, ""],
    c2: Float[Array, "2"],
    r2: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute the union area of two disks analytically.

    Uses the standard two-circle intersection area formula:
    A_union = A_1 + A_2 - A_intersection.

    Args:
        c1: Center of disk 1.
        r1: Radius of disk 1.
        c2: Center of disk 2.
        r2: Radius of disk 2.

    Returns:
        Union area as a scalar.
    """
    d = jnp.linalg.norm(c1 - c2)
    area1 = jnp.pi * r1**2
    area2 = jnp.pi * r2**2
    intersection = _intersection_area(d, r1, r2)
    return area1 + area2 - intersection


def _intersection_area(
    d: Float[Array, ""],
    r1: Float[Array, ""],
    r2: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute the intersection area of two disks given center distance.

    Handles three regimes via branchless selection:
    - Disjoint: d >= r1 + r2 -> 0
    - Contained: d <= |r1 - r2| -> pi * min(r1, r2)^2
    - Partial overlap: standard lens area formula
    """
    # Clamp to [-1, 1] to keep arccos numerically safe
    cos_alpha = jnp.clip((d**2 + r1**2 - r2**2) / (2.0 * d * r1), -1.0, 1.0)
    cos_beta = jnp.clip((d**2 + r2**2 - r1**2) / (2.0 * d * r2), -1.0, 1.0)
    alpha = jnp.arccos(cos_alpha)
    beta = jnp.arccos(cos_beta)
    lens = r1**2 * (alpha - jnp.sin(2.0 * alpha) / 2.0) + r2**2 * (
        beta - jnp.sin(2.0 * beta) / 2.0
    )

    disjoint = d >= r1 + r2
    contained = d <= jnp.abs(r1 - r2)
    contained_area = jnp.pi * jnp.minimum(r1, r2) ** 2

    return jnp.where(disjoint, 0.0, jnp.where(contained, contained_area, lens))


def disk_disk_stratum_label(
    c1: Float[Array, "2"],
    r1: Float[Array, ""],
    c2: Float[Array, "2"],
    r2: Float[Array, ""],
) -> Integer[Array, ""]:
    """Classify the topological stratum of a two-disk configuration.

    Returns:
        Integer-valued scalar: 0 = disjoint, 1 = intersecting, 2 = contained.
    """
    d = jnp.linalg.norm(c1 - c2)
    return jnp.where(
        d >= r1 + r2,
        0,
        jnp.where(d <= jnp.abs(r1 - r2), 2, 1),
    )


def disk_disk_boundary_distance(
    c1: Float[Array, "2"],
    r1: Float[Array, ""],
    c2: Float[Array, "2"],
    r2: Float[Array, ""],
) -> Float[Array, ""]:
    """Distance from current configuration to the nearest stratum boundary.

    Measures the minimum of distance to external tangency (d = r1 + r2)
    and internal tangency (d = |r1 - r2|).
    """
    d = jnp.linalg.norm(c1 - c2)
    dist_external = jnp.abs(d - (r1 + r2))
    dist_internal = jnp.abs(d - jnp.abs(r1 - r2))
    return jnp.minimum(dist_external, dist_internal)
