"""Closed-form solutions for two-sphere Boolean operations.

Provides analytically differentiable union volume, stratum classification,
and boundary distance for pairs of 3D spheres. Extends the two-disk
pattern to three dimensions.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float, Integer


def sphere_sphere_union_volume(
    c1: Float[Array, "3"],
    r1: Float[Array, ""],
    c2: Float[Array, "3"],
    r2: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute the union volume of two spheres analytically.

    Uses the spherical cap intersection formula:
    V_union = V_1 + V_2 - V_intersection.

    Args:
        c1: Center of sphere 1.
        r1: Radius of sphere 1.
        c2: Center of sphere 2.
        r2: Radius of sphere 2.

    Returns:
        Union volume as a scalar.
    """
    d = jnp.linalg.norm(c1 - c2)
    vol1 = (4.0 / 3.0) * jnp.pi * r1**3
    vol2 = (4.0 / 3.0) * jnp.pi * r2**3
    intersection = _intersection_volume(d, r1, r2)
    return vol1 + vol2 - intersection


def _intersection_volume(
    d: Float[Array, ""],
    r1: Float[Array, ""],
    r2: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute the intersection volume of two spheres given center distance.

    Uses the spherical cap formula. Each sphere contributes a cap whose
    height is determined by the geometry of the overlap.

    Handles three regimes via branchless selection:
    - Disjoint: d >= r1 + r2 -> 0
    - Contained: d <= |r1 - r2| -> (4/3)*pi*min(r1, r2)^3
    - Partial overlap: sum of two spherical cap volumes
    """
    # See docs/explanation/jax_where_gradient_pitfall.md and ADR-0004.
    safe_d = jnp.maximum(d, 1e-10)

    # Cap heights: h_i = r_i - x_i where x_i is the distance from
    # center_i to the intersection plane
    h1 = (2.0 * safe_d * r1 - safe_d**2 - r1**2 + r2**2) / (2.0 * safe_d)
    h2 = (2.0 * safe_d * r2 - safe_d**2 - r2**2 + r1**2) / (2.0 * safe_d)

    # Clamp to valid range to prevent NaN in unused branches
    h1 = jnp.clip(h1, 0.0, 2.0 * r1)
    h2 = jnp.clip(h2, 0.0, 2.0 * r2)

    # Spherical cap volume: V_cap = (pi * h^2 / 3) * (3r - h)
    cap1 = (jnp.pi * h1**2 / 3.0) * (3.0 * r1 - h1)
    cap2 = (jnp.pi * h2**2 / 3.0) * (3.0 * r2 - h2)
    lens = cap1 + cap2

    disjoint = d >= r1 + r2
    contained = d <= jnp.abs(r1 - r2)
    contained_vol = (4.0 / 3.0) * jnp.pi * jnp.minimum(r1, r2) ** 3

    return jnp.where(disjoint, 0.0, jnp.where(contained, contained_vol, lens))


def sphere_sphere_stratum_label(
    c1: Float[Array, "3"],
    r1: Float[Array, ""],
    c2: Float[Array, "3"],
    r2: Float[Array, ""],
) -> Integer[Array, ""]:
    """Classify the topological stratum of a two-sphere configuration.

    Returns:
        Integer-valued scalar: 0 = disjoint, 1 = intersecting, 2 = contained.
    """
    d = jnp.linalg.norm(c1 - c2)
    return jnp.where(
        d >= r1 + r2,
        0,
        jnp.where(d <= jnp.abs(r1 - r2), 2, 1),
    )


def sphere_sphere_boundary_distance(
    c1: Float[Array, "3"],
    r1: Float[Array, ""],
    c2: Float[Array, "3"],
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
