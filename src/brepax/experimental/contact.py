"""Analytic area and first shape derivatives for a circular cut in a rectangle.

This limited experiment uses Green's theorem and boundary flux, not a sampled
indicator or a correction to a future contact event. It does not edit a B-Rep.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class _Intersection(NamedTuple):
    area: Float[Array, ""]
    center_gradient: Float[Array, "2"]
    radius_gradient: Float[Array, ""]
    extents_gradient: Float[Array, "2"]


def _segment_factor(angle: Float[Array, "9"]) -> Float[Array, "9"]:
    """Evaluate angle - sin(angle) without cancellation for a small cap."""
    squared = angle * angle
    series = angle**3 * (
        1 / 6
        + squared
        * (
            -1 / 120
            + squared * (1 / 5040 + squared * (-1 / 362880 + squared / 39916800))
        )
    )
    return jnp.where(angle < 0.1, series, angle - jnp.sin(angle))


def _intersection_geometry(
    center: Float[Array, "2"],
    radius: Float[Array, ""],
    half_extents: Float[Array, "2"],
) -> _Intersection:
    # Eight possible circle/edge intersections partition the circle into a
    # fixed-size set of arcs. Clipped non-intersections only subdivide arcs.
    sides = jnp.stack((-half_extents, half_extents))
    ratios = (sides - center) / radius
    interior = jnp.abs(ratios) < 1
    safe_ratios = jnp.where(interior, ratios, 0)
    # Keep unused endpoints finite under ordinary AD too. This does not smooth
    # or move the actual contacts, unlike clipping to 1-epsilon.
    x_angles = jnp.where(
        interior[:, 0],
        jnp.arccos(safe_ratios[:, 0]),
        jnp.where(ratios[:, 0] <= -1, jnp.pi, 0),
    )
    y_angles = jnp.where(
        interior[:, 1],
        jnp.arcsin(safe_ratios[:, 1]),
        jnp.where(ratios[:, 1] <= -1, -jnp.pi / 2, jnp.pi / 2),
    )
    angles = jnp.sort(
        jnp.concatenate(
            (
                jnp.zeros(1, dtype=center.dtype),
                x_angles,
                2 * jnp.pi - x_angles,
                jnp.mod(y_angles, 2 * jnp.pi),
                jnp.mod(jnp.pi - y_angles, 2 * jnp.pi),
                jnp.array([2 * jnp.pi], dtype=center.dtype),
            )
        )
    )
    widths = jnp.diff(angles)
    midpoints = (angles[:-1] + angles[1:]) / 2
    midpoint_normals = jnp.stack((jnp.cos(midpoints), jnp.sin(midpoints)), axis=-1)
    samples = center + radius * midpoint_normals
    active = jnp.all(jnp.abs(samples) <= half_extents, axis=-1)

    # The nearest rectangle point to the circle center lies in the intersection
    # whenever it is nonempty. This gauge avoids subtracting large Green
    # integrals to recover a tiny overlap. Subtract before adding radius terms.
    gauge = jnp.clip(center, -half_extents, half_extents)
    points = (center - gauge) + radius * jnp.stack(
        (jnp.cos(angles), jnp.sin(angles)), axis=-1
    )
    cross = points[:-1, 0] * points[1:, 1] - points[:-1, 1] * points[1:, 0]
    arc_area = jnp.sum(
        jnp.where(active, (cross + radius**2 * _segment_factor(widths)) / 2, 0.0)
    )

    # Axis-aligned chords avoid the cancellation in a generic quadratic
    # discriminant, especially for tiny circles next to a long rectangle edge.
    distances = jnp.abs(sides - center)
    squared_chords = (radius - distances) * (radius + distances)
    intersects = distances < radius
    half_chords = jnp.where(
        intersects, jnp.sqrt(jnp.where(intersects, squared_chords, 1)), 0
    )
    other_center = center[::-1]
    other_extents = half_extents[::-1]
    lower = jnp.maximum(-half_chords, -other_extents - other_center)
    upper = jnp.minimum(half_chords, other_extents - other_center)
    lengths = jnp.where(intersects, jnp.maximum(upper - lower, 0), 0)
    # For each edge, (p-gauge) dot outward_normal is constant; its Green
    # contribution is this distance times the clipped edge length / 2.
    edge_offsets = jnp.stack((half_extents + gauge, half_extents - gauge))
    edge_area = jnp.sum(edge_offsets * lengths) / 2
    weights = jnp.where(active, 2 * jnp.sin(widths / 2), 0)
    center_gradient = radius * jnp.sum(weights[:, None] * midpoint_normals, axis=0)
    radius_gradient = radius * jnp.sum(jnp.where(active, widths, 0))
    extents_gradient = jnp.sum(lengths, axis=0)

    # Exact simple-region identities also remove trigonometric roundoff from
    # genuinely constant center derivatives and fully covered rectangles.
    inside = jnp.all(jnp.abs(center) + radius <= half_extents)
    outside = jnp.linalg.norm(jnp.maximum(jnp.abs(center) - half_extents, 0)) >= radius
    covered = jnp.linalg.norm(jnp.abs(center) + half_extents) <= radius
    area = jnp.where(inside, jnp.pi * radius**2, arc_area + edge_area)
    area = jnp.where(covered, 4 * jnp.prod(half_extents), jnp.where(outside, 0, area))
    center_gradient = jnp.where(inside | outside | covered, 0, center_gradient)
    radius_gradient = jnp.where(inside, 2 * jnp.pi * radius, radius_gradient)
    radius_gradient = jnp.where(outside | covered, 0, radius_gradient)
    extents_gradient = jnp.where(inside | outside, 0, extents_gradient)
    extents_gradient = jnp.where(covered, 4 * half_extents[::-1], extents_gradient)
    valid = (
        jnp.all(jnp.isfinite(center))
        & jnp.isfinite(radius)
        & (radius > 0)
        & jnp.all(jnp.isfinite(half_extents) & (half_extents > 0))
    )
    return _Intersection(
        jnp.where(valid, area, jnp.nan),
        jnp.where(valid, center_gradient, jnp.nan),
        jnp.where(valid, radius_gradient, jnp.nan),
        jnp.where(valid, extents_gradient, jnp.nan),
    )


@jax.custom_jvp
def circle_rectangle_intersection_area(
    center: Float[Array, "2"],
    radius: Float[Array, ""],
    half_extents: Float[Array, "2"],
) -> Float[Array, ""]:
    """Area of a disk intersected with a centered, axis-aligned rectangle.

    All inputs use one length unit. ``center`` is in the rectangle's local
    frame; ``radius`` and both ``half_extents`` must be positive and finite.
    Invalid numerical inputs return NaN for the value and first derivatives.
    Use floating JAX arrays of matching dtype; x64 is recommended near contact.

    Supports ``jit``, ``vmap``, JVP and reverse-mode first derivatives in all
    arguments, including at circle/edge and circle/corner contacts (C1).
    The custom JVP integrates normal boundary velocity for the same continuous
    area as the primal. It avoids differentiating singular arc endpoints whose
    contributions cancel and arbitrary AD conventions at tied sorted endpoints.
    Only first derivatives are part of this contract;
    second derivatives may not exist at contact.

    Scaling all lengths by s>0 scales area by s**2 and length derivatives by s.
    The local frame must move with the rectangle under rigid transformations.
    Tiny remaining areas and extreme radius/rectangle ratios can lose relative
    floating-point accuracy; this is not a general robust Boolean kernel.

    Example::

        area = circle_rectangle_intersection_area(
            jnp.array([1.5, 0.2]), jnp.array(1.8), jnp.array([3.0, 2.0])
        )
    """
    return _intersection_geometry(center, radius, half_extents).area


@circle_rectangle_intersection_area.defjvp
def _area_jvp(
    primals: tuple[Float[Array, "2"], Float[Array, ""], Float[Array, "2"]],
    tangents: tuple[Float[Array, "2"], Float[Array, ""], Float[Array, "2"]],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    result = _intersection_geometry(*primals)
    center_tangent, radius_tangent, extents_tangent = tangents
    derivative = (
        jnp.sum(result.center_gradient * center_tangent)
        + result.radius_gradient * radius_tangent
        + jnp.sum(result.extents_gradient * extents_tangent)
    )
    return result.area, derivative


def plate_with_hole_volume(
    center: Float[Array, "2"],
    radius: Float[Array, ""],
    half_extents: Float[Array, "2"],
    height: Float[Array, ""],
) -> Float[Array, ""]:
    """Remaining volume after a perpendicular circular through-cut in a plate.

    The stock is ``[-bx,bx] x [-by,by] x [0,height]``; the cutter spans its
    full height. The cut may open through edges or corners, unlike bounded
    fixed-topology hole editing. This function evaluates a scalar; it does
    not regenerate CAD or accept arbitrary STEP geometry.

    Inputs are floating JAX arrays in one length unit. Positive finite sizes
    are required; invalid inputs produce NaN values and first derivatives.
    First differentiation in all
    arguments and JAX transformations follow ``circle_rectangle_intersection_area``.
    Volume scales as length cubed and length derivatives as length squared.
    Interior center motion correctly has zero volume derivative. Near total
    stock removal, subtraction limits relative accuracy of the remaining volume.

    Example::

        radius = jnp.array(1.8)
        objective = lambda r: plate_with_hole_volume(
            jnp.array([1.5, 0.2]), r, jnp.array([3.0, 2.0]), jnp.array(2.0)
        )
        volume, dvolume_dr = jax.value_and_grad(objective)(radius)
    """
    area = 4 * jnp.prod(half_extents) - circle_rectangle_intersection_area(
        center, radius, half_extents
    )
    return area * height * jnp.where(jnp.isfinite(height) & (height > 0), 1.0, jnp.nan)
