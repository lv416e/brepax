"""Matched sampled estimators for convex-prism rigid face contact.

Only the SDF preparation is new. Method C, its detection, quadrature, and
backward are the unchanged repository implementation. The other estimators
use the matched contact-transition benchmark integrands. All return linear
gradient proxies; no classical gradient of exact contact volume is claimed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from benchmarks.contact_transition.baselines import RigidPrimitive
from brepax.boolean.stratum import (
    _boolean_measure_with_dispatch,
    _detect_stratum_generic,
    _make_grid_nd,
)
from brepax.primitives import Box
from brepax.primitives._base import Primitive


class ConvexPrism(Primitive):
    """Exact segment-distance SDF for a CCW convex polygon's finite extrusion."""

    vertices: Float[Array, "n 2"]
    lower: Float[Array, ""]
    upper: Float[Array, ""]

    def sdf(self, points: Float[Array, "... 3"]) -> Float[Array, ...]:
        start = self.vertices
        edge = jnp.roll(start, -1, axis=0) - start
        relative = points[..., None, :2] - start
        projection = jnp.clip(
            jnp.sum(relative * edge, axis=-1) / jnp.sum(edge * edge, axis=-1),
            0.0,
            1.0,
        )
        residual = relative - projection[..., None] * edge
        distance_squared = jnp.min(jnp.sum(residual * residual, axis=-1), axis=-1)
        # No derivative is claimed at the polygon's segment-distance ties.
        distance = jnp.where(
            distance_squared > 0,
            jnp.sqrt(jnp.where(distance_squared > 0, distance_squared, 1.0)),
            0.0,
        )
        cross = edge[:, 0] * relative[..., 1] - edge[:, 1] * relative[..., 0]
        polygon_distance = jnp.where(jnp.all(cross >= 0, axis=-1), -distance, distance)
        axial_distance = jnp.maximum(
            self.lower - points[..., 2], points[..., 2] - self.upper
        )
        pair = jnp.stack((polygon_distance, axial_distance), axis=-1)
        positive_squared = jnp.sum(jnp.maximum(pair, 0.0) ** 2, axis=-1)
        outside = jnp.where(
            positive_squared > 0,
            jnp.sqrt(jnp.where(positive_squared > 0, positive_squared, 1.0)),
            0.0,
        )
        return outside + jnp.minimum(jnp.max(pair, axis=-1), 0.0)

    def volume(self) -> Float[Array, ""]:
        following = jnp.roll(self.vertices, -1, axis=0)
        twice_area = jnp.sum(
            self.vertices[:, 0] * following[:, 1]
            - self.vertices[:, 1] * following[:, 0]
        )
        return 0.5 * twice_area * (self.upper - self.lower)

    def parameters(self) -> dict[str, Array]:
        return {"vertices": self.vertices, "lower": self.lower, "upper": self.upper}


def rotation_matrix(vector: Float[Array, 3]) -> Float[Array, "3 3"]:
    """Rodrigues map with ordinary AD defined at zero rotation vector."""
    x, y, z = vector
    skew = jnp.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    theta2 = jnp.dot(vector, vector)
    small = theta2 < 1e-8
    safe2 = jnp.where(small, 1.0, theta2)
    theta = jnp.sqrt(safe2)
    a = jnp.where(small, 1 - theta2 / 6 + theta2**2 / 120, jnp.sin(theta) / theta)
    b = jnp.where(
        small, 0.5 - theta2 / 24 + theta2**2 / 720, (1 - jnp.cos(theta)) / safe2
    )
    return jnp.eye(3) + a * skew + b * (skew @ skew)


Measure = Callable[[Array, Array, Array, Array, Array, Array, float, float], Array]


@dataclass(frozen=True)
class Estimators:
    """Dynamic polygon, q, gap, pivot, and common placement; fixed world grid."""

    stratum: Measure
    sigmoid: Measure
    smoothing: Measure
    classify: Callable[[Array, Array, Array, Array, Array, Array], Array]
    resolution: int
    domain: tuple[tuple[float, ...], tuple[float, ...]]


def estimators(resolution: int) -> Estimators:
    """Use normalized world coordinates and a fixed [-3,3]^3 sample domain.

    q=(v_x,v_y,v_z,omega_x,omega_y,omega_z) is a finite pose, not a derivative.
    vertices, gap and pivot are local lengths. Common placement is a proper
    rotation plus translation; units can be changed externally without a
    second discretization. Stock is [-2,2]^2 x [-1,0], moving prism height 1.
    """
    if resolution < 2:
        raise ValueError("Resolution must be at least two")
    domain = ((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0))

    lo, hi = map(jnp.asarray, domain)
    grid, cell = _make_grid_nd(lo, hi, resolution)
    width = cell ** (1 / 3)

    def pair(
        q: Array,
        vertices: Array,
        gap: Array,
        pivot: Array,
        common_rotation: Array,
        common_shift: Array,
    ) -> tuple[Primitive, Primitive]:
        stock = Box(jnp.array([0.0, 0.0, -0.5]), jnp.array([2.0, 2.0, 0.5]))

        rotation = rotation_matrix(q[3:])
        shift = pivot - rotation @ pivot + q[:3]
        prism = ConvexPrism(vertices, gap, gap + 1.0)
        return (
            RigidPrimitive(stock, common_rotation, common_shift),
            RigidPrimitive(
                prism,
                common_rotation @ rotation,
                common_shift + common_rotation @ shift,
            ),
        )

    def stratum(
        q: Array,
        vertices: Array,
        gap: Array,
        pivot: Array,
        common_rotation: Array,
        common_shift: Array,
        k: float,
        beta: float,
    ) -> Array:
        a, b = pair(q, vertices, gap, pivot, common_rotation, common_shift)
        return _boolean_measure_with_dispatch(
            a, b, "intersect", resolution, lo + common_shift, hi + common_shift
        )

    def sigmoid(
        q: Array,
        vertices: Array,
        gap: Array,
        pivot: Array,
        common_rotation: Array,
        common_shift: Array,
        k: float,
        beta: float,
    ) -> Array:
        a, b = pair(q, vertices, gap, pivot, common_rotation, common_shift)
        sdf = jnp.maximum(a.sdf(grid + common_shift), b.sdf(grid + common_shift))
        return jnp.sum(jax.nn.sigmoid(-sdf / (width * beta))) * cell

    def smoothing(
        q: Array,
        vertices: Array,
        gap: Array,
        pivot: Array,
        common_rotation: Array,
        common_shift: Array,
        k: float,
        beta: float,
    ) -> Array:
        a, b = pair(q, vertices, gap, pivot, common_rotation, common_shift)
        smooth_sdf = (
            width
            * k
            * jnp.logaddexp(
                a.sdf(grid + common_shift) / (width * k),
                b.sdf(grid + common_shift) / (width * k),
            )
        )
        return jnp.sum(jax.nn.sigmoid(-smooth_sdf / (width * beta))) * cell

    def classify(
        q: Array,
        vertices: Array,
        gap: Array,
        pivot: Array,
        common_rotation: Array,
        common_shift: Array,
    ) -> Array:
        return _detect_stratum_generic(
            *pair(q, vertices, gap, pivot, common_rotation, common_shift),
            grid + common_shift,
        )

    return Estimators(stratum, sigmoid, smoothing, classify, resolution, domain)


def polygon_moments(vertices: np.ndarray) -> tuple[float, np.ndarray]:
    """Shoelace area/centroid for baseline ablations only, not the oracle."""
    following = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * following[:, 1] - vertices[:, 1] * following[:, 0]
    area = float(cross.sum() / 2)
    centroid = ((vertices + following) * cross[:, None]).sum(axis=0) / (6 * area)
    return area, centroid


def ablations(
    vertices: np.ndarray, direction: np.ndarray, pivot: np.ndarray
) -> dict[str, float]:
    area, centroid = polygon_moments(vertices)
    velocity, omega = direction[:3], direction[3:]
    speed_centroid = -float(
        (velocity + np.cross(omega, np.r_[centroid, 0.0] - pivot))[2]
    )
    return {
        "translation_only": area * max(-float(velocity[2]), 0.0),
        "centroid_positive_flux": area * max(speed_centroid, 0.0),
        "signed_spatial_flux": area * speed_centroid,
    }
