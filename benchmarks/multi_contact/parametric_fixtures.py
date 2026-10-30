"""Fixed finite-edit cases with independent prism construction geometry."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]
Family = Literal["ramp", "tent", "split", "orthant"]
SecondParameter = Literal["stock_height", "slab_width"]


@dataclass(frozen=True)
class Prism:
    """Convex CCW XY footprint and finite z interval, in working length units."""

    footprint: tuple[Vector2, ...]
    lower_z: float
    upper_z: float

    def vertices(self) -> NDArray[np.float64]:
        return np.array(
            [
                (x, y, z)
                for z in (self.lower_z, self.upper_z)
                for x, y in self.footprint
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class PrismBodies:
    """Explicit disjoint-interior cells and one prescribed translation path."""

    fixed: tuple[Prism, ...]
    moving: tuple[Prism, ...]
    velocity: Vector3
    max_step: float


@dataclass(frozen=True)
class VertexFamily:
    """Affine vertices; parameters are dimensionless and displacements lengths."""

    vertices: NDArray[np.float64]
    displacements: NDArray[np.float64]


@dataclass(frozen=True)
class ParametricFixture:
    """A frozen analytic family, including the path extent used by every policy.

    The ramp/tent design parameters are ``(delta_gap/a, delta_dimension/a)``.
    The second dimension is stock extrusion height by default, or slab width
    for the separately identified sensitivity control. Target volumes remain
    fixed at the reference design when the parameters vary.
    """

    name: str
    family: Family
    a: float
    width: float
    gap: float
    height: float = 1.0
    separation: float = 0.0
    codimension: int = 1

    def __post_init__(self) -> None:
        if (
            not np.isfinite(
                [self.a, self.width, self.gap, self.height, self.separation]
            ).all()
            or min(self.a, self.width, self.height) <= 0
            or self.gap < 0
        ):
            raise ValueError("Expected finite positive fixture dimensions")
        if self.family in ("ramp", "tent", "split") and self.width >= self.a:
            raise ValueError(
                "This reference family requires slab width below stock extent"
            )
        if self.family == "split" and self.separation <= self.a + self.width:
            raise ValueError("Split-stock overlap windows must be separated")
        if self.family == "orthant" and self.codimension not in (1, 2, 3):
            raise ValueError("Orthant contact order must be one, two or three")

    @property
    def max_step(self) -> float:
        if self.family == "orthant":
            return 2.5
        extent = (
            self.separation + self.a
            if self.family == "split"
            else self.a * (2 if self.family == "tent" else 1)
        )
        return self.gap + self.width + extent + extent / 8

    def bodies(
        self,
        theta: tuple[float, float] = (0.0, 0.0),
        *,
        second: SecondParameter = "stock_height",
    ) -> PrismBodies:
        """Rebuild physical geometry from the design parameters.

        Parameterized families are restricted to the positive-gap ramp/tent
        fixtures. Construction retains a fixed oversized cutter in y/z;
        changing stock height therefore changes material, not the cutter.
        """
        if second not in ("stock_height", "slab_width"):
            raise ValueError("Unknown second design parameter")
        parameters = np.asarray(theta, dtype=np.float64)
        if parameters.shape != (2,) or not np.isfinite(parameters).all():
            raise ValueError("Expected two finite dimensionless parameters")
        if self.family not in ("ramp", "tent") and np.any(parameters != 0):
            raise ValueError("Only ramp and tent fixtures have design parameters")
        gap = self.gap + self.a * theta[0]
        width = self.width + (self.a * theta[1] if second == "slab_width" else 0)
        height = self.height + (self.a * theta[1] if second == "stock_height" else 0)
        if gap < 0 or width <= 0 or height <= 0:
            raise ValueError("Expected nonnegative gap and positive dimensions")
        if self.family in ("ramp", "tent"):
            extent = self.a * (2 if self.family == "tent" else 1)
            apex = self.a
            triangle = ((0.0, 0.0), (extent, 0.0), (apex, self.a))
            margin = self.a / 2
            if height >= self.height + margin:
                raise ValueError("Perturbed stock exceeds the supplied cutter height")
            fixed: tuple[Prism, ...] = (Prism(triangle, 0.0, height),)
            moving = (
                rectangle(
                    -gap - width,
                    -gap,
                    -margin,
                    self.a + margin,
                    -margin,
                    self.height + margin,
                ),
            )
            return PrismBodies(fixed, moving, (1.0, 0.0, 0.0), self.max_step)
        if self.family == "split":
            fixed = tuple(
                rectangle(start, start + self.a, 0, 1, 0, 1)
                for start in (0.0, self.separation)
            )
            moving = (rectangle(-gap - width, -gap, 0, 1, 0, 1),)
            return PrismBodies(fixed, moving, (1.0, 0.0, 0.0), self.max_step)
        length = self.a
        low = [-length if i < self.codimension else 0.0 for i in range(3)]
        high = [0.0 if i < self.codimension else length for i in range(3)]
        velocity = tuple(length if i < self.codimension else 0.0 for i in range(3))
        return PrismBodies(
            (rectangle(0, length, 0, length, 0, length),),
            (rectangle(low[0], high[0], low[1], high[1], low[2], high[2]),),
            (velocity[0], velocity[1], velocity[2]),
            self.max_step,
        )

    def vertex_families(
        self, *, second: SecondParameter = "stock_height"
    ) -> tuple[tuple[VertexFamily, ...], tuple[VertexFamily, ...]]:
        """Return the exact affine vertex maps for the declared design family."""
        if self.family not in ("ramp", "tent"):
            raise ValueError("Only ramp and tent fixtures have design parameters")
        body = self.bodies(second=second)
        fixed = []
        for prism in body.fixed:
            points = prism.vertices()
            displacement = np.zeros((2, *points.shape), dtype=np.float64)
            if second == "stock_height":
                displacement[1, len(prism.footprint) :, 2] = self.a
            fixed.append(VertexFamily(points, displacement))
        moving = []
        for prism in body.moving:
            points = prism.vertices()
            displacement = np.zeros((2, *points.shape), dtype=np.float64)
            displacement[0, :, 0] = -self.a
            if second == "slab_width":
                displacement[1, points[:, 0] == np.min(points[:, 0]), 0] = -self.a
            moving.append(VertexFamily(points, displacement))
        return tuple(fixed), tuple(moving)


def rectangle(
    xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float
) -> Prism:
    return Prism(((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)), zmin, zmax)


def parametric_fixtures() -> tuple[ParametricFixture, ...]:
    """Twelve predeclared groups; targets and placements are not new groups."""
    ramps = (
        (1, 1 / 4, 1 / 8, 1),
        (5 / 4, 3 / 8, 3 / 16, 3 / 4),
        (3 / 4, 1 / 16, 1 / 32, 5 / 4),
    )
    tents = (
        (1, 1 / 2, 1 / 8, 1),
        (3 / 4, 1 / 4, 3 / 16, 5 / 4),
        (5 / 4, 3 / 8, 1 / 16, 3 / 4),
    )
    splits = (
        (1 / 8, 3 / 4, 1 / 16, 1 / 32),
        (3 / 16, 1, 1 / 8, 1 / 16),
        (1 / 16, 1 / 2, 1 / 32, 3 / 32),
    )
    families: tuple[
        tuple[Family, tuple[tuple[float, float, float, float], ...]], ...
    ] = (("ramp", ramps), ("tent", tents))
    result = [
        ParametricFixture(f"{kind}_{i}", kind, a, w, g, h)
        for kind, values in families
        for i, (a, w, g, h) in enumerate(values)
    ]
    result += [
        ParametricFixture(f"split_{i}", "split", d, w, g, separation=b)
        for i, (d, b, w, g) in enumerate(splits)
    ]
    result += [
        ParametricFixture(f"orthant_{k}", "orthant", length, length, 0, codimension=k)
        for k, length in enumerate((3 / 4, 5 / 4, 1 / 2), start=1)
    ]
    return tuple(result)
