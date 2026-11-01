"""One fixed-normal design dimension and frozen bounded translation paths."""

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import cast

import numpy as np
from numpy.typing import NDArray

from brepax.experimental.polyhedral_parameter import AffinePolyhedralCell

VertexBody = tuple[NDArray[np.float64], ...]
AffineBodies = tuple[tuple[AffinePolyhedralCell, ...], tuple[AffinePolyhedralCell, ...]]


@dataclass(frozen=True)
class Family:
    name: str
    target: Fraction
    extent: float
    velocity: tuple[float, float, float]
    thetas: tuple[Fraction, ...]


WIDTH_PATH = tuple(
    map(Fraction, (-1 / 4, -1 / 8, -1 / 32, -1 / 64, 0, 1 / 64, 1 / 8, 1 / 4))
)
FAMILIES = (
    Family("ramp_full", Fraction(1, 2), 3.0, (1.0, 0.0, 0.0), WIDTH_PATH),
    Family(
        "ramp_birth",
        Fraction(3, 8),
        3.0,
        (1.0, 0.0, 0.0),
        tuple(map(Fraction, (-1 / 8, -1 / 32, 0, 1 / 32, 1 / 8))),
    ),
    Family("tetra_full", Fraction(1, 6), 4.0, (-1.0, 0.0, 0.0), WIDTH_PATH),
    Family(
        "independent",
        Fraction(3, 8),
        10.0,
        (1.0, 0.0, 0.0),
        tuple(map(Fraction, (-1 / 8, -1 / 16, -1 / 64, 0, 1 / 64, 1 / 16, 1 / 8))),
    ),
)


def box_vertices(
    lower: Sequence[Fraction | float | int], upper: Sequence[Fraction | float | int]
) -> NDArray[np.float64]:
    return np.array(list(product(*zip(lower, upper, strict=True))), dtype=np.float64)


def width(family: Family, theta: Fraction) -> Fraction:
    result = (Fraction(1, 2) if family.name == "ramp_birth" else Fraction(1)) + theta
    if result <= 0:
        raise ValueError("Width must be positive")
    return result


def vertices(family: Family, theta: Fraction) -> tuple[VertexBody, VertexBody]:
    if family.name.startswith("ramp"):
        w = width(family, theta)
        fixed = np.array(
            [(x, y, z) for z in (0, 1) for x, y in ((0, 0), (1, 0), (1, 1))],
            dtype=np.float64,
        )
        moving = box_vertices(
            (-Fraction(1, 4) - w, -Fraction(1, 4), -Fraction(1, 4)),
            (-Fraction(1, 4), Fraction(5, 4), Fraction(5, 4)),
        )
        return (fixed,), (moving,)
    if family.name == "tetra_full":
        width(family, theta)
        fixed = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=np.float64)
        return (fixed,), (box_vertices((2, -1, -1), (3 + theta, 2, 2)),)
    if abs(theta) >= Fraction(1, 4):
        raise ValueError("Independent-contact control requires |theta| < 1/4")
    return (
        box_vertices((0, 0, 0), (4, 1, 1)),
        box_vertices((0, 3, 0), (5, 5, 1)),
    ), (
        box_vertices(
            (-3 - theta, -Fraction(1, 4), -Fraction(1, 4)),
            (-1 - theta, Fraction(5, 4), Fraction(5, 4)),
        ),
        box_vertices(
            (-4 + theta, Fraction(11, 4), -Fraction(1, 4)),
            (-1 + theta, Fraction(21, 4), Fraction(5, 4)),
        ),
    )


def affine_cells(family: Family) -> AffineBodies:
    """Exact dyadic affine bases for theta=0 and the predeclared fixed normals."""
    base = vertices(family, Fraction(0))
    displaced = vertices(family, Fraction(1, 8))
    return cast(
        AffineBodies,
        tuple(
            tuple(
                AffinePolyhedralCell(points, ((shifted - points) * 8)[None, ...])
                for points, shifted in zip(body, changed, strict=True)
            )
            for body, changed in zip(base, displaced, strict=True)
        ),
    )
