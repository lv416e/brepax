"""Independent section integrals and exact target inventories for the fixtures."""

from fractions import Fraction
from math import sqrt
from typing import TypedDict, cast

from .event_boundary_fixtures import Family, width


class ReferenceInventory(TypedDict):
    """Independent target and event inventory for a frozen family and dimension."""

    maximum: Fraction
    stock_volume: Fraction
    roots: list[float]
    intervals: list[tuple[float, float]]
    exact_reachable: bool
    tolerance_reachable: bool
    knots: list[Fraction]


def _ramp_primitive(x: Fraction) -> Fraction:
    return Fraction(0) if x <= 0 else Fraction(1, 2) if x >= 1 else x * x / 2


def _tetra_tail(x: Fraction) -> Fraction:
    return Fraction(1, 6) if x <= 0 else Fraction(0) if x >= 1 else (1 - x) ** 3 / 6


def _overlap(a: int, b: int, c: Fraction, d: Fraction) -> Fraction | int:
    upper = cast(Fraction | int, min(b, d))
    lower = cast(Fraction | int, max(a, c))
    return max(Fraction(0), upper - lower)


def reference_volume(
    family: Family, theta: Fraction, step: Fraction | float | int
) -> Fraction | int:
    """Exact independent cross-section integrals, never used by selection."""
    t = Fraction(step)
    if family.name.startswith("ramp"):
        w = width(family, theta)
        x = t - Fraction(1, 4)
        return _ramp_primitive(x) - _ramp_primitive(x - w)
    if family.name == "tetra_full":
        width(family, theta)
        return _tetra_tail(2 - t) - _tetra_tail(3 + theta - t)
    return _overlap(0, 4, t - 3 - theta, t - 1 - theta) + 2 * _overlap(
        0, 5, t - 4 + theta, t - 1 + theta
    )


def reference(family: Family, theta: Fraction) -> ReferenceInventory:
    """Independent maximum and complete exact target inventory for frozen q."""
    q = family.target
    roots: list[float] = []
    intervals: list[tuple[float, float]] = []
    if family.name.startswith("ramp"):
        w = width(family, theta)
        maximum = Fraction(1, 2) if w >= 1 else w - w * w / 2
        stock = Fraction(1, 2)
        knots = sorted(
            {
                Fraction(0),
                Fraction(1, 4),
                Fraction(1, 4) + w,
                Fraction(5, 4),
                Fraction(5, 4) + w,
                Fraction(family.extent),
            }
        )
        if q <= maximum:
            if q == Fraction(1, 2) and w >= 1:
                if w == 1:
                    roots = [1.25]
                else:
                    intervals = [(1.25, float(Fraction(1, 4) + w))]
            else:
                entry = Fraction(1, 4) + (
                    sqrt(float(2 * q)) if q <= w * w / 2 else q / w + w / 2
                )
                exit_step = float(Fraction(1, 4) + w) + sqrt(float(1 - 2 * q))
                roots = [float(entry)] if q == maximum else [float(entry), exit_step]
    elif family.name == "tetra_full":
        width(family, theta)
        maximum = Fraction(1, 6) if theta >= 0 else (1 + theta**3) / 6
        stock = Fraction(1, 6)
        knots = sorted(
            {Fraction(0), Fraction(1), Fraction(2), 2 + theta, 3 + theta, Fraction(4)}
        )
        if theta == 0:
            roots = [2.0]
        elif theta > 0:
            intervals = [(2.0, float(2 + theta))]
    else:
        maximum, stock = Fraction(8), Fraction(14)
        knots = sorted(
            {
                Fraction(0),
                1 + theta,
                3 + theta,
                5 + theta,
                7 + theta,
                1 - theta,
                4 - theta,
                6 - theta,
                9 - theta,
                Fraction(10),
            }
        )
        entry = (
            Fraction(11, 8) + theta
            if theta < -Fraction(3, 16)
            else Fraction(19, 16) - theta
            if theta > Fraction(3, 32)
            else Fraction(9, 8) - theta / 3
        )
        roots = [float(entry), float(Fraction(141, 16) - theta)]
    return {
        "maximum": maximum,
        "stock_volume": stock,
        "roots": roots,
        "intervals": intervals,
        "exact_reachable": bool(roots or intervals),
        "tolerance_reachable": maximum >= q - q / 1000,
        "knots": knots,
    }
