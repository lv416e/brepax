"""Exact real-root counting and bounded isolation for polynomials of degree <= 3.

Rational Sturm sequences count distinct roots, including repeated roots after
square-free reduction. Refinement may fail its numerical acceptance condition;
that does not erase the exact root count. This host code adds no solver dependency.
"""

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise

Polynomial = tuple[Fraction, ...]


def trim(polynomial: Polynomial) -> Polynomial:
    while polynomial and polynomial[-1] == 0:
        polynomial = polynomial[:-1]
    return polynomial


def value(polynomial: Polynomial, position: Fraction) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(polynomial):
        result = result * position + coefficient
    return result


def derivative(polynomial: Polynomial) -> Polynomial:
    return trim(tuple(i * c for i, c in enumerate(polynomial) if i))


def _divide(first: Polynomial, second: Polynomial) -> tuple[Polynomial, Polynomial]:
    remainder = list(trim(first))
    quotient = [Fraction(0)] * max(0, len(first) - len(second) + 1)
    while remainder and len(remainder) >= len(second):
        power = len(remainder) - len(second)
        factor = remainder[-1] / second[-1]
        quotient[power] = factor
        for i, coefficient in enumerate(second):
            remainder[i + power] -= factor * coefficient
        remainder = list(trim(tuple(remainder)))
    return trim(tuple(quotient)), tuple(remainder)


def _sturm(polynomial: Polynomial) -> tuple[Polynomial, ...]:
    first, second = polynomial, derivative(polynomial)
    while second:
        _, remainder = _divide(first, second)
        first, second = second, remainder
    gcd = tuple(c / first[-1] for c in first)
    square_free, _ = _divide(polynomial, gcd)
    chain = [square_free, derivative(square_free)]
    while chain[-1]:
        _, remainder = _divide(chain[-2], chain[-1])
        if not remainder:
            break
        # A positive normalization preserves Sturm signs while limiting growth.
        chain.append(tuple(-c / abs(remainder[-1]) for c in remainder))
    return tuple(p for p in chain if p)


def _side_sign(polynomial: Polynomial, position: Fraction, side: int) -> int:
    order = 0
    while polynomial:
        at_position = value(polynomial, position)
        if at_position:
            sign = 1 if at_position > 0 else -1
            return -sign if side < 0 and order % 2 else sign
        polynomial = derivative(polynomial)
        order += 1
    return 0


def _variations(chain: tuple[Polynomial, ...], position: Fraction, side: int) -> int:
    signs = [_side_sign(p, position, side) for p in chain]
    return sum(a != b for a, b in pairwise(signs))


@dataclass(frozen=True)
class RootBracket:
    """An exact root or an interval containing ``count`` distinct interior roots.

    A resolved nonpoint bracket also has nonroot endpoints, so its closed
    interval isolates one root. Unresolved brackets may have root endpoints;
    those are counted separately and excluded from ``count``.
    """

    lower: Fraction
    upper: Fraction
    count: int
    resolved: bool


def isolate_roots(
    polynomial: Polynomial,
    lower: Fraction,
    upper: Fraction,
    accept: Callable[[Fraction, Fraction], bool],
    max_refinements: int,
) -> tuple[RootBracket, ...]:
    """Enumerate all distinct roots in a closed interval, retaining failures.

    ``accept`` controls numerical refinement, never existence. The depth limit
    applies per subdivision path; unresolved clusters retain their exact count.
    Identically zero polynomials are handled by the caller as solution intervals.
    """
    polynomial = trim(polynomial)
    if len(polynomial) <= 1:
        return ()
    if len(polynomial) > 4:
        raise ValueError("Only polynomial degrees up to three are supported")
    chain = _sturm(polynomial)
    roots = [
        RootBracket(x, x, 1, accept(x, x))
        for x in (lower, upper)
        if value(polynomial, x) == 0
    ]
    count = _variations(chain, lower, 1) - _variations(chain, upper, -1)
    stack = [(lower, upper, count, 0)]
    while stack:
        lo, hi, count, depth = stack.pop()
        if not count:
            continue
        if (
            count == 1
            and value(polynomial, lo) != 0
            and value(polynomial, hi) != 0
            and accept(lo, hi)
        ):
            roots.append(RootBracket(lo, hi, 1, True))
            continue
        if depth >= max_refinements:
            roots.append(RootBracket(lo, hi, count, False))
            continue
        midpoint = (lo + hi) / 2
        at_root = value(polynomial, midpoint) == 0
        if at_root:
            roots.append(RootBracket(midpoint, midpoint, 1, accept(midpoint, midpoint)))
        left = _variations(chain, lo, 1) - _variations(chain, midpoint, -1)
        right = _variations(chain, midpoint, 1) - _variations(chain, hi, -1)
        if left + right + int(at_root) != count:
            raise ArithmeticError("Inconsistent exact polynomial root count")
        stack.extend(
            ((midpoint, hi, right, depth + 1), (lo, midpoint, left, depth + 1))
        )
    return tuple(sorted(roots, key=lambda root: (root.lower, root.upper)))


def value_bounds(
    polynomial: Polynomial, lower: Fraction, upper: Fraction
) -> tuple[Fraction, Fraction]:
    """Enclose a polynomial on a rational interval with exact Horner arithmetic."""
    lo = hi = Fraction(0)
    for coefficient in reversed(polynomial):
        products = (lo * lower, lo * upper, hi * lower, hi * upper)
        lo, hi = min(products) + coefficient, max(products) + coefficient
    return lo, hi
