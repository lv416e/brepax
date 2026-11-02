"""Exact bounds on parameter-dependent volume strips for bounded host search."""

from fractions import Fraction
from math import comb

from brepax.experimental._polyhedral_boundary import _Side
from brepax.experimental._polyhedral_parameter import parameter_derivative_polynomials

_Polynomial = dict[tuple[int, int], Fraction]


def _add(first: _Polynomial, second: _Polynomial) -> _Polynomial:
    result = dict(first)
    for key, coefficient in second.items():
        result[key] = result.get(key, Fraction(0)) + coefficient
    return {key: c for key, c in result.items() if c}


def _scale(polynomial: _Polynomial, factor: Fraction) -> _Polynomial:
    return {key: c * factor for key, c in polynomial.items() if c * factor}


def _multiply(first: _Polynomial, second: _Polynomial) -> _Polynomial:
    result: _Polynomial = {}
    for (i, j), a in first.items():
        for (k, power), b in second.items():
            key = i + k, j + power
            result[key] = result.get(key, Fraction(0)) + a * b
    return {key: c for key, c in result.items() if c}


def _strip_polynomial(
    side: _Side,
    index: int,
    lower: Fraction,
    upper: Fraction,
    coefficients: tuple[Fraction, ...],
) -> _Polynomial:
    """Map a moving event strip to the unit square, without dividing by its width."""
    geometry = side.geometry
    parameter = {(0, 0): lower - side.center, (1, 0): upper - lower}

    def affine(expression: tuple[Fraction, ...]) -> _Polynomial:
        return _add({(0, 0): expression[0]}, _scale(parameter, expression[1]))

    lo, hi = map(affine, geometry.knots[index : index + 2])
    step = _add(
        lo, _multiply(_add(hi, _scale(lo, Fraction(-1))), {(0, 1): Fraction(1)})
    )
    normalized = _scale(
        _add(step, _scale(affine(geometry.origins[index]), Fraction(-1))),
        1 / geometry.widths[index],
    )
    substitutions = ({(0, 0): Fraction(1)}, normalized, parameter)
    result: _Polynomial = {}
    for term, coefficient in zip(geometry.terms, coefficients, strict=True):
        current = {(0, 0): coefficient}
        for variable in term:
            current = _multiply(current, substitutions[variable])
        result = _add(result, current)
    return result


def _bernstein_bounds(polynomial: _Polynomial) -> tuple[Fraction, Fraction]:
    """Enclose the polynomial by its tensor Bernstein coefficients on [0,1]^2."""
    n = max((i for i, _ in polynomial), default=0)
    m = max((j for _, j in polynomial), default=0)
    coefficients = [
        sum(
            (
                c
                * Fraction(comb(i, k), comb(n, k))
                * Fraction(comb(j, power), comb(m, power))
                for (k, power), c in polynomial.items()
                if k <= i and power <= j
            ),
            Fraction(0),
        )
        for i in range(n + 1)
        for j in range(m + 1)
    ]
    return min(coefficients), max(coefficients)


def interval_volume_bound(
    side: _Side,
    lower: Fraction,
    upper: Fraction,
    midpoint_maximum: Fraction | None,
) -> Fraction:
    """Bound max V over a design interval contained in one chart closure.

    Raw Bernstein bounds alone need not converge as only theta is subdivided.
    A physical partial-theta bound over every strip also bounds the Lipschitz
    constant of max_t V: the finite piecewise-polynomial path is continuous
    across moving translation events. The caller must separately account for
    parameter endpoints not certified by this chart or an adjoining boundary.
    """
    geometry = side.geometry
    maximum = Fraction(0)
    lipschitz = Fraction(0)
    for index, coefficients in enumerate(geometry.coefficients):
        polynomial = _strip_polynomial(side, index, lower, upper, coefficients)
        maximum = max(maximum, _bernstein_bounds(polynomial)[1])
        derivative = parameter_derivative_polynomials(
            coefficients,
            geometry.terms,
            geometry.origins[index],
            geometry.widths[index],
        )[1]
        bounds = _bernstein_bounds(
            _strip_polynomial(side, index, lower, upper, derivative)
        )
        lipschitz = max(lipschitz, abs(bounds[0]), abs(bounds[1]))
    if midpoint_maximum is not None:
        maximum = min(maximum, midpoint_maximum + lipschitz * (upper - lower) / 2)
    return maximum
