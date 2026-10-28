"""Analytical checks for the independent face-contact section reference."""

import numpy as np
import pytest

from benchmarks.face_contact.references import (
    directional_reference,
    positive_affine_reference,
)

SQUARE = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


@pytest.mark.parametrize(
    ("velocity", "omega", "expected"),
    [
        ((0, 0, -1), (0, 0, 0), 4.0),
        ((0, 0, 1), (0, 0, 0), 0.0),
        ((0, 0, 0), (1, 0, 0), 1.0),
        ((0, 0, 0), (-1, 0, 0), 1.0),
        ((1, 2, 0), (0, 0, 3), 0.0),
        ((0, 0, 0), (-1, 1, 0), 4 / 3),
    ],
)
def test_exact_directional_limits(velocity, omega, expected):
    actual = directional_reference(SQUARE, velocity, omega, (0, 0, 0))
    assert actual == pytest.approx(expected, rel=2e-14, abs=1e-15)


@pytest.mark.parametrize("epsilon", [1e-3, 1e-8, 1e-14])
def test_tiny_triangle_caps_without_absolute_cutoff(epsilon):
    # f=epsilon+x+y cuts a right triangle of leg epsilon from the origin.
    one = positive_affine_reference([[-1, 0], [0, -1], [0, 0]], (epsilon, 1, 1))
    assert one == pytest.approx(epsilon**3 / 6, rel=3e-14, abs=0)
    # Along the vertical edge f=epsilon; the triangular strip integral is
    # epsilon²/2 - epsilon³/6 when the sloped edge is x=y-1.
    two = positive_affine_reference([[-1, 0], [0, 0], [0, 1]], (epsilon, 1, 0))
    assert two == pytest.approx(epsilon**2 / 2 - epsilon**3 / 6, rel=3e-14, abs=0)


def test_pivot_orientation_and_dimensional_scaling():
    velocity, omega, pivot = (
        np.array([1.0, 2.0, -0.25]),
        np.array([0.5, 0.7, 0.2]),
        np.zeros(3),
    )
    base = directional_reference(SQUARE, velocity, omega, pivot)
    shift = np.array([3.0, -2.0, 4.0])
    moved_pivot = directional_reference(
        SQUARE[::-1], velocity + np.cross(omega, shift), omega, shift
    )
    assert moved_pivot == pytest.approx(base, rel=2e-14)
    for scale in (1e-6, 1e6):
        actual = directional_reference(
            SQUARE * scale, velocity * scale, omega, pivot * scale
        )
        assert actual / scale**3 == pytest.approx(base, rel=2e-14)


def test_zero_endpoints_degeneracy_and_invalid_order():
    assert positive_affine_reference(SQUARE, (-2, 1, 1)) == 0
    assert positive_affine_reference([(0, 0), (1, 0), (2, 0)], (1, 1, 1)) == 0
    with pytest.raises(ValueError, match="convex"):
        positive_affine_reference(SQUARE[[0, 2, 1, 3]], (1, 0, 0))
