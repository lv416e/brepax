"""Independent checks for the contact benchmark's numerical oracles."""

import math

import pytest

from benchmarks.contact_transition.references import (
    breakout_volume_reference,
    circle_rectangle_area,
    contact_reference,
    radius_finite_differences,
    single_edge_circle_reference,
)


@pytest.mark.parametrize(
    ("center", "half_extents", "expected"),
    [
        ((0.0, 0.0), (2.0, 3.0), math.pi),
        ((4.0, 0.0), (2.0, 3.0), 0.0),
        ((2.0, 0.0), (2.0, 3.0), math.pi / 2),
        ((2.0, 3.0), (2.0, 3.0), math.pi / 4),
    ],
)
def test_section_oracle_known_areas(center, half_extents, expected):
    result = circle_rectangle_area(center, 1.0, half_extents)
    assert result.value == pytest.approx(expected, abs=1e-12)
    assert result.quadrature_error < 1e-11


def test_section_oracle_scaling_and_rectangle_containment():
    for scale in (1e-3, 1.0, 1e3):
        result = circle_rectangle_area((0, 0), 5 * scale, (2 * scale, 3 * scale))
        assert result.value / scale**2 == pytest.approx(24.0, abs=1e-10)


def test_contact_orders_and_one_sided_face_derivatives():
    for kind, power in ("sphere", 2), ("cylinder", 1.5), ("box", 1):
        coarse = contact_reference(kind, 1e-4)
        fine = contact_reference(kind, 1e-6)
        assert coarse.value / fine.value == pytest.approx(100**power, rel=2e-5)
        disjoint = contact_reference(kind, -1e-4)
        assert disjoint.value == disjoint.derivative == 0.0
    face = contact_reference("box", 0.0)
    assert face.derivative is None
    assert (face.left_derivative, face.right_derivative) == (0.0, 4.0)
    for kind in ("sphere", "cylinder"):
        tangent = contact_reference(kind, 0.0)
        assert tangent.left_derivative == tangent.right_derivative == 0.0


def test_cylinder_quadrature_against_independent_lens_formula():
    separation = 1.7
    expected = 2 * (
        2 * math.acos(separation / 2) - separation * math.sqrt(4 - separation**2) / 2
    )
    result = contact_reference("cylinder", 2 - separation)
    assert result.value == pytest.approx(expected, abs=1e-12)
    assert result.quadrature_error < 1e-12


@pytest.mark.parametrize("radius", [1.0, 1.5, 1.500001, 1.51, 1.7, 1.8])
def test_breakout_volume_against_section_integral(radius):
    section = circle_rectangle_area((1.5, 0.2), radius, (3.0, 2.0))
    analytical = breakout_volume_reference(radius)
    assert analytical.value == pytest.approx(2 * (24 - section.value), abs=1e-10)
    assert analytical.derivative < 0


def test_radius_derivative_contact_continuity_and_step_convergence():
    center, half_extents, radius = (1.5, 0.2), (3.0, 2.0), 1.5
    exact = single_edge_circle_reference(radius, 1.5).derivative
    assert exact == pytest.approx(3 * math.pi)
    differences = radius_finite_differences(
        center, radius, half_extents, (1e-3, 1e-5, 1e-7)
    )
    errors = [abs(row.central - exact) for row in differences]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 2e-3
    near = single_edge_circle_reference(radius + 1e-10, 1.5).derivative
    assert near == pytest.approx(exact, abs=4e-5)


def test_opened_hole_radius_derivative_against_section_differences():
    radius = 1.7
    expected = -breakout_volume_reference(radius).derivative / 2
    differences = radius_finite_differences((1.5, 0.2), radius, (3.0, 2.0))
    errors = [abs(row.central - expected) for row in differences]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1e-8


def test_invalid_reference_domains_are_explicit():
    with pytest.raises(ValueError, match="Penetration"):
        contact_reference("sphere", 1.0)
    with pytest.raises(ValueError, match="positive"):
        circle_rectangle_area((0, 0), 0, (2, 3))
    with pytest.raises(ValueError, match=r"1\.8"):
        breakout_volume_reference(1.8001)
    with pytest.raises(ValueError, match="resolution"):
        radius_finite_differences((0, 0), 1, (2, 3), (1e-20,))
