"""Tests for OCCT BRepGProp ground-truth quantities."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from brepax.brep.gprop import compute_gprop_ground_truth
from brepax.io import read_step

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture()
def box_gt() -> dict:
    """Ground truth for sample_box.step: 10 x 20 x 30 axis-aligned box at origin."""
    shape = read_step(FIXTURES / "sample_box.step")
    return compute_gprop_ground_truth(shape)


@pytest.fixture()
def sphere_gt() -> dict:
    """Ground truth for sample_sphere.step: radius 3 sphere at origin."""
    shape = read_step(FIXTURES / "sample_sphere.step")
    return compute_gprop_ground_truth(shape)


class TestAxisAlignedBox:
    """sample_box is a 10 x 20 x 30 rectangular solid at [0,0,0]."""

    def test_volume(self, box_gt) -> None:
        assert box_gt["volume"] == pytest.approx(10 * 20 * 30, rel=1e-9)

    def test_surface_area(self, box_gt) -> None:
        expected = 2 * (10 * 20 + 20 * 30 + 10 * 30)
        assert box_gt["surface_area"] == pytest.approx(expected, rel=1e-9)

    def test_center_of_mass(self, box_gt) -> None:
        expected = np.array([5.0, 10.0, 15.0])
        np.testing.assert_allclose(box_gt["center_of_mass"], expected, rtol=1e-9)

    def test_inertia_diagonal_about_com(self, box_gt) -> None:
        # I_ii = (V / 12) * (sum of squares of the other two dimensions)
        v = 10.0 * 20.0 * 30.0
        expected_diag = np.array(
            [
                v * (20**2 + 30**2) / 12.0,
                v * (10**2 + 30**2) / 12.0,
                v * (10**2 + 20**2) / 12.0,
            ]
        )
        diag = np.diag(box_gt["moment_of_inertia"])
        np.testing.assert_allclose(diag, expected_diag, rtol=1e-9)

    def test_inertia_off_diagonal_zero(self, box_gt) -> None:
        # Axis-aligned box about its own CoM has no products of inertia.
        matrix = box_gt["moment_of_inertia"]
        off_diag = matrix - np.diag(np.diag(matrix))
        assert np.max(np.abs(off_diag)) < 1e-6

    def test_inertia_symmetric(self, box_gt) -> None:
        matrix = box_gt["moment_of_inertia"]
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-9)


class TestSphere:
    """sample_sphere is a radius-3 sphere centered at origin."""

    def test_volume(self, sphere_gt) -> None:
        expected = (4.0 / 3.0) * math.pi * 3**3
        assert sphere_gt["volume"] == pytest.approx(expected, rel=1e-3)

    def test_surface_area(self, sphere_gt) -> None:
        expected = 4.0 * math.pi * 3**2
        assert sphere_gt["surface_area"] == pytest.approx(expected, rel=1e-3)

    def test_center_of_mass(self, sphere_gt) -> None:
        np.testing.assert_allclose(sphere_gt["center_of_mass"], np.zeros(3), atol=1e-3)

    def test_inertia_isotropic(self, sphere_gt) -> None:
        # Solid sphere: I = (2/5) * M * R^2, isotropic; M here is V.
        v = (4.0 / 3.0) * math.pi * 3**3
        expected = (2.0 / 5.0) * v * 3**2
        diag = np.diag(sphere_gt["moment_of_inertia"])
        np.testing.assert_allclose(diag, np.full(3, expected), rtol=1e-3)

    def test_inertia_off_diagonal_zero(self, sphere_gt) -> None:
        matrix = sphere_gt["moment_of_inertia"]
        off_diag = matrix - np.diag(np.diag(matrix))
        assert np.max(np.abs(off_diag)) < 1e-3


class TestReturnShape:
    """Structural invariants of the returned dict."""

    def test_required_keys(self, box_gt) -> None:
        assert set(box_gt.keys()) == {
            "volume",
            "surface_area",
            "center_of_mass",
            "moment_of_inertia",
        }

    def test_com_shape(self, box_gt) -> None:
        assert box_gt["center_of_mass"].shape == (3,)

    def test_inertia_shape(self, box_gt) -> None:
        assert box_gt["moment_of_inertia"].shape == (3, 3)

    def test_scalar_types(self, box_gt) -> None:
        assert isinstance(box_gt["volume"], float)
        assert isinstance(box_gt["surface_area"], float)
