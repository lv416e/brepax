"""Tests for 3D shape visualization."""

from __future__ import annotations

from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from brepax.io.step import read_step  # noqa: E402
from brepax.viz.plot3d import plot_shape  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


class TestPlotShape:
    """Tests for plot_shape()."""

    @pytest.fixture(autouse=True)
    def _close_figures(self):
        """Close all matplotlib figures after each test."""
        yield
        plt.close("all")

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_plot_box_runs(self):
        shape = read_step(FIXTURES / "sample_box.step")
        plot_shape(shape)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_plot_cylinder_runs(self):
        shape = read_step(FIXTURES / "sample_cylinder.step")
        plot_shape(shape)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_plot_without_face_colors(self):
        shape = read_step(FIXTURES / "sample_box.step")
        plot_shape(shape, face_colors=False)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_plot_with_fine_tessellation(self):
        shape = read_step(FIXTURES / "sample_box.step")
        plot_shape(shape, linear_deflection=0.01)
