"""Tests for STEP file reading."""

from __future__ import annotations

from pathlib import Path

import pytest

from brepax.io.step import read_step

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


class TestReadStep:
    """Tests for read_step()."""

    def test_read_box(self):
        shape = read_step(FIXTURES / "sample_box.step")
        assert not shape.IsNull()

    def test_read_cylinder(self):
        shape = read_step(FIXTURES / "sample_cylinder.step")
        assert not shape.IsNull()

    def test_accepts_string_path(self):
        shape = read_step(str(FIXTURES / "sample_box.step"))
        assert not shape.IsNull()

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="STEP file not found"):
            read_step(tmp_path / "nonexistent.step")

    def test_invalid_step_content(self, tmp_path):
        bad = tmp_path / "bad.step"
        bad.write_text("this is not a valid STEP file")
        with pytest.raises(ValueError, match="Failed to read STEP file"):
            read_step(bad)
