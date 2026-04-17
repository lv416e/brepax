"""Tests for face-to-primitive conversion."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from brepax.brep.convert import faces_to_primitives
from brepax.io.step import read_step
from brepax.primitives import BSplineSurface, Cone, Cylinder, Plane, Sphere, Torus

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


class TestFaceToPrimitive:
    """Tests for face_to_primitive() and faces_to_primitives()."""

    def test_box_yields_planes(self) -> None:
        shape = read_step(FIXTURES / "sample_box.step")
        prims = faces_to_primitives(shape)
        assert len(prims) == 6
        assert all(isinstance(p, Plane) for p in prims)

    def test_cylinder_yields_mixed(self) -> None:
        shape = read_step(FIXTURES / "sample_cylinder.step")
        prims = faces_to_primitives(shape)
        assert len(prims) == 3
        planes = [p for p in prims if isinstance(p, Plane)]
        cylinders = [p for p in prims if isinstance(p, Cylinder)]
        assert len(planes) == 2
        assert len(cylinders) == 1

    def test_sphere_yields_sphere(self) -> None:
        shape = read_step(FIXTURES / "sample_sphere.step")
        prims = faces_to_primitives(shape)
        assert len(prims) == 1
        assert isinstance(prims[0], Sphere)

    def test_cone_yields_cone(self) -> None:
        shape = read_step(FIXTURES / "sample_cone.step")
        prims = faces_to_primitives(shape)
        assert len(prims) == 3
        cones = [p for p in prims if isinstance(p, Cone)]
        planes = [p for p in prims if isinstance(p, Plane)]
        assert len(cones) == 1
        assert len(planes) == 2

    def test_torus_yields_torus(self) -> None:
        shape = read_step(FIXTURES / "sample_torus.step")
        prims = faces_to_primitives(shape)
        assert len(prims) == 1
        assert isinstance(prims[0], Torus)

    def test_primitive_parameters_match_fixture(self) -> None:
        """Box is 10x20x30 at origin; verify plane normals and offsets."""
        shape = read_step(FIXTURES / "sample_box.step")
        prims = faces_to_primitives(shape)

        offsets = sorted(float(p.offset) for p in prims if isinstance(p, Plane))
        # Expect offsets: 0, 0, 0, 10, 20, 30
        assert offsets == pytest.approx([0.0, 0.0, 0.0, 10.0, 20.0, 30.0], abs=1e-6)

        normals = [p.normal for p in prims if isinstance(p, Plane)]
        # Each normal should be a unit vector
        for n in normals:
            assert float(jnp.linalg.norm(n)) == pytest.approx(1.0, abs=1e-10)

    def test_sphere_center_and_radius(self) -> None:
        """Sphere at origin with radius 3.0."""
        shape = read_step(FIXTURES / "sample_sphere.step")
        prims = faces_to_primitives(shape)
        sph = prims[0]
        assert isinstance(sph, Sphere)
        assert float(sph.radius) == pytest.approx(3.0, abs=1e-6)
        assert jnp.allclose(sph.center, jnp.zeros(3), atol=1e-6)

    def test_torus_radii(self) -> None:
        """Torus with major radius 5.0, minor radius 1.5."""
        shape = read_step(FIXTURES / "sample_torus.step")
        prims = faces_to_primitives(shape)
        tor = prims[0]
        assert isinstance(tor, Torus)
        assert float(tor.major_radius) == pytest.approx(5.0, abs=1e-6)
        assert float(tor.minor_radius) == pytest.approx(1.5, abs=1e-6)
        assert jnp.allclose(tor.center, jnp.zeros(3), atol=1e-6)

    def test_cylinder_radius(self) -> None:
        """Cylinder fixture has radius 5.0."""
        shape = read_step(FIXTURES / "sample_cylinder.step")
        prims = faces_to_primitives(shape)
        cyls = [p for p in prims if isinstance(p, Cylinder)]
        assert len(cyls) == 1
        assert float(cyls[0].radius) == pytest.approx(5.0, abs=1e-6)

    def test_sdf_evaluates_on_converted_sphere(self) -> None:
        """Converted sphere SDF returns correct signed distance."""
        shape = read_step(FIXTURES / "sample_sphere.step")
        prims = faces_to_primitives(shape)
        sph = prims[0]
        assert isinstance(sph, Sphere)
        # Point at distance 5 from origin: SDF should be 5 - 3 = 2
        pt = jnp.array([5.0, 0.0, 0.0])
        assert float(sph.sdf(pt)) == pytest.approx(2.0, abs=1e-6)
        # Point at origin: SDF should be -3
        origin = jnp.array([0.0, 0.0, 0.0])
        assert float(sph.sdf(origin)) == pytest.approx(-3.0, abs=1e-6)

    def test_bspline_face_converted(self) -> None:
        """NURBS STEP face converts to BSplineSurface primitive."""
        shape = read_step(FIXTURES / "nurbs_saddle.step")
        prims = faces_to_primitives(shape)
        assert len(prims) == 1
        assert isinstance(prims[0], BSplineSurface)

    def test_bspline_control_points_shape(self) -> None:
        """Converted B-spline has correct control point grid shape."""
        shape = read_step(FIXTURES / "nurbs_saddle.step")
        prims = faces_to_primitives(shape)
        surf = prims[0]
        assert isinstance(surf, BSplineSurface)
        assert surf.control_points.shape == (4, 4, 3)
        assert surf.degree_u == 3
        assert surf.degree_v == 3

    def test_bspline_knots_clamped(self) -> None:
        """Converted knot vectors are in clamped repeated form."""
        shape = read_step(FIXTURES / "nurbs_saddle.step")
        prims = faces_to_primitives(shape)
        surf = prims[0]
        assert isinstance(surf, BSplineSurface)
        expected = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        assert jnp.allclose(surf.knots_u, expected)
        assert jnp.allclose(surf.knots_v, expected)

    def test_bspline_sdf_evaluates(self) -> None:
        """Converted B-spline SDF returns correct distance."""
        shape = read_step(FIXTURES / "nurbs_saddle.step")
        prims = faces_to_primitives(shape)
        surf = prims[0]
        assert isinstance(surf, BSplineSurface)
        dist = surf.sdf(jnp.array([0.5, 0.5, 1.0]))
        assert jnp.isfinite(dist)
        assert jnp.isclose(jnp.abs(dist), 1.0, rtol=0.05)
