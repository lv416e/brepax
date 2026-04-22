"""Foot-of-perpendicular on analytical primitives, checked against OCCT."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brepax._occt.backend import (
    BRepAdaptor_Surface,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeVertex,
    BRepExtrema_DistShapeShape,
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeSphere,
    BRepPrimAPI_MakeTorus,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    TopAbs_FACE,
    TopExp_Explorer,
    TopoDS,
    gp_Ax2,
    gp_Dir,
    gp_Pln,
    gp_Pnt,
)
from brepax.primitives.foot import (
    foot_on_cone,
    foot_on_cylinder,
    foot_on_plane,
    foot_on_sphere,
    foot_on_torus,
)

ABS_TOL = 1e-5


def _first_face_of_type(shape, surf_type) -> object:
    """Return the first face of the given GeomAbs surface type."""
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        adaptor = BRepAdaptor_Surface(face)
        if adaptor.GetType() == surf_type:
            return face
        exp.Next()
    raise AssertionError(f"no face of type {surf_type} found")


def _occt_foot_on_face(face, query: np.ndarray) -> np.ndarray:
    """Use OCCT BRepExtrema_DistShapeShape to get the foot-of-perpendicular."""
    vx, vy, vz = (float(c) for c in query)
    vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(vx, vy, vz)).Vertex()
    dss = BRepExtrema_DistShapeShape(face, vertex)
    dss.Perform()
    assert dss.IsDone()
    pnt = dss.PointOnShape1(1)
    return np.array([pnt.X(), pnt.Y(), pnt.Z()], dtype=np.float64)


class TestPlaneFoot:
    """Closest point on z=1 plane."""

    def _make_face(self) -> object:
        pln = gp_Pln(gp_Pnt(0, 0, 1), gp_Dir(0, 0, 1))
        return BRepBuilderAPI_MakeFace(pln, -10, 10, -10, 10).Face()

    @pytest.mark.parametrize(
        "query",
        [
            np.array([0.3, 0.4, 2.5]),
            np.array([-1.0, 2.0, 0.2]),
            np.array([0.0, 0.0, 1.0]),  # on the plane
        ],
    )
    def test_foot_matches_occt(self, query: np.ndarray) -> None:
        foot = foot_on_plane(
            jnp.asarray(query, dtype=jnp.float64),
            jnp.array([0.0, 0.0, 1.0]),
            jnp.asarray(1.0),
        )
        expected = _occt_foot_on_face(self._make_face(), query)
        np.testing.assert_allclose(np.asarray(foot), expected, atol=ABS_TOL)


class TestSphereFoot:
    """Closest point on sphere radius 3 at origin."""

    def _face(self) -> object:
        solid = BRepPrimAPI_MakeSphere(3.0).Shape()
        return _first_face_of_type(solid, GeomAbs_Sphere)

    @pytest.mark.parametrize(
        "query",
        [
            np.array([5.0, 0.0, 0.0]),  # outside +x
            np.array([0.0, 1.0, 0.0]),  # inside
            np.array([-2.0, 2.0, 1.5]),  # generic
        ],
    )
    def test_foot_matches_occt(self, query: np.ndarray) -> None:
        foot = foot_on_sphere(
            jnp.asarray(query, dtype=jnp.float64),
            jnp.array([0.0, 0.0, 0.0]),
            jnp.asarray(3.0),
        )
        expected = _occt_foot_on_face(self._face(), query)
        np.testing.assert_allclose(np.asarray(foot), expected, atol=ABS_TOL)


class TestCylinderFoot:
    """Closest point on cylinder radius 2, axis +z, height 10 starting at origin."""

    def _face(self) -> object:
        solid = BRepPrimAPI_MakeCylinder(2.0, 10.0).Shape()
        return _first_face_of_type(solid, GeomAbs_Cylinder)

    @pytest.mark.parametrize(
        "query",
        [
            np.array([3.0, 0.0, 5.0]),  # outside +x at mid-height
            np.array([0.5, 0.5, 3.0]),  # inside
            np.array([-4.0, 1.0, 7.0]),
        ],
    )
    def test_foot_matches_occt(self, query: np.ndarray) -> None:
        foot = foot_on_cylinder(
            jnp.asarray(query, dtype=jnp.float64),
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 1.0]),
            jnp.asarray(2.0),
        )
        expected = _occt_foot_on_face(self._face(), query)
        np.testing.assert_allclose(np.asarray(foot), expected, atol=ABS_TOL)


class TestConeFoot:
    """Closest point on cone, apex at origin, axis +z, half-angle atan(r/h)."""

    def _face(self) -> object:
        # OCCT MakeCone(R1, R2, H): bottom radius R1, top radius R2, height H
        # Choose R1=3, R2=0, H=9 → half-angle atan(3/9) = atan(1/3)
        solid = BRepPrimAPI_MakeCone(3.0, 0.0, 9.0).Shape()
        return _first_face_of_type(solid, GeomAbs_Cone)

    @pytest.mark.parametrize(
        "query",
        [
            np.array([4.0, 0.0, 3.0]),  # outside the cone
            np.array([0.5, 0.0, 6.0]),  # inside
            np.array([2.0, 1.0, 4.5]),
        ],
    )
    def test_foot_matches_occt(self, query: np.ndarray) -> None:
        # OCCT MakeCone: apex at (0, 0, 9), axis pointing from base to apex is -z
        # but OCCT's frame has apex at top so axis in BRepAX terms is -z from apex.
        # We express in BRepAX convention: apex=(0,0,9), axis=(0,0,-1), half_angle.
        apex = jnp.array([0.0, 0.0, 9.0])
        axis = jnp.array([0.0, 0.0, -1.0])
        half_angle = jnp.asarray(np.arctan(3.0 / 9.0))
        foot = foot_on_cone(
            jnp.asarray(query, dtype=jnp.float64),
            apex,
            axis,
            half_angle,
        )
        expected = _occt_foot_on_face(self._face(), query)
        np.testing.assert_allclose(np.asarray(foot), expected, atol=ABS_TOL)


class TestTorusFoot:
    """Closest point on torus, major=5, minor=1, axis +z, center at origin."""

    def _face(self) -> object:
        ax = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        solid = BRepPrimAPI_MakeTorus(ax, 5.0, 1.0).Shape()
        return _first_face_of_type(solid, GeomAbs_Torus)

    @pytest.mark.parametrize(
        "query",
        [
            np.array([7.0, 0.0, 0.0]),  # outer equator
            np.array([5.0, 0.0, 2.0]),  # above tube center
            np.array([3.0, 0.0, 0.0]),  # inner equator (inside torus)
            np.array([4.2, 2.1, 0.8]),
        ],
    )
    def test_foot_matches_occt(self, query: np.ndarray) -> None:
        foot = foot_on_torus(
            jnp.asarray(query, dtype=jnp.float64),
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 1.0]),
            jnp.asarray(5.0),
            jnp.asarray(1.0),
        )
        expected = _occt_foot_on_face(self._face(), query)
        np.testing.assert_allclose(np.asarray(foot), expected, atol=ABS_TOL)


class TestGradient:
    """jax.grad through each foot function must be finite."""

    def test_plane_grad_finite(self) -> None:
        def loss(params):
            foot = foot_on_plane(
                jnp.array([0.3, 0.4, 2.5]),
                params["normal"],
                params["offset"],
            )
            return jnp.sum(foot**2)

        g = jax.grad(loss)(
            {"normal": jnp.array([0.0, 0.0, 1.0]), "offset": jnp.asarray(1.0)}
        )
        assert jnp.all(jnp.isfinite(g["normal"]))
        assert jnp.isfinite(g["offset"])

    def test_sphere_grad_finite(self) -> None:
        def loss(radius):
            foot = foot_on_sphere(jnp.array([5.0, 0.0, 0.0]), jnp.zeros(3), radius)
            return jnp.sum(foot**2)

        g = jax.grad(loss)(jnp.asarray(3.0))
        assert jnp.isfinite(g)

    def test_cylinder_grad_finite(self) -> None:
        def loss(radius):
            foot = foot_on_cylinder(
                jnp.array([3.0, 0.0, 5.0]),
                jnp.zeros(3),
                jnp.array([0.0, 0.0, 1.0]),
                radius,
            )
            return jnp.sum(foot**2)

        g = jax.grad(loss)(jnp.asarray(2.0))
        assert jnp.isfinite(g)

    def test_cone_grad_finite(self) -> None:
        def loss(angle):
            foot = foot_on_cone(
                jnp.array([4.0, 0.0, 3.0]),
                jnp.array([0.0, 0.0, 9.0]),
                jnp.array([0.0, 0.0, -1.0]),
                angle,
            )
            return jnp.sum(foot**2)

        g = jax.grad(loss)(jnp.asarray(np.arctan(1.0 / 3.0)))
        assert jnp.isfinite(g)

    def test_torus_grad_finite(self) -> None:
        def loss(params):
            foot = foot_on_torus(
                jnp.array([7.0, 0.0, 0.0]),
                jnp.zeros(3),
                jnp.array([0.0, 0.0, 1.0]),
                params["major"],
                params["minor"],
            )
            return jnp.sum(foot**2)

        g = jax.grad(loss)({"major": jnp.asarray(5.0), "minor": jnp.asarray(1.0)})
        assert jnp.isfinite(g["major"])
        assert jnp.isfinite(g["minor"])


class TestDegenerateAxisFallback:
    """Query points on the axis with axis parallel to +x.

    Tests that the orthogonal-vector fallback produces a unit radial
    direction for any axis orientation.  The 3D foot must still land on
    the surface (at distance ``radius`` / along the cone slant / on the
    torus tube) and the gradient must stay finite.
    """

    def test_cylinder_on_axis_parallel_to_x(self) -> None:
        # Cylinder with axis +x through the origin, radius 2.
        # Query on the axis itself: foot must be on the surface (|foot - axis| = 2).
        point = jnp.zeros(3)
        axis = jnp.array([1.0, 0.0, 0.0])
        query = jnp.array([5.0, 0.0, 0.0])  # exactly on axis
        foot = foot_on_cylinder(query, point, axis, jnp.asarray(2.0))
        # Foot should be at axial position 5 and at radial distance 2.
        axial = jnp.dot(foot - point, axis)
        perp = foot - point - axial * axis
        assert jnp.isclose(axial, 5.0)
        assert jnp.isclose(jnp.linalg.norm(perp), 2.0, atol=1e-6)
        assert jnp.all(jnp.isfinite(foot))

    def test_cone_on_axis_parallel_to_x(self) -> None:
        # Apex at origin, axis +x, half-angle pi/6. Query on the axis.
        apex = jnp.zeros(3)
        axis = jnp.array([1.0, 0.0, 0.0])
        angle = jnp.asarray(np.pi / 6)
        query = jnp.array([3.0, 0.0, 0.0])  # on axis, h=3
        foot = foot_on_cone(query, apex, axis, angle)
        # For query on axis, optimal t = h*cos(angle) + 0.
        # Foot slant length equals t, and foot should lie on the cone surface.
        h_foot = jnp.dot(foot - apex, axis)
        r_foot = jnp.linalg.norm(foot - apex - h_foot * axis)
        assert jnp.isclose(r_foot, h_foot * jnp.tan(angle), atol=1e-6)
        assert jnp.all(jnp.isfinite(foot))

    def test_torus_on_axis_parallel_to_x(self) -> None:
        # Torus centered at origin, axis +x, major=5, minor=1.
        # Query on the central axis: foot should land on the tube
        # surface (distance minor_radius from a point on the major ring).
        center = jnp.zeros(3)
        axis = jnp.array([1.0, 0.0, 0.0])
        query = jnp.array([0.0, 0.0, 0.0])  # exactly on the central axis
        foot = foot_on_torus(query, center, axis, jnp.asarray(5.0), jnp.asarray(1.0))
        # Foot must be on the torus surface — closest tube point from origin
        # is at (0, +/- (major - minor), 0) or in the y-z plane at radius 4.
        # The implementation-defined fallback is the +y or +z direction depending
        # on the seed choice; we verify the invariant numerically.
        h_foot = jnp.dot(foot - center, axis)
        r_foot = jnp.linalg.norm(foot - center - h_foot * axis)
        # On-axis foot has h=0 and r = major_radius - minor_radius (closer tube side)
        # or major + minor (farther side). The canonical fallback lands on the
        # tube at radial distance major_radius, then outward by minor_radius.
        tube_side_dist = jnp.abs(jnp.sqrt((r_foot - 5.0) ** 2 + h_foot**2) - 1.0)
        assert jnp.isclose(tube_side_dist, 0.0, atol=1e-6)
        assert jnp.all(jnp.isfinite(foot))

    def test_sphere_grad_finite_at_center(self) -> None:
        # Degenerate: query coincides with center.  Gradient through radius
        # must stay finite thanks to the safe-denom pattern.
        def loss(params):
            foot = foot_on_sphere(params["query"], params["center"], params["radius"])
            return jnp.sum(foot**2)

        params = {
            "query": jnp.array([0.0, 0.0, 0.0]),
            "center": jnp.array([0.0, 0.0, 0.0]),
            "radius": jnp.asarray(3.0),
        }
        g = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(g["query"]))
        assert jnp.all(jnp.isfinite(g["center"]))
        assert jnp.isfinite(g["radius"])

    def test_cylinder_grad_finite_on_axis(self) -> None:
        def loss(params):
            foot = foot_on_cylinder(
                params["query"], jnp.zeros(3), params["axis"], params["radius"]
            )
            return jnp.sum(foot**2)

        params = {
            "query": jnp.array([5.0, 0.0, 0.0]),
            "axis": jnp.array([1.0, 0.0, 0.0]),
            "radius": jnp.asarray(2.0),
        }
        g = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(g["query"]))
        assert jnp.all(jnp.isfinite(g["axis"]))
        assert jnp.isfinite(g["radius"])

    def test_cone_grad_finite_on_axis(self) -> None:
        def loss(params):
            foot = foot_on_cone(
                params["query"], jnp.zeros(3), params["axis"], params["angle"]
            )
            return jnp.sum(foot**2)

        params = {
            "query": jnp.array([3.0, 0.0, 0.0]),
            "axis": jnp.array([1.0, 0.0, 0.0]),
            "angle": jnp.asarray(np.pi / 6),
        }
        g = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(g["query"]))
        assert jnp.all(jnp.isfinite(g["axis"]))
        assert jnp.isfinite(g["angle"])

    def test_torus_grad_finite_on_axis(self) -> None:
        def loss(params):
            foot = foot_on_torus(
                params["query"],
                jnp.zeros(3),
                params["axis"],
                params["major"],
                params["minor"],
            )
            return jnp.sum(foot**2)

        params = {
            "query": jnp.array([0.0, 0.0, 0.0]),
            "axis": jnp.array([1.0, 0.0, 0.0]),
            "major": jnp.asarray(5.0),
            "minor": jnp.asarray(1.0),
        }
        g = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(g["query"]))
        assert jnp.all(jnp.isfinite(g["axis"]))
        assert jnp.isfinite(g["major"])
        assert jnp.isfinite(g["minor"])
