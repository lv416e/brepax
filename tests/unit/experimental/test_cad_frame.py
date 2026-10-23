"""Dimensional and placement contracts of the fixed experimental CAD frame."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from OCP.BRep import BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape

from brepax.experimental.applications.cad_frame import CadFrame


@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
def test_dimensional_gradients_and_point_covariance(scale):
    frame = CadFrame(scale, (1e3, -2e3, 3e3))

    def measures(q):
        radius, height = frame.working_lengths(q)
        return jnp.stack(
            (
                frame.external_measure(jnp.pi * radius**2 * height, 3),
                frame.external_measure(2 * jnp.pi * radius * (radius + height), 2),
            )
        )

    inputs = scale * np.array([[3.0, 10.0], [3.5, 12.0]])
    actual = jax.jit(jax.vmap(measures))(inputs)
    derivative = jax.jit(jax.vmap(jax.jacrev(measures)))(inputs)
    r, h = inputs.T
    expected = np.column_stack((np.pi * r**2 * h, 2 * np.pi * r * (r + h)))
    expected_jac = np.stack(
        (
            np.column_stack((2 * np.pi * r * h, np.pi * r**2)),
            np.column_stack((2 * np.pi * (2 * r + h), 2 * np.pi * r)),
        ),
        axis=1,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    np.testing.assert_allclose(derivative, expected_jac, rtol=1e-12)
    points = jnp.array([[1.0, 2.0, 3.0], [-2.0, 3.0, 1.0]])
    np.testing.assert_allclose(
        jax.jit(frame.external_points)(points),
        np.array(frame.origin) + scale * points,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        jax.jacfwd(frame.external_points)(points[0]), scale * np.eye(3)
    )


@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
def test_restored_rotated_cad_dimensions_and_placement(scale):
    """Independent volume and vertex oracles detect lost scale, origin or rotation."""
    frame = CadFrame(scale, (1e3, -2e3, 3e3))
    external_dimensions = jnp.array([40.0, 20.0, 10.0]) * scale
    working = np.asarray(frame.working_lengths(external_dimensions))
    box = BRepPrimAPI_MakeBox(*working).Shape()
    rotation = gp_Trsf()
    rotation.SetRotation(gp_Ax1(gp_Pnt(), gp_Dir(1, 2, 3)), 0.7)
    shape = BRepBuilderAPI_Transform(box, rotation, True, False).Shape()
    restored = frame.restore_shape(shape)
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(restored, props)
    assert props.Mass() == pytest.approx(float(jnp.prod(external_dimensions)), rel=1e-9)
    matrix = np.array([[rotation.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])
    corners = np.array([[x, y, z] for x in (0, 40) for y in (0, 20) for z in (0, 10)])
    expected = np.array(frame.origin) + scale * (corners @ matrix.T)
    vertices = []
    explorer = TopExp_Explorer(restored, TopAbs_VERTEX)
    while explorer.More():
        p = BRep_Tool.Pnt_s(TopoDS.Vertex_s(explorer.Current()))
        vertices.append([p.X(), p.Y(), p.Z()])
        explorer.Next()
    error = np.linalg.norm(expected[:, None] - np.array(vertices), axis=2).min(axis=1)
    assert error.max() / scale < 1e-6


@pytest.mark.parametrize(
    "scale,origin",
    [
        (0.0, (0.0, 0.0, 0.0)),
        (-1.0, (0.0, 0.0, 0.0)),
        (float("nan"), (0.0, 0.0, 0.0)),
        (float("inf"), (0.0, 0.0, 0.0)),
        (1.0, (float("inf"), 0.0, 0.0)),
    ],
)
def test_invalid_frame_rejected(scale, origin):
    with pytest.raises(ValueError):
        CadFrame(scale, origin)


def test_null_cad_rejected():
    with pytest.raises(ValueError, match="null"):
        CadFrame(1.0).restore_shape(TopoDS_Shape())
