"""Independent CAD, analytic derivative and placement oracles for hole edits."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from OCP.BRep import BRep_Builder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Compound, TopoDS_Shape

from brepax.brep.triangulate import (
    divergence_volume,
    mesh_surface_area,
    triangulate_shape,
)
from brepax.experimental.applications.cad_frame import CadFrame
from brepax.experimental.applications.hole_edit import prepare_hole_edit
from brepax.io.step import read_step


def cylinder(base, axis, radius, height):
    return BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(*base), gp_Dir(*axis)), radius, height
    ).Shape()


def stock(kind="rectangular"):
    if kind == "round":
        shape = BRepPrimAPI_MakeCylinder(20, 6).Shape()
        bases, axes, heights = [(-7, 0, 0), (7, 0, 0)], [(0, 0, 1)] * 2, [6, 6]
        volume, area, center = np.pi * 20**2 * 6, 2 * np.pi * 20 * 26, [0, 0, 3]
    elif kind == "orthogonal":
        shape = BRepPrimAPI_MakeBox(40, 24, 16).Shape()
        bases, axes, heights = (
            [(10, 6, 0), (0, 18, 8)],
            [(0, 0, 1), (1, 0, 0)],
            [16, 40],
        )
        volume, area, center = (
            40 * 24 * 16,
            2 * (40 * 24 + 24 * 16 + 40 * 16),
            [20, 12, 8],
        )
    else:
        shape = BRepPrimAPI_MakeBox(40, 24, 6).Shape()
        bases, axes, heights = [(10, 12, 0), (28, 12, 0)], [(0, 0, 1)] * 2, [6, 6]
        volume, area, center = 40 * 24 * 6, 2 * (40 * 24 + 24 * 6 + 40 * 6), [20, 12, 3]
    for base, axis, radius, height in zip(bases, axes, [2, 3], heights, strict=True):
        shape = BRepAlgoAPI_Cut(shape, cylinder(base, axis, radius, height)).Shape()
    centers = np.array(bases) + np.array(heights)[:, None] * np.array(axes) / 2
    return shape, volume, area, np.array(center), np.array(heights), centers


def oracle(kind, radii):
    _, stock_volume, stock_area, stock_center, heights, centers = stock(kind)
    voids = np.pi * radii**2 * heights
    volume = stock_volume - voids.sum()
    area = stock_area + np.sum(2 * np.pi * radii * heights - 2 * np.pi * radii**2)
    center = (stock_volume * stock_center - voids @ centers) / volume
    dv = -2 * np.pi * radii * heights
    da = 2 * np.pi * heights - 4 * np.pi * radii
    dc = ((centers - center) * dv[:, None] / volume).T
    return np.array([volume, area]), np.stack([dv, da]), center, dc


def cad_metrics(shape):
    volume, area = GProp_GProps(), GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, volume)
    BRepGProp.SurfaceProperties_s(shape, area)
    center = volume.CentreOfMass()
    return np.array([volume.Mass(), area.Mass()]), np.array(
        [center.X(), center.Y(), center.Z()]
    )


def ordered_edit(shape, **kwargs):
    initial = prepare_hole_edit(shape, **kwargs)
    ids = [hole.face_id for hole in sorted(initial.holes, key=lambda hole: hole.radius)]
    return prepare_hole_edit(shape, ids, **kwargs)


def metrics(model, radii):
    return jnp.stack([model.volume(radii), model.surface_area(radii)])


def assert_closed(triangles):
    # Exact coordinate connectivity: the public output must share its vertices.
    _, inverse = np.unique(
        np.asarray(triangles).reshape(-1, 3), axis=0, return_inverse=True
    )
    faces = inverse.reshape(-1, 3)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    directed = {tuple(edge) for edge in edges}
    assert len(directed) == len(edges)
    assert all((b, a) in directed for a, b in directed)


@pytest.mark.parametrize("kind", ["rectangular", "round", "orthogonal"])
def test_values_derivatives_closed_boundaries_and_independent_cad(kind):
    shape, *_ = stock(kind)
    edit = ordered_edit(shape, deflection=0.000025)
    radii = jnp.array([2.5, 2.4])
    expected, derivative, center, center_derivative = oracle(kind, np.asarray(radii))
    values = jax.jit(metrics)(edit.model, radii)
    jacobian = jax.jit(jax.jacrev(metrics, argnums=1))(edit.model, radii)
    np.testing.assert_allclose(values, expected, rtol=1e-5)
    np.testing.assert_allclose(jacobian, derivative, rtol=1e-4)
    np.testing.assert_allclose(edit.model.center_of_mass(radii), center, atol=1e-5)
    np.testing.assert_allclose(
        jax.jacrev(edit.model.center_of_mass)(radii),
        center_derivative,
        rtol=2e-4,
        atol=1e-7,
    )
    rebuilt = edit.rebuild(radii)
    np.testing.assert_allclose(cad_metrics(rebuilt)[0], expected, rtol=1e-10)
    fresh, _ = triangulate_shape(rebuilt, deflection=0.000025)
    np.testing.assert_allclose(
        [divergence_volume(fresh), mesh_surface_area(fresh)], expected, rtol=1e-5
    )
    trials = jnp.stack(
        [
            edit.model.reference_radii,
            edit.model.lower_radii,
            edit.model.upper_radii,
            radii,
        ]
    )
    assert np.asarray(jax.jit(jax.vmap(edit.model.is_valid))(trials)).all()
    for mesh in jax.jit(jax.vmap(edit.model.triangles))(trials):
        assert_closed(mesh)
    # Independent rebuilt-CAD finite differences, not derivatives of the mesh.
    for step in (1e-2, 1e-3):
        columns = []
        for i in range(2):
            delta = jnp.eye(2)[i] * step
            columns.append(
                (
                    cad_metrics(edit.rebuild(radii + delta))[0]
                    - cad_metrics(edit.rebuild(radii - delta))[0]
                )
                / (2 * step)
            )
        np.testing.assert_allclose(jacobian, np.stack(columns, axis=1), rtol=1e-4)


@pytest.mark.parametrize("scale", [0.001, 1000.0])
def test_fresh_cad_rotation_translation_units_and_gradient_laws(scale):
    shape, *_ = stock("rectangular")
    rotation = gp_Trsf()
    rotation.SetRotation(gp_Ax1(gp_Pnt(), gp_Dir(1, 2, 3)), 0.7)
    matrix = np.array([[rotation.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])
    origin = np.array([1000.0, -2000.0, 3000.0])
    rotation.SetValues(*np.column_stack([scale * matrix, origin]).reshape(-1))
    moved = BRepBuilderAPI_Transform(shape, rotation, True, False).Shape()
    edit = ordered_edit(
        moved, frame=CadFrame(scale, tuple(origin)), deflection=0.000025
    )
    radii = scale * jnp.array([2.5, 2.4])
    values, derivative, center, center_derivative = oracle(
        "rectangular", np.array([2.5, 2.4])
    )
    np.testing.assert_allclose(
        jax.jit(metrics)(edit.model, radii), values * [scale**3, scale**2], rtol=1e-5
    )
    np.testing.assert_allclose(
        jax.jacrev(metrics, argnums=1)(edit.model, radii),
        derivative * np.array([scale**2, scale])[:, None],
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        (edit.model.center_of_mass(radii) - origin) / scale, matrix @ center, atol=1e-5
    )
    np.testing.assert_allclose(
        jax.jacrev(edit.model.center_of_mass)(radii),
        matrix @ center_derivative,
        rtol=2e-4,
        atol=1e-7,
    )
    rebuilt_values, rebuilt_center = cad_metrics(edit.rebuild(radii))
    np.testing.assert_allclose(rebuilt_values, values * [scale**3, scale**2], rtol=1e-8)
    np.testing.assert_allclose(
        (rebuilt_center - origin) / scale, matrix @ center, atol=1e-7
    )


def step_roundtrip(shape, path):
    writer = STEPControl_Writer()
    assert writer.Transfer(shape, STEPControl_AsIs) == IFSelect_RetDone
    assert writer.Write(str(path)) == IFSelect_RetDone
    return read_step(path)


def test_step_selection_preserves_unselected_hole_and_input(tmp_path):
    source = read_step(Path(__file__).parents[2] / "fixtures/box_with_holes.step")
    all_holes = prepare_hole_edit(source)
    selected = next(hole for hole in all_holes.holes if hole.radius == pytest.approx(4))
    edit = prepare_hole_edit(source, [selected.face_id], deflection=0.0001)
    original = cad_metrics(source)[0]
    radii = jnp.array([4.4])
    edited = step_roundtrip(edit.rebuild(radii), tmp_path / "edited.step")
    found = prepare_hole_edit(edited)
    np.testing.assert_allclose(
        sorted(hole.radius for hole in found.holes), [3, 4.4], atol=1e-8
    )
    np.testing.assert_allclose(cad_metrics(source)[0], original, rtol=1e-14)
    np.testing.assert_allclose(
        cad_metrics(edit.rebuild(radii))[0], cad_metrics(edited)[0], rtol=1e-10
    )
    np.testing.assert_allclose(
        metrics(edit.model, radii), cad_metrics(edited)[0], rtol=1e-5
    )
    base = np.asarray(edit.model.triangles(edit.model.reference_radii))
    moved = np.asarray(edit.model.triangles(radii))
    fixed = np.isclose(np.linalg.norm(base[..., :2] - [30, 15], axis=-1), 3, atol=1e-7)
    assert fixed.any()
    np.testing.assert_array_equal(moved[fixed], base[fixed])


@pytest.mark.parametrize("kind", ["blind", "oblique", "near_boundary", "split"])
def test_unsupported_holes_rejected(kind):
    shape = BRepPrimAPI_MakeBox(40, 24, 6).Shape()
    if kind == "blind":
        tool = cylinder((10, 12, 2), (0, 0, 1), 2, 4)
    elif kind == "oblique":
        tool = cylinder((10, 12, -4), (0.2, 0, 1), 2, 15)
    elif kind == "near_boundary":
        tool = cylinder((2.000001, 12, 0), (0, 0, 1), 2, 6)
    else:
        a = cylinder((10, 12, 0), (0, 0, 1), 2, 3)
        b = cylinder((10, 12, 3), (0, 0, 1), 2, 3)
        # Split the otherwise complete cylindrical wall by a coplanar Boolean seam.
        tool = BRepAlgoAPI_Fuse(a, b).Shape()
    with pytest.raises(ValueError, match="No supported through-holes"):
        prepare_hole_edit(BRepAlgoAPI_Cut(shape, tool).Shape())


def test_invalid_inputs_and_traced_failure_mask():
    shape, *_ = stock()
    edit = prepare_hole_edit(shape)
    model = edit.model
    assert bool(eqx.filter_jit(model.is_valid)(model.reference_radii))
    np.testing.assert_allclose(
        eqx.filter_jit(model.volume)(model.reference_radii),
        model.volume(model.reference_radii),
        rtol=1e-12,
    )
    bad = jnp.stack(
        [model.lower_radii * 0.9, model.upper_radii * 1.1, jnp.array([jnp.nan, 3.0])]
    )
    assert not np.asarray(jax.jit(jax.vmap(model.is_valid))(bad)).any()
    assert np.isnan(jax.jit(jax.vmap(model.volume))(bad)).all()
    for radii in bad:
        with pytest.raises(ValueError, match="Invalid radii"):
            edit.rebuild(radii)
    for ids in ([], [0], [edit.holes[0].face_id] * 2):
        with pytest.raises(ValueError):
            prepare_hole_edit(shape, ids)
    for deflection in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="deflection"):
            prepare_hole_edit(shape, deflection=deflection)
    with pytest.raises(ValueError, match="non-null"):
        prepare_hole_edit(TopoDS_Shape())
    with pytest.raises(ValueError, match="No supported"):
        prepare_hole_edit(BRepPrimAPI_MakeBox(1, 2, 3).Shape())
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="x64"):
            prepare_hole_edit(shape)
    finally:
        jax.config.update("jax_enable_x64", True)


def test_connected_material_inside_bore_rejected():
    shape = BRepAlgoAPI_Cut(
        BRepPrimAPI_MakeBox(40, 24, 6).Shape(),
        cylinder((10, 12, 0), (0, 0, 1), 3, 6),
    ).Shape()
    # An overhead bracket holds a peg inside the hole without touching its wall.
    for piece in (
        BRepPrimAPI_MakeBox(gp_Pnt(3, 10, 6), 2, 4, 4).Shape(),
        BRepPrimAPI_MakeBox(gp_Pnt(3, 10, 8), 9, 4, 2).Shape(),
        cylinder((10, 12, 1), (0, 0, 1), 1, 8),
    ):
        shape = BRepAlgoAPI_Fuse(shape, piece).Shape()
    with pytest.raises(ValueError, match="contains other solid material"):
        prepare_hole_edit(shape)


def test_inverse_edit_of_three_step_holes_preserves_fourth(tmp_path):
    centers = [(12, 10), (42, 10), (25, 30), (50, 30)]

    def plate(radii):
        shape = BRepPrimAPI_MakeBox(60, 40, 8).Shape()
        for (x, y), radius in zip(centers, radii, strict=True):
            shape = BRepAlgoAPI_Cut(
                shape, cylinder((x, y, 0), (0, 0, 1), radius, 8)
            ).Shape()
        return shape

    source = step_roundtrip(plate([2.4, 2.8, 3.2, 2.0]), tmp_path / "source.step")
    discovery = prepare_hole_edit(source)
    selected = sorted(discovery.holes, key=lambda hole: hole.radius)[1:]
    edit = prepare_hole_edit(
        source, [hole.face_id for hole in selected], deflection=0.000025
    )
    target_radii = np.array([2.8, 2.4, 3.0])
    values, center = cad_metrics(plate([*target_radii, 2.0]))
    target = jnp.array([values[0], *center[:2]])

    def residual(radii):
        actual = jnp.concatenate(
            [
                jnp.atleast_1d(edit.model.volume(radii)),
                edit.model.center_of_mass(radii)[:2],
            ]
        )
        return (actual - target) / jnp.array([19200.0, 60.0, 60.0])

    value_and_jacobian = jax.jit(lambda r: (residual(r), jax.jacfwd(residual)(r)))
    radii = edit.model.reference_radii
    for _ in range(5):
        residual_value, jacobian = value_and_jacobian(radii)
        assert np.linalg.matrix_rank(np.asarray(jacobian), tol=1e-9) == 3
        radii = radii - jnp.linalg.solve(jacobian, residual_value)
        assert bool(edit.model.is_valid(radii))
    np.testing.assert_allclose(radii, target_radii, rtol=1e-4)
    output = step_roundtrip(edit.rebuild(radii), tmp_path / "edited.step")
    rebuilt, rebuilt_center = cad_metrics(output)
    np.testing.assert_allclose(rebuilt, values, rtol=1e-5)
    np.testing.assert_allclose(rebuilt_center, center, atol=1e-5)
    # Output feature IDs can change; discover the unchanged smallest bore again.
    found = prepare_hole_edit(output)
    assert min(hole.radius for hole in found.holes) == pytest.approx(2.0, abs=1e-8)


def test_counterbore_shoulder_is_not_a_through_hole_end():
    shape = BRepPrimAPI_MakeBox(40, 24, 6).Shape()
    for radius, base, height in ((2, (10, 12, 0), 6), (4, (10, 12, 3), 3)):
        shape = BRepAlgoAPI_Cut(
            shape, cylinder(base, (0, 0, 1), radius, height)
        ).Shape()
    edit = prepare_hole_edit(shape, deflection=0.000025)
    assert len(edit.holes) == 1
    assert edit.holes[0].radius == pytest.approx(2)
    assert len(edit.rejected) == 1
    assert "orientation" in edit.rejected[0].reason
    with pytest.raises(ValueError, match="not a supported through-hole"):
        prepare_hole_edit(shape, [edit.rejected[0].face_id])
    # The smaller bore still passes through its own lower slab and is editable.
    radii = jnp.array([2.2])
    expected = BRepAlgoAPI_Cut(shape, cylinder((10, 12, 0), (0, 0, 1), 2.2, 3)).Shape()
    np.testing.assert_allclose(
        metrics(edit.model, radii), cad_metrics(expected)[0], rtol=1e-5
    )
    np.testing.assert_allclose(
        cad_metrics(edit.rebuild(radii))[0], cad_metrics(expected)[0], rtol=1e-10
    )
    step = 0.001
    references = [
        cad_metrics(
            BRepAlgoAPI_Cut(shape, cylinder((10, 12, 0), (0, 0, 1), r, 3)).Shape()
        )[0]
        for r in (2.2 - step, 2.2 + step)
    ]
    np.testing.assert_allclose(
        jax.jacrev(metrics, argnums=1)(edit.model, radii)[:, 0],
        (references[1] - references[0]) / (2 * step),
        rtol=1e-4,
    )


def compound(*shapes):
    result = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(result)
    for shape in shapes:
        builder.Add(result, shape)
    return result


@pytest.mark.parametrize(
    "kind", [TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_SOLID]
)
def test_duplicate_topological_occurrences_rejected(kind):
    shape, *_ = stock()
    duplicate = TopExp_Explorer(shape, kind).Current()
    with pytest.raises(ValueError, match="outside its solid"):
        prepare_hole_edit(compound(shape, duplicate))


def test_nested_single_solid_compound_is_editable():
    shape, *_ = stock()
    edit = prepare_hole_edit(compound(compound(shape)))
    radii = edit.model.reference_radii * 1.05
    np.testing.assert_allclose(
        metrics(edit.model, radii), cad_metrics(edit.rebuild(radii))[0], rtol=1e-4
    )
