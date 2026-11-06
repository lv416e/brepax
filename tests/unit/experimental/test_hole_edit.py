"""Independent CAD, analytic derivative and placement oracles for hole edits.

The exact model is held to CAD precision; the optional frozen mesh to its
discretization error.
"""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from OCP.BRep import BRep_Builder
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_Copy,
    BRepBuilderAPI_NurbsConvert,
    BRepBuilderAPI_Transform,
)
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeSphere,
)
from OCP.GeomAbs import GeomAbs_Cylinder
from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.ShapeUpgrade import ShapeUpgrade_ShapeDivideClosed
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shape
from OCP.TopTools import TopTools_IndexedMapOfShape

from brepax.brep.topology import build_adjacency_graph, neighbors
from brepax.brep.triangulate import (
    divergence_volume,
    mesh_surface_area,
    triangulate_shape,
)
from brepax.experimental.applications._hole_geometry import (
    _local_edit_error,
    normalize_shape,
    rebuild_holes,
    recognize_holes,
)
from brepax.experimental.applications.cad_frame import CadFrame
from brepax.experimental.applications.hole_edit import prepare_hole_edit
from brepax.experimental.applications.hole_mesh import prepare_hole_mesh
from brepax.io.step import read_step


def cylinder(base, axis, radius, height):
    return BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(*base), gp_Dir(*axis)), radius, height
    ).Shape()


def stock(kind="rectangular", radii=(2, 3), shifts=None):
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
    if shifts is not None:
        bases = np.asarray(bases) + shifts
    for base, axis, radius, height in zip(bases, axes, radii, heights, strict=True):
        shape = BRepAlgoAPI_Cut(shape, cylinder(base, axis, radius, height)).Shape()
    centers = np.array(bases) + np.array(heights)[:, None] * np.array(axes) / 2
    return shape, volume, area, np.array(center), np.array(heights), centers


def oracle(kind, radii, shifts=None):
    _, stock_volume, stock_area, stock_center, heights, centers = stock(kind)
    if shifts is not None:
        centers = centers + shifts
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
    edit = ordered_edit(shape)
    radii = jnp.array([2.5, 2.4])
    expected, derivative, center, center_derivative = oracle(kind, np.asarray(radii))
    values = jax.jit(metrics)(edit.model, radii)
    jacobian = jax.jit(jax.jacrev(metrics, argnums=1))(edit.model, radii)
    np.testing.assert_allclose(values, expected, rtol=1e-10)
    np.testing.assert_allclose(jacobian, derivative, rtol=1e-12)
    np.testing.assert_allclose(edit.model.center_of_mass(radii), center, atol=1e-9)
    np.testing.assert_allclose(
        jax.jacrev(edit.model.center_of_mass)(radii),
        center_derivative,
        rtol=1e-8,
        atol=1e-10,
    )
    rebuilt = edit.rebuild(radii)
    np.testing.assert_allclose(cad_metrics(rebuilt)[0], expected, rtol=1e-10)
    trials = jnp.stack(
        [
            edit.model.reference_radii,
            edit.model.lower_radii,
            (edit.model.reference_radii + edit.model.upper_radii) / 2,
            radii,
        ]
    )
    assert np.asarray(jax.jit(jax.vmap(edit.model.is_valid))(trials)).all()
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
    edit = ordered_edit(moved, frame=CadFrame(scale, tuple(origin)))
    radii = scale * jnp.array([2.5, 2.4])
    values, derivative, center, center_derivative = oracle(
        "rectangular", np.array([2.5, 2.4])
    )
    np.testing.assert_allclose(
        jax.jit(metrics)(edit.model, radii), values * [scale**3, scale**2], rtol=1e-9
    )
    np.testing.assert_allclose(
        jax.jacrev(metrics, argnums=1)(edit.model, radii),
        derivative * np.array([scale**2, scale])[:, None],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        (edit.model.center_of_mass(radii) - origin) / scale, matrix @ center, atol=1e-8
    )
    np.testing.assert_allclose(
        jax.jacrev(edit.model.center_of_mass)(radii),
        matrix @ center_derivative,
        rtol=1e-8,
        atol=1e-10,
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
    edit = prepare_hole_edit(source, [selected.face_id])
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
        metrics(edit.model, radii), cad_metrics(edited)[0], rtol=1e-9
    )
    mesh = prepare_hole_mesh(edit, deflection=0.0001)
    base = np.asarray(mesh.triangles(edit.model.reference_radii))
    moved = np.asarray(mesh.triangles(radii))
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


def _single_hole_box_measures(base, axis, height, radius, shift=(0.0, 0.0, 0.0)):
    """Independent analytic volume/area/center for one hole in a 40x24x6 box.

    Standard box-minus-cylinder mass properties, matching the formulas
    ``oracle()`` uses for the multi-hole stocks but written out directly so
    this does not depend on how ``prepare_hole_edit`` itself recognizes or
    partitions the hole.
    """
    box_volume, box_area = 40 * 24 * 6, 2 * (40 * 24 + 24 * 6 + 40 * 6)
    box_center = np.array([20.0, 12.0, 3.0])
    center = (
        np.asarray(base, dtype=float)
        + height / 2 * np.asarray(axis, dtype=float)
        + np.asarray(shift, dtype=float)
    )
    void = np.pi * radius**2 * height
    volume = box_volume - void
    area = box_area + 2 * np.pi * radius * height - 2 * np.pi * radius**2
    centroid = (box_volume * box_center - void * center) / volume
    return volume, area, centroid


def test_merge_same_domain_recovers_axially_split_hole():
    base, axis, height = (10, 12, 0), (0, 0, 1), 6
    shape = BRepPrimAPI_MakeBox(40, 24, 6).Shape()
    a = cylinder(base, axis, 2, height / 2)
    b = cylinder((10, 12, height / 2), axis, 2, height / 2)
    # Same construction as the rejected "split" case: a coplanar Boolean seam
    # divides the otherwise complete cylindrical wall into two faces.
    tool = BRepAlgoAPI_Fuse(a, b).Shape()
    split = BRepAlgoAPI_Cut(shape, tool).Shape()

    with pytest.raises(ValueError, match="No supported through-holes"):
        prepare_hole_edit(split)

    edit = ordered_edit(split, merge_same_domain=True)
    assert len(edit.holes) == 1
    radii = jnp.array([2.6])
    shifts = jnp.array([[0.1, -0.05, 0.0]])
    volume, area, center = _single_hole_box_measures(
        base, axis, height, float(radii[0]), (0.1, -0.05, 0.0)
    )
    np.testing.assert_allclose(metrics(edit.model, radii), [volume, area], rtol=1e-9)
    np.testing.assert_allclose(
        edit.model.center_of_mass(radii, shifts), center, atol=1e-8
    )
    rebuilt = edit.rebuild(radii, shifts)
    np.testing.assert_allclose(cad_metrics(rebuilt)[0], [volume, area], rtol=1e-9)


def test_merge_same_domain_recovers_half_cylinder_faces():
    base, axis, height, radius0 = (10, 12, 0), (0, 0, 1), 6, 2.0
    box = BRepPrimAPI_MakeBox(40, 24, 6).Shape()
    tool = cylinder(base, axis, radius0, height)
    shape = BRepAlgoAPI_Cut(box, tool).Shape()
    # ShapeUpgrade_ShapeDivideClosed splits the periodic cylindrical face at
    # one seam, as a STEP exporter's own face partition commonly does.
    divider = ShapeUpgrade_ShapeDivideClosed(shape)
    divider.SetNbSplitPoints(1)
    assert divider.Perform()
    split = divider.Result()

    with pytest.raises(ValueError, match="No supported through-holes"):
        prepare_hole_edit(split)

    edit = ordered_edit(split, merge_same_domain=True)
    assert len(edit.holes) == 1
    radii = jnp.array([2.6])
    volume, area, center = _single_hole_box_measures(
        base, axis, height, float(radii[0])
    )
    np.testing.assert_allclose(metrics(edit.model, radii), [volume, area], rtol=1e-9)
    np.testing.assert_allclose(edit.model.center_of_mass(radii), center, atol=1e-8)
    rebuilt = edit.rebuild(radii)
    np.testing.assert_allclose(cad_metrics(rebuilt)[0], [volume, area], rtol=1e-9)


def test_merge_same_domain_default_is_unchanged():
    shape, *_ = stock("rectangular")
    default = prepare_hole_edit(shape)
    explicit = prepare_hole_edit(shape, merge_same_domain=False)
    assert [h.face_id for h in default.holes] == [h.face_id for h in explicit.holes]
    meshes = [
        prepare_hole_mesh(edit, deflection=0.0001) for edit in (default, explicit)
    ]
    np.testing.assert_array_equal(
        np.asarray(meshes[0].vertices), np.asarray(meshes[1].vertices)
    )
    np.testing.assert_array_equal(
        np.asarray(meshes[0].faces), np.asarray(meshes[1].faces)
    )


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
            prepare_hole_mesh(edit, deflection=deflection)
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
    with pytest.raises(ValueError, match="touches the hole opening"):
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
    edit = prepare_hole_edit(source, [hole.face_id for hole in selected])
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
    np.testing.assert_allclose(radii, target_radii, rtol=1e-8)
    output = step_roundtrip(edit.rebuild(radii), tmp_path / "edited.step")
    rebuilt, rebuilt_center = cad_metrics(output)
    np.testing.assert_allclose(rebuilt, values, rtol=1e-8)
    np.testing.assert_allclose(rebuilt_center, center, atol=1e-8)
    # Output feature IDs can change; discover the unchanged smallest bore again.
    found = prepare_hole_edit(output)
    assert min(hole.radius for hole in found.holes) == pytest.approx(2.0, abs=1e-8)


def test_counterbore_shoulder_is_not_a_through_hole_end():
    shape = BRepPrimAPI_MakeBox(40, 24, 6).Shape()
    for radius, base, height in ((2, (10, 12, 0), 6), (4, (10, 12, 3), 3)):
        shape = BRepAlgoAPI_Cut(
            shape, cylinder(base, (0, 0, 1), radius, height)
        ).Shape()
    edit = prepare_hole_edit(shape)
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
        metrics(edit.model, radii), cad_metrics(expected)[0], rtol=1e-9
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
        metrics(edit.model, radii), cad_metrics(edit.rebuild(radii))[0], rtol=1e-9
    )


def position_metrics(model, radii, shifts):
    return jnp.concatenate(
        [
            jnp.array([model.volume(radii, shifts), model.surface_area(radii, shifts)]),
            model.center_of_mass(radii, shifts),
        ]
    )


@pytest.mark.parametrize("kind", ["rectangular", "round", "orthogonal"])
def test_center_and_radius_edits_against_independent_cad(kind):
    source, *_ = stock(kind)
    edit = ordered_edit(source)
    radii = jnp.array([2.3, 2.8])
    shifts = jnp.array([[0.2, -0.3, 0.0], [-0.25, 0.2, 0.0]])
    if kind == "orthogonal":
        shifts = shifts.at[1].set(jnp.array([0.0, 0.2, -0.25]))
    expected, dr, center, dc = oracle(kind, np.asarray(radii), shifts)
    values = jax.jit(position_metrics)(edit.model, radii, shifts)
    np.testing.assert_allclose(values[:2], expected, rtol=1e-10)
    np.testing.assert_allclose(values[2:], center, atol=1e-9)
    radius_jac, shift_jac = jax.jit(jax.jacrev(position_metrics, argnums=(1, 2)))(
        edit.model, radii, shifts
    )
    np.testing.assert_allclose(
        radius_jac, np.concatenate([dr, dc]), rtol=1e-8, atol=1e-10
    )
    # Translation leaves V/A unchanged; COM responds to the translated void.
    axes = np.array([hole.axis for hole in edit.holes])
    projectors = np.eye(3) - axes[:, :, None] * axes[:, None, :]
    voids = np.pi * np.asarray(radii) ** 2 * [hole.height for hole in edit.holes]
    expected_dc = np.einsum("h,hij->ihj", -voids / expected[0], projectors)
    np.testing.assert_allclose(shift_jac[:2], 0, atol=1e-12)
    np.testing.assert_allclose(shift_jac[2:], expected_dc, rtol=1e-8, atol=1e-10)
    independent, *_ = stock(kind, radii, shifts)
    rebuilt = edit.rebuild(radii, shifts)
    for shape in (independent, rebuilt):
        actual, actual_center = cad_metrics(shape)
        np.testing.assert_allclose(actual, expected, rtol=1e-10)
        np.testing.assert_allclose(actual_center, center, atol=1e-9)
    # Perturb independent CAD construction, not the editing implementation.
    for step in (0.01, 0.001):
        for h, component in ((0, 0), (1, 1)):
            delta = np.zeros((2, 3))
            delta[h, component] = step
            samples = []
            for sign in (-1, 1):
                shape, *_ = stock(kind, radii, shifts + sign * delta)
                measures, com = cad_metrics(shape)
                samples.append(np.concatenate([measures, com]))
            np.testing.assert_allclose(
                shift_jac[:, h, component],
                (samples[1] - samples[0]) / (2 * step),
                rtol=2e-4,
                atol=1e-7,
            )


@pytest.mark.parametrize("scale", [0.001, 1000.0])
def test_center_edit_fresh_placement_and_gradient_covariance(scale):
    shape, *_ = stock("orthogonal")
    transform = gp_Trsf()
    transform.SetRotation(gp_Ax1(gp_Pnt(), gp_Dir(1, 2, 3)), 0.7)
    rotation = np.array([[transform.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])
    origin = np.array([1000.0, -2000.0, 3000.0])
    transform.SetValues(*np.column_stack([scale * rotation, origin]).reshape(-1))
    moved = BRepBuilderAPI_Transform(shape, transform, True, False).Shape()
    edit = ordered_edit(moved, frame=CadFrame(scale, tuple(origin)))
    radii = np.array([2.3, 2.8])
    shifts = np.array([[0.2, -0.3, 0.0], [0.0, 0.2, -0.25]])
    moved_radii = jnp.asarray(scale * radii)
    moved_shifts = jnp.asarray(scale * shifts @ rotation.T)
    expected, dr, center, dc = oracle("orthogonal", radii, shifts)
    values = position_metrics(edit.model, moved_radii, moved_shifts)
    np.testing.assert_allclose(values[:2], expected * [scale**3, scale**2], rtol=1e-9)
    np.testing.assert_allclose(
        (values[2:] - origin) / scale, rotation @ center, atol=1e-8
    )
    radius_jac, shift_jac = jax.jit(jax.jacrev(position_metrics, argnums=(1, 2)))(
        edit.model, moved_radii, moved_shifts
    )
    np.testing.assert_allclose(
        radius_jac[:2] / np.array([scale**2, scale])[:, None], dr, rtol=1e-8
    )
    np.testing.assert_allclose(radius_jac[2:], rotation @ dc, rtol=1e-8, atol=1e-10)
    axes = np.array([[0, 0, 1], [1, 0, 0]]) @ rotation.T
    projectors = np.eye(3) - axes[:, :, None] * axes[:, None, :]
    voids = np.pi * radii**2 * [16, 40]
    expected_dc = np.einsum("h,hij->ihj", -voids / expected[0], projectors)
    np.testing.assert_allclose(
        shift_jac[:2] / np.array([scale**2, scale])[:, None, None], 0, atol=1e-10
    )
    np.testing.assert_allclose(shift_jac[2:], expected_dc, rtol=1e-8, atol=1e-10)
    actual, actual_center = cad_metrics(edit.rebuild(moved_radii, moved_shifts))
    np.testing.assert_allclose(actual, expected * [scale**3, scale**2], rtol=1e-8)
    np.testing.assert_allclose(
        (actual_center - origin) / scale, rotation @ center, atol=1e-7
    )


def test_center_edit_budget_rejection_batching_and_zero_default():
    shape, *_ = stock()
    edit = ordered_edit(shape)
    model = edit.model
    radii = model.reference_radii
    zero = jnp.zeros((2, 3))
    np.testing.assert_array_equal(model.volume(radii), model.volume(radii, zero))
    np.testing.assert_allclose(
        jax.jacrev(model.volume, argnums=1)(radii, zero), 0, atol=1e-12
    )
    limit = model.upper_radii - radii
    at_limit = zero.at[0, 0].set(limit[0])
    assert bool(eqx.filter_jit(model.is_valid)(radii, at_limit))
    assert_closed(prepare_hole_mesh(edit).triangles(radii, 0.1 * at_limit))
    actual, center = cad_metrics(edit.rebuild(radii, at_limit))
    np.testing.assert_allclose(
        position_metrics(model, radii, at_limit),
        np.concatenate([actual, center]),
        rtol=1e-8,
        atol=1e-8,
    )
    cases = jnp.stack(
        [
            zero,
            at_limit,
            at_limit * 1.01,
            zero.at[0, 2].set(1e-4),
            zero.at[1, 0].set(jnp.inf),
            zero.at[0, 1].set(jnp.nan),
        ]
    )
    flags = jax.jit(jax.vmap(model.is_valid, in_axes=(None, 0)))(radii, cases)
    np.testing.assert_array_equal(flags, [True, True, False, False, False, False])
    volumes = jax.jit(jax.vmap(model.volume, in_axes=(None, 0)))(radii, cases)
    np.testing.assert_allclose(volumes[:2], volumes[0], rtol=1e-12)
    assert np.isnan(volumes[2:]).all()
    for shifts in cases[2:]:
        with pytest.raises(ValueError, match="center shifts"):
            edit.rebuild(radii, shifts)
    # Individually legal growth and movement must not spend clearance twice.
    first = jnp.eye(2)[0] * limit[0]
    assert not bool(model.is_valid(radii + 0.6 * first, 0.6 * at_limit))
    assert bool(model.is_valid(radii + 0.4 * first, 0.6 * at_limit))


def test_inverse_center_edit_step_roundtrip_preserves_unselected_hole(tmp_path):
    source, *_ = stock()
    source = step_roundtrip(source, tmp_path / "source.step")
    found = prepare_hole_edit(source)
    selected = min(found.holes, key=lambda hole: hole.radius)
    edit = prepare_hole_edit(source, [selected.face_id])
    target_shift = np.array([0.6, -0.4, 0.0])
    target, *_ = stock(radii=(2.3, 3.0), shifts=np.array([target_shift, np.zeros(3)]))
    target_values, target_center = cad_metrics(target)

    def decode(parameters):
        return parameters[:1], jnp.array([[parameters[1], parameters[2], 0.0]])

    def residual(parameters):
        r, d = decode(parameters)
        return jnp.concatenate(
            [
                jnp.atleast_1d((edit.model.volume(r, d) - target_values[0]) / 5760),
                (edit.model.center_of_mass(r, d)[:2] - target_center[:2]) / 40,
            ]
        )

    evaluate = jax.jit(lambda p: (residual(p), jax.jacfwd(residual)(p)))
    parameters = jnp.array([2.0, 0.0, 0.0])
    for _ in range(5):
        value, derivative = evaluate(parameters)
        assert np.linalg.matrix_rank(np.asarray(derivative), tol=1e-9) == 3
        parameters -= jnp.linalg.solve(derivative, value)
        assert bool(edit.model.is_valid(*decode(parameters)))
    np.testing.assert_allclose(parameters, [2.3, *target_shift[:2]], rtol=1e-7)
    r, d = decode(parameters)
    result = step_roundtrip(edit.rebuild(r, d), tmp_path / "edited.step")
    actual, actual_center = cad_metrics(result)
    np.testing.assert_allclose(actual, target_values, rtol=1e-8)
    np.testing.assert_allclose(actual_center, target_center, atol=1e-8)
    holes = sorted(prepare_hole_edit(result).holes, key=lambda hole: hole.radius)
    np.testing.assert_allclose(holes[0].base, [10.6, 11.6, 0], atol=1e-6)
    np.testing.assert_allclose(holes[1].base, [28, 12, 0], atol=1e-9)
    assert holes[1].radius == pytest.approx(3, abs=1e-9)
    np.testing.assert_allclose(
        cad_metrics(source)[0], oracle("rectangular", np.array([2, 3]))[0], rtol=1e-12
    )


def domed_plate():
    # The B-spline dome shares the top face with the hole. Booleans that
    # rebuild that face shift whole-solid OCCT properties by about 1e-7.
    dome = BRepAlgoAPI_Cut(
        BRepPrimAPI_MakeSphere(gp_Pnt(30, 12, 6), 5).Shape(),
        BRepPrimAPI_MakeBox(gp_Pnt(20, 0, -10), gp_Pnt(40, 24, 6)).Shape(),
    ).Shape()
    return BRepAlgoAPI_Fuse(
        BRepPrimAPI_MakeBox(40, 24, 6).Shape(),
        BRepBuilderAPI_NurbsConvert(dome, True).Shape(),
    ).Shape()


def difference_volume(a, b):
    region = BRepAlgoAPI_Cut(
        BRepBuilderAPI_Copy(a, True, False).Shape(),
        BRepBuilderAPI_Copy(b, True, False).Shape(),
    ).Shape()
    return cad_metrics(region)[0][0]


@pytest.mark.parametrize("radius, shift", [(2.1, (0, 0, 0)), (2.05, (0.03, -0.02, 0))])
def test_rebuild_accepts_exact_edit_beside_bspline_faces(radius, shift):
    source = BRepAlgoAPI_Cut(domed_plate(), cylinder((10, 12, 0), (0, 0, 1), 2, 6))
    working, _ = normalize_shape(source.Shape(), CadFrame(1.0, (0.0, 0.0, 0.0)))
    (hole,), _, _ = recognize_holes(working)
    radii, shifts = np.array([radius]), np.array([shift], dtype=float)
    rebuilt = rebuild_holes(working, (hole,), radii, shifts)
    base = np.array([10.0, 12.0, 0.0]) + shifts[0]
    direct = BRepAlgoAPI_Cut(
        domed_plate(), cylinder(base, (0, 0, 1), radius, 6)
    ).Shape()
    assert difference_volume(rebuilt, direct) == pytest.approx(0, abs=1e-9)
    assert difference_volume(direct, rebuilt) == pytest.approx(0, abs=1e-9)
    misplaced = BRepAlgoAPI_Cut(
        domed_plate(), cylinder(base + np.array([0.2, 0, 0]), (0, 0, 1), radius, 6)
    ).Shape()
    assert _local_edit_error(working, misplaced, (hole,), radii, shifts) is not None
    assert _local_edit_error(working, rebuilt, (hole,), radii, shifts) is None
    # A notch far from the hole is outside the envelope; 0.008 exceeds the
    # whole-solid drift allowance (1e-7 of about 5800).
    notched = BRepAlgoAPI_Cut(
        rebuilt, BRepPrimAPI_MakeBox(gp_Pnt(39.8, 23.8, 0), 0.2, 0.2, 0.2).Shape()
    ).Shape()
    assert _local_edit_error(working, notched, (hole,), radii, shifts) == (
        "Rebuilt CAD changed material away from the edited holes"
    )


def test_rebuild_at_reference_returns_the_input_geometry():
    shape, *_ = stock()
    edit = prepare_hole_edit(shape)
    rebuilt = edit.rebuild(edit.model.reference_radii)
    np.testing.assert_allclose(
        cad_metrics(rebuilt)[0], cad_metrics(shape)[0], rtol=1e-12
    )


@pytest.mark.slow
def test_rebuild_leaves_unedited_step_holes_alone():
    # CTC-05's hole walls deviate from their ideal cylinders within its CAD
    # tolerance, so filling and re-cutting an unedited hole leaves a sliver
    # that the region check rejects.
    path = Path(__file__).resolve().parents[2] / "fixtures" / "nist"
    path = path / "nist_ctc_05_asme1_rd.stp"
    if not path.exists():
        pytest.skip("NIST fixture not available")
    explorer = TopExp_Explorer(read_step(path), TopAbs_SOLID)
    edit = prepare_hole_edit(compound(explorer.Current()), merge_same_domain=True)
    assert len(edit.holes) == 10
    model = edit.model
    edit.rebuild(model.reference_radii)
    radii = model.reference_radii.at[0].set(
        (model.reference_radii[0] + model.upper_radii[0]) / 2
    )
    rebuilt = edit.rebuild(radii)
    removed = (
        np.pi * (radii[0] ** 2 - model.reference_radii[0] ** 2) * edit.holes[0].height
    )
    before = cad_metrics(compound(explorer.Current()))[0][0]
    assert cad_metrics(rebuilt)[0][0] == pytest.approx(before - removed, rel=1e-6)


@pytest.mark.slow
def test_rebuild_accepts_many_edited_step_holes():
    # Whole-solid properties of CTC-05 drift past their bound once three of
    # its holes are edited, and the difference Booleans between the two
    # nearly coincident solids return a wrong region.
    path = Path(__file__).resolve().parents[2] / "fixtures" / "nist"
    path = path / "nist_ctc_05_asme1_rd.stp"
    if not path.exists():
        pytest.skip("NIST fixture not available")
    explorer = TopExp_Explorer(read_step(path), TopAbs_SOLID)
    source = compound(explorer.Current())
    edit = prepare_hole_edit(source, merge_same_domain=True)
    model = edit.model
    radii = (model.reference_radii + model.upper_radii) / 2
    rebuilt = edit.rebuild(radii)
    heights = np.array([hole.height for hole in edit.holes])
    removed = np.pi * np.sum((radii**2 - model.reference_radii**2) * heights)
    before = cad_metrics(source)[0][0]
    assert cad_metrics(rebuilt)[0][0] == pytest.approx(before - removed, rel=1e-6)


def test_rebuild_fallback_rejects_change_between_edited_bores():
    shape, *_ = stock()
    working, _ = normalize_shape(shape, CadFrame(1.0, (0.0, 0.0, 0.0)))
    holes, _, _ = recognize_holes(working)
    radii = np.array([hole.radius + 0.1 for hole in holes])
    shifts = np.zeros((len(holes), 3))
    rebuilt = rebuild_holes(working, holes, radii, shifts)
    assert _local_edit_error(working, rebuilt, holes, radii, shifts) is None
    # Between the bores at x=10 and x=28, inside the box spanning both but
    # outside the box of either one; 0.008 exceeds 1e-7 of the volume.
    notched = BRepAlgoAPI_Cut(
        rebuilt, BRepPrimAPI_MakeBox(gp_Pnt(18.9, 11.9, 5.8), 0.2, 0.2, 0.2).Shape()
    ).Shape()
    assert _local_edit_error(working, notched, holes, radii, shifts) == (
        "Rebuilt CAD changed material away from the edited holes"
    )


def nist_solid(name):
    path = Path(__file__).resolve().parents[2] / "fixtures" / "nist" / name
    if not path.exists():
        pytest.skip("NIST fixture not available")
    return compound(TopExp_Explorer(read_step(path), TopAbs_SOLID).Current())


def test_reference_measures_are_the_cad_measures():
    shape, *_ = stock()
    model = ordered_edit(shape).model
    volume, area = GProp_GProps(), GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, volume, 1e-9)
    BRepGProp.SurfaceProperties_s(shape, area, 1e-9)
    center = volume.CentreOfMass()
    reference = model.reference_radii
    # The laws start from the measured solid, not from a discretized copy.
    assert float(model.volume(reference)) == pytest.approx(volume.Mass(), rel=1e-12)
    assert float(model.surface_area(reference)) == pytest.approx(area.Mass(), rel=1e-12)
    np.testing.assert_allclose(
        model.center_of_mass(reference), [center.X(), center.Y(), center.Z()], atol=1e-9
    )


@pytest.mark.parametrize("kind", ["rectangular", "round", "orthogonal"])
def test_mesh_matches_analytic_values_within_its_discretization(kind):
    shape, *_ = stock(kind)
    edit = ordered_edit(shape)
    mesh = prepare_hole_mesh(edit, deflection=0.000025)
    radii = jnp.array([2.5, 2.4])
    expected, derivative, center, center_derivative = oracle(kind, np.asarray(radii))
    np.testing.assert_allclose(jax.jit(metrics)(mesh, radii), expected, rtol=1e-5)
    np.testing.assert_allclose(
        jax.jit(jax.jacrev(metrics, argnums=1))(mesh, radii), derivative, rtol=1e-4
    )
    np.testing.assert_allclose(mesh.center_of_mass(radii), center, atol=1e-5)
    np.testing.assert_allclose(
        jax.jacrev(mesh.center_of_mass)(radii), center_derivative, rtol=2e-4, atol=1e-7
    )
    model = edit.model
    trials = jnp.stack([model.reference_radii, model.lower_radii, radii])
    assert np.asarray(jax.jit(jax.vmap(mesh.is_valid))(trials)).all()
    for triangles in jax.jit(jax.vmap(mesh.triangles))(trials):
        assert_closed(triangles)
    fresh, _ = triangulate_shape(edit.rebuild(radii), deflection=0.000025)
    np.testing.assert_allclose(
        [divergence_volume(fresh), mesh_surface_area(fresh)], expected, rtol=1e-5
    )
    shifts = jnp.array([[0.2, -0.3, 0.0], [-0.25, 0.2, 0.0]])
    if kind == "orthogonal":
        shifts = shifts.at[1].set(jnp.array([0.0, 0.2, -0.25]))
    moved = jnp.array([2.3, 2.8])
    expected, _, center, _ = oracle(kind, np.asarray(moved), shifts)
    values = jax.jit(position_metrics)(mesh, moved, shifts)
    np.testing.assert_allclose(values[:2], expected, rtol=1e-5)
    np.testing.assert_allclose(values[2:], center, atol=1e-5)
    assert_closed(mesh.triangles(moved, shifts))


@pytest.mark.parametrize("scale", [0.001, 1000.0])
def test_mesh_units_and_placement(scale):
    shape, *_ = stock("rectangular")
    transform = gp_Trsf()
    transform.SetRotation(gp_Ax1(gp_Pnt(), gp_Dir(1, 2, 3)), 0.7)
    rotation = np.array([[transform.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])
    origin = np.array([1000.0, -2000.0, 3000.0])
    transform.SetValues(*np.column_stack([scale * rotation, origin]).reshape(-1))
    moved = BRepBuilderAPI_Transform(shape, transform, True, False).Shape()
    edit = ordered_edit(moved, frame=CadFrame(scale, tuple(origin)))
    mesh = prepare_hole_mesh(edit, deflection=0.000025)
    radii = scale * jnp.array([2.5, 2.4])
    values, derivative, center, _ = oracle("rectangular", np.array([2.5, 2.4]))
    np.testing.assert_allclose(
        jax.jit(metrics)(mesh, radii), values * [scale**3, scale**2], rtol=1e-5
    )
    np.testing.assert_allclose(
        jax.jacrev(metrics, argnums=1)(mesh, radii),
        derivative * np.array([scale**2, scale])[:, None],
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        (mesh.center_of_mass(radii) - origin) / scale, rotation @ center, atol=1e-5
    )


@pytest.mark.slow
def test_mesh_limits_do_not_restrict_the_edit():
    # FTC-08's end planes hold thin triangles that flip before the radius
    # bound, although the CAD edit is fine there.
    source = nist_solid("nist_ftc_08_asme1_rc.stp")
    edit = prepare_hole_edit(source, merge_same_domain=True)
    model = edit.model
    radii = (model.reference_radii + model.upper_radii) / 2
    assert bool(model.is_valid(radii))
    assert not bool(prepare_hole_mesh(edit).is_valid(radii))
    expected = float(model.volume(radii))
    rebuilt = edit.rebuild(radii)
    assert cad_metrics(rebuilt)[0][0] == pytest.approx(expected, rel=1e-6)


@pytest.mark.slow
def test_a_crack_away_from_the_holes_only_blocks_the_mesh():
    # FTC-07 has a real gap of about 5 CAD tolerances between a sphere and a
    # cylinder face, away from the recognized holes.
    source = nist_solid("nist_ftc_07_asme1_rd.stp")
    edit = prepare_hole_edit(source, merge_same_domain=True)
    with pytest.raises(ValueError, match="gap"):
        prepare_hole_mesh(edit)
    model = edit.model
    radii = (model.reference_radii + model.upper_radii) / 2
    rebuilt = edit.rebuild(radii)
    assert cad_metrics(rebuilt)[0][0] == pytest.approx(
        float(model.volume(radii)), rel=1e-6
    )


def test_thin_material_inside_a_bore_is_refused_twice():
    # A peg this thin passes the bore emptiness check, which compares volumes.
    # It has to cross an opening to get there, and that is what recognition sees.
    bare = BRepAlgoAPI_Cut(
        BRepPrimAPI_MakeBox(40, 24, 6).Shape(), cylinder((10, 12, 0), (0, 0, 1), 3, 6)
    ).Shape()
    plate = bare
    for corner, size in (((3, 10, 6), (2, 4, 4)), ((3, 10, 8), (9, 4, 2))):
        box = BRepPrimAPI_MakeBox(gp_Pnt(*corner), *size).Shape()
        plate = BRepAlgoAPI_Fuse(plate, box).Shape()
    plate = BRepAlgoAPI_Fuse(
        plate, cylinder((10.8, 12, 1), (0, 0, 1), 0.0001, 8)
    ).Shape()
    frame = CadFrame(1.0, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="touches the hole opening"):
        prepare_hole_edit(plate, frame=frame)
    # Filling the bore would erase it, and only the surface area shows the loss.
    hole = recognize_holes(normalize_shape(bare, frame)[0])[0][0]
    working = normalize_shape(plate, frame)[0]
    with pytest.raises(RuntimeError, match="surface area"):
        rebuild_holes(working, (hole,), np.array([3.1]), np.zeros((1, 3)))


def wall_distance(shape, base, axis, radius):
    """Distance from one rebuilt hole wall to all but its own end faces."""
    faces = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, faces)
    faces = [TopoDS.Face_s(faces.FindKey(i + 1)) for i in range(faces.Extent())]
    axis = np.asarray(axis)
    walls = []
    for index, face in enumerate(faces):
        adaptor = BRepAdaptor_Surface(face)
        if adaptor.GetType() != GeomAbs_Cylinder:
            continue
        surface = adaptor.Cylinder()
        location = surface.Location()
        offset = np.array([location.X(), location.Y(), location.Z()]) - base
        if (
            abs(surface.Radius() - radius) < 1e-9
            and np.linalg.norm(offset - (offset @ axis) * axis) < 1e-9
        ):
            walls.append(index)
    assert len(walls) == 1
    wall = faces[walls[0]]
    graph = build_adjacency_graph(shape)
    skip = {walls[0], *neighbors(graph, walls[0])}
    return min(
        BRepExtrema_DistShapeShape(wall, face).Value()
        for index, face in enumerate(faces)
        if index not in skip
    )


def two_hole_plate():
    shape = BRepPrimAPI_MakeBox(40, 24, 6).Shape()
    for base, radius in (((10, 12, 0), 2), ((22, 12, 0), 3)):
        shape = BRepAlgoAPI_Cut(shape, cylinder(base, (0, 0, 1), radius, 6)).Shape()
    return shape


def test_ligament_bounds_are_reached_when_holes_grow():
    edit = ordered_edit(two_hole_plate(), frame=CadFrame(1.0, (0.0, 0.0, 0.0)))
    model = edit.model
    reference = np.asarray(model.reference_radii)
    # The holes face each other 7 apart; the nearest wall is 8 from the first.
    assert np.asarray(model.ligament_bounds(model.reference_radii)) == pytest.approx(
        [7.0, 7.0]
    )
    radii = jnp.asarray(reference + np.array([0.3, 0.5]))
    assert np.asarray(model.ligament_bounds(radii)) == pytest.approx([6.2, 6.2])
    rebuilt = edit.rebuild(radii)
    for hole, bound, radius in zip(
        edit.holes, model.ligament_bounds(radii), radii, strict=True
    ):
        measured = wall_distance(rebuilt, np.asarray(hole.base), hole.axis, radius)
        assert measured == pytest.approx(float(bound), abs=1e-9)


def test_only_selected_holes_count_as_moving():
    shape = two_hole_plate()
    frame = CadFrame(1.0, (0.0, 0.0, 0.0))
    first = ordered_edit(shape, frame=frame).holes[0]
    edit = prepare_hole_edit(shape, [first.face_id], frame=frame)
    # The unselected hole is a fixed face, so only this hole's travel counts.
    radii = edit.model.reference_radii + 0.5
    assert np.asarray(edit.model.ligament_bounds(radii)) == pytest.approx([6.5])


@pytest.mark.parametrize(
    "radii_shift, shifts",
    [
        ((-0.5, -0.5), ((0, 0, 0), (0, 0, 0))),
        ((0.0, 0.0), ((1.0, 0, 0), (-1.0, 0, 0))),
        ((0.0, 0.0), ((-1.0, 0, 0), (1.0, 0, 0))),
        ((0.2, -0.4), ((0.3, 0.4, 0), (0, -0.5, 0))),
    ],
)
def test_ligament_bounds_never_exceed_measured_distances(radii_shift, shifts):
    edit = ordered_edit(two_hole_plate(), frame=CadFrame(1.0, (0.0, 0.0, 0.0)))
    radii = edit.model.reference_radii + jnp.asarray(radii_shift)
    moves = jnp.asarray(shifts, dtype=jnp.float64)
    assert bool(edit.model.is_valid(radii, moves))
    bounds = np.asarray(edit.model.ligament_bounds(radii, moves))
    rebuilt = edit.rebuild(radii, moves)
    for hole, bound, radius, move in zip(
        edit.holes, bounds, np.asarray(radii), np.asarray(moves), strict=True
    ):
        base = np.asarray(hole.base) + move
        assert wall_distance(rebuilt, base, hole.axis, radius) >= bound - 1e-9


def test_ligament_bound_derivatives_match_the_closest_pair():
    edit = ordered_edit(two_hole_plate(), frame=CadFrame(1.0, (0.0, 0.0, 0.0)))
    model = edit.model
    radii = model.reference_radii + jnp.array([0.1, 0.2])
    jacobian = jax.jacfwd(model.ligament_bounds)(radii)
    assert np.asarray(jacobian) == pytest.approx(np.array([[-1.0, -1.0], [-1.0, -1.0]]))
    # No NaN from the shift norm at the default zero shift, under jit and vmap.
    shift_gradient = jax.grad(
        lambda s: model.ligament_bounds(radii, s)[0],
    )(jnp.zeros((2, 3)))
    assert np.all(np.isfinite(shift_gradient))
    batch = jnp.stack([radii, model.reference_radii])
    assert np.asarray(
        jax.vmap(jax.jit(lambda r: model.ligament_bounds(r)))(batch)
    ) == pytest.approx(np.array([[6.7, 6.7], [7.0, 7.0]]))


def plate_edit():
    return ordered_edit(two_hole_plate(), frame=CadFrame(1.0, (0.0, 0.0, 0.0)))


def assert_matches_cad(model, rebuilt, radii):
    (volume, area), center = cad_metrics(rebuilt)
    np.testing.assert_allclose(model.volume(radii), volume, rtol=1e-9)
    np.testing.assert_allclose(model.surface_area(radii), area, rtol=1e-9)
    np.testing.assert_allclose(model.center_of_mass(radii), center, atol=1e-8)


def test_one_hole_can_grow_until_the_contact_margin():
    edit = plate_edit()
    model = edit.model
    margin = float(model.contact_margin)
    assert margin == pytest.approx(3e-5)  # 30 times the 1e-6 tolerance floor.
    # The holes face each other 7 apart, which is the nearer feature for both.
    radii = model.reference_radii.at[0].set(model.upper_radii[0])
    assert float(radii[0]) == pytest.approx(2 + 7 - margin)
    assert bool(model.is_valid(radii))
    rebuilt = edit.rebuild(radii)
    assert_matches_cad(model, rebuilt, radii)
    base = np.asarray(edit.holes[0].base)
    distance = wall_distance(rebuilt, base, edit.holes[0].axis, radii[0])
    assert distance == pytest.approx(margin, abs=1e-9)


def test_two_holes_share_the_distance_between_them():
    edit = plate_edit()
    model = edit.model
    reference = model.reference_radii
    margin = float(model.contact_margin)
    # Each is fine alone, and both together would meet.
    assert not bool(model.is_valid(model.upper_radii))
    assert np.all(np.isnan(model.volume(model.upper_radii)))
    with pytest.raises(ValueError, match="Invalid radii"):
        edit.rebuild(model.upper_radii)
    apart = reference + jnp.array([5.0, 1.9])
    assert bool(model.is_valid(apart))
    rebuilt = edit.rebuild(apart)
    assert_matches_cad(model, rebuilt, apart)
    base = np.asarray(edit.holes[1].base)
    distance = wall_distance(rebuilt, base, edit.holes[1].axis, apart[1])
    assert distance == pytest.approx(0.1, abs=1e-9)
    assert bool(model.is_valid(reference + jnp.array([5.0, 2.0 - 2 * margin])))
    assert not bool(model.is_valid(reference + jnp.array([5.0, 2.0])))
    # Any point halfway to the single-hole limits is valid together.
    assert bool(model.is_valid((reference + model.upper_radii) / 2))


def test_shifts_share_the_distance_between_holes():
    edit = plate_edit()
    model = edit.model
    radii = model.reference_radii
    # The first hole sits at lower x, so these approach each other.
    toward = jnp.array([[3.6, 0.0, 0.0], [-3.4 + 0.1, 0.0, 0.0]])
    assert bool(model.is_valid(radii, toward))
    assert not bool(model.is_valid(radii, toward.at[1, 0].set(-3.4)))
    rebuilt = edit.rebuild(radii, toward)
    actual, center = cad_metrics(rebuilt)
    np.testing.assert_allclose(model.volume(radii, toward), actual[0], rtol=1e-9)
    np.testing.assert_allclose(model.center_of_mass(radii, toward), center, atol=1e-8)


def test_a_shrinking_hole_leaves_its_neighbor_the_whole_distance():
    edit = plate_edit()
    model = edit.model
    radii = model.upper_radii.at[0].set(model.lower_radii[0])
    assert bool(model.is_valid(radii))
    assert_matches_cad(model, edit.rebuild(radii), radii)


def plate_with_a_peg_above_its_hole(gap):
    # A radius-3 peg over a radius-5 bore, held by a bridge and a pillar.
    shape = BRepPrimAPI_MakeBox(40, 24, 6).Shape()
    shape = BRepAlgoAPI_Cut(shape, cylinder((20, 12, 0), (0, 0, 1), 5, 6)).Shape()
    for part in (
        BRepPrimAPI_MakeBox(gp_Pnt(29, 8, 6), 5, 8, 6).Shape(),
        BRepPrimAPI_MakeBox(gp_Pnt(18, 10, 10), 16, 4, 2).Shape(),
        cylinder((20, 12, 6 + gap), (0, 0, 1), 3, 6 - gap),
    ):
        shape = BRepAlgoAPI_Fuse(shape, part).Shape()
    return shape


def test_material_resting_on_an_opening_is_refused():
    # Filling part of the opening would join the peg, which no law describes.
    frame = CadFrame(1.0, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="No supported") as caught:
        prepare_hole_edit(plate_with_a_peg_above_its_hole(0.0), frame=frame)
    assert "touches the hole opening" in str(caught.value)


def test_a_tip_near_the_center_of_an_opening_is_refused():
    # Nearly on the opening, yet slightly more than one radius from the wall.
    gap = 2.5e-7
    shape = BRepAlgoAPI_Cut(
        BRepPrimAPI_MakeBox(40, 24, 6).Shape(), cylinder((20, 12, 0), (0, 0, 1), 5, 6)
    ).Shape()
    tip = BRepPrimAPI_MakeCone(
        gp_Ax2(gp_Pnt(20, 12, 6 + gap), gp_Dir(0, 0, 1)), 0, 3e-7, 4 - gap
    ).Shape()
    for part in (
        BRepPrimAPI_MakeBox(gp_Pnt(29, 8, 6), 5, 8, 6).Shape(),
        BRepPrimAPI_MakeBox(gp_Pnt(18, 10, 10), 16, 4, 2).Shape(),
        tip,
    ):
        shape = BRepAlgoAPI_Fuse(shape, part).Shape()
    with pytest.raises(ValueError, match="touches the hole opening"):
        prepare_hole_edit(shape, frame=CadFrame(1.0, (0.0, 0.0, 0.0)))


def test_material_clear_of_an_opening_leaves_the_hole_editable():
    frame = CadFrame(1.0, (0.0, 0.0, 0.0))
    edit = prepare_hole_edit(plate_with_a_peg_above_its_hole(0.5), frame=frame)
    model = edit.model
    radii = jnp.array([2.5])
    moved = jnp.array([[1.0, 0.0, 0.0]])
    assert bool(model.is_valid(radii, moved))
    (volume, area), center = cad_metrics(edit.rebuild(radii, moved))
    np.testing.assert_allclose(model.volume(radii, moved), volume, rtol=1e-9)
    np.testing.assert_allclose(model.surface_area(radii, moved), area, rtol=1e-9)
    np.testing.assert_allclose(model.center_of_mass(radii, moved), center, atol=1e-8)


def test_rebuild_refuses_holes_that_met_even_past_the_gate():
    # The range check keeps edits from here. This is what stands behind it:
    # the hole breaks into its neighbor, which the isolated-hole laws cannot
    # describe, and reconstruction must not return that solid as the edit.
    edit = plate_edit()
    radii = np.array(edit.model.reference_radii)
    radii[0] += 7.2
    with pytest.raises(RuntimeError, match="isolated-hole edit"):
        rebuild_holes(
            edit.working_shape, edit.working_holes[:1], radii[:1], np.zeros((1, 3))
        )


def test_the_mesh_range_is_narrower_than_the_edit():
    # The mesh's planar annulus only reaches 0.4*clearance. On this plate a
    # wall moving further folds it, while the CAD edit is fine there.
    edit = plate_edit()
    model = edit.model
    mesh = prepare_hole_mesh(edit)
    radii = model.reference_radii.at[0].set(model.upper_radii[0])
    assert bool(model.is_valid(radii))
    assert not bool(mesh.is_valid(radii))
    assert np.isnan(np.asarray(mesh.triangles(radii))).all()


def test_ligament_bounds_are_nan_for_invalid_edits():
    model = ordered_edit(two_hole_plate()).model
    assert np.all(np.isnan(model.ligament_bounds(model.upper_radii * 1.5)))


@pytest.mark.slow
@pytest.mark.parametrize(
    "name", ["nist_ftc_07_asme1_rd.stp", "nist_stc_07_asme1_ap242-e3.stp"]
)
def test_each_hole_rebuilds_at_its_limit_on_a_part_with_a_large_tolerance(name):
    # A check envelope landing on a neighboring wall once refused these edits.
    edit = prepare_hole_edit(nist_solid(name), merge_same_domain=True)
    model = edit.model
    for i in range(len(edit.holes)):
        radii = model.reference_radii.at[i].set(model.upper_radii[i])
        assert bool(model.is_valid(radii))
        volume = cad_metrics(edit.rebuild(radii))[0][0]
        assert volume == pytest.approx(float(model.volume(radii)), rel=1e-6)


@pytest.mark.slow
def test_ligament_bounds_hold_on_an_imported_part():
    edit = prepare_hole_edit(nist_solid("nist_ctc_01_asme1_rd.stp"))
    model = edit.model
    radii = (model.reference_radii + model.upper_radii) / 2
    assert bool(model.is_valid(radii))
    bounds = np.asarray(model.ligament_bounds(radii))
    assert np.all(bounds > 0)
    rebuilt = edit.rebuild(radii)
    for hole, bound, radius in zip(edit.holes, bounds, np.asarray(radii), strict=True):
        measured = wall_distance(rebuilt, np.asarray(hole.base), hole.axis, radius)
        assert measured >= bound - 1e-6 * float(model.scale)


def test_gradient_of_a_caller_response_matches_rebuilt_cad():
    # A response with no law of its own, built on the volume and ligament laws.
    # Its derivative must equal the finite difference through rebuilt CAD.
    shape = two_hole_plate()
    edit = ordered_edit(shape, frame=CadFrame(1.0, (0.0, 0.0, 0.0)))
    model = edit.model

    def combine(volume, ligaments):
        return volume / 5000.0 + jnp.sum(1.0 / ligaments**2)

    def from_laws(radii):
        return combine(model.volume(radii), model.ligament_bounds(radii))

    def from_cad(radii):
        rebuilt = edit.rebuild(jnp.asarray(radii))
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(rebuilt, props, 1e-9)
        ligaments = [
            wall_distance(rebuilt, np.asarray(hole.base), hole.axis, radius)
            for hole, radius in zip(edit.holes, radii, strict=True)
        ]
        return float(combine(props.Mass(), jnp.asarray(ligaments)))

    radii = np.asarray(model.reference_radii) + np.array([0.3, 0.4])
    gradient = np.asarray(jax.grad(from_laws)(jnp.asarray(radii)))
    assert from_cad(radii) == pytest.approx(float(from_laws(jnp.asarray(radii))))
    step = 1e-3
    for i in range(2):
        move = np.zeros(2)
        move[i] = step
        difference = (from_cad(radii + move) - from_cad(radii - move)) / (2 * step)
        assert gradient[i] == pytest.approx(difference, rel=1e-5)
