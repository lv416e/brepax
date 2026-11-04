"""Independent CAD oracles for planar face offsets."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt, gp_Trsf, gp_Vec
from OCP.GProp import GProp_GProps

from brepax.experimental.applications.cad_frame import CadFrame
from brepax.experimental.applications.face_offset import (
    prepare_face_offset,
    recognize_face_offsets,
)
from brepax.io.step import read_step

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
UNIT = CadFrame(1.0, (0.0, 0.0, 0.0))


def box(corner, size):
    return BRepPrimAPI_MakeBox(gp_Pnt(*corner), *size).Shape()


def boss_block(height=4.0):
    """40 x 24 x 6 block with a 10 x 10 boss of the given height on top."""
    return BRepAlgoAPI_Fuse(
        box((0, 0, 0), (40, 24, 6)), box((5, 7, 6), (10, 10, height))
    ).Shape()


def pocket_block(depth=3.0):
    """40 x 24 x 6 block with a 10 x 10 pocket of the given depth."""
    return BRepAlgoAPI_Cut(
        box((0, 0, 0), (40, 24, 6)), box((20, 7, 6 - depth), (10, 10, depth + 1))
    ).Shape()


def mass(shape):
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props, 1e-9)
    center = props.CentreOfMass()
    return props.Mass(), np.array([center.X(), center.Y(), center.Z()])


def face_at(shape, normal, height, frame=UNIT):
    faces, _ = recognize_face_offsets(shape, frame=frame)
    matches = [
        face
        for face in faces
        if np.allclose(face.normal, normal) and np.isclose(face.centroid[2], height)
    ]
    assert len(matches) == 1
    return matches[0]


def test_boss_top_interval_law_and_rebuild_match_independent_cad():
    frame = UNIT
    shape = boss_block()
    top = face_at(shape, (0, 0, 1), 10.0, frame)
    # Pushing in ends where the boss walls end, at the block top.
    assert top.lower == pytest.approx(-4.0, abs=1e-4)
    assert top.lower >= -4.0
    assert top.upper > 4.0
    assert top.area == pytest.approx(100.0)
    edit = prepare_face_offset(shape, [top.face_id], frame=frame)
    for offset in (-3.5, -1.0, 2.5):
        offsets = jnp.array([offset])
        volume, center = mass(boss_block(4.0 + offset))
        assert float(edit.model.volume(offsets)) == pytest.approx(volume, rel=1e-9)
        np.testing.assert_allclose(
            edit.model.center_of_mass(offsets), center, rtol=1e-9, atol=1e-9
        )
        assert float(jax.grad(lambda d: edit.model.volume(d)[()])(offsets)[0]) == (
            pytest.approx(100.0)
        )
        rebuilt_volume, rebuilt_center = mass(edit.rebuild(offsets))
        assert rebuilt_volume == pytest.approx(volume, rel=1e-9)
        np.testing.assert_allclose(rebuilt_center, center, rtol=1e-9, atol=1e-9)


def test_pocket_floor_interval_ends_at_the_pocket_and_block_bottom():
    frame = UNIT
    shape = pocket_block()
    floor = face_at(shape, (0, 0, 1), 3.0, frame)
    # Raising the floor ends at the block top, where the pocket walls end;
    # lowering it ends at the block bottom.
    assert floor.upper == pytest.approx(3.0, abs=1e-4)
    assert floor.upper <= 3.0
    assert floor.lower == pytest.approx(-3.0, abs=1e-4)
    edit = prepare_face_offset(shape, [floor.face_id], frame=frame)
    offsets = jnp.array([1.25])
    volume, center = mass(pocket_block(3.0 - 1.25))
    assert float(edit.model.volume(offsets)) == pytest.approx(volume, rel=1e-9)
    rebuilt_volume, rebuilt_center = mass(edit.rebuild(offsets))
    assert rebuilt_volume == pytest.approx(volume, rel=1e-9)
    np.testing.assert_allclose(rebuilt_center, center, rtol=1e-9, atol=1e-9)


def test_independent_faces_add_up_and_interacting_faces_are_refused():
    frame = UNIT
    shape = boss_block()
    faces, _ = recognize_face_offsets(shape, frame=frame)
    top = next(f for f in faces if np.allclose(f.normal, (0, 0, 1)) and f.area == 100)
    bottom = next(f for f in faces if np.allclose(f.normal, (0, 0, -1)))
    side = next(
        f for f in faces if np.allclose(f.normal, (1, 0, 0)) and f.centroid[0] < 20
    )
    edit = prepare_face_offset(shape, [top.face_id, bottom.face_id], frame=frame)
    offsets = jnp.array([1.0, 0.5])
    expected = mass(boss_block())[0] + 100.0 * 1.0 + 40 * 24 * 0.5
    assert float(edit.model.volume(offsets)) == pytest.approx(expected, rel=1e-12)
    assert mass(edit.rebuild(offsets))[0] == pytest.approx(expected, rel=1e-9)
    with pytest.raises(ValueError, match="may interact"):
        prepare_face_offset(shape, [top.face_id, side.face_id], frame=frame)


def test_out_of_interval_offsets_are_nan_and_refused():
    frame = UNIT
    shape = boss_block()
    top = face_at(shape, (0, 0, 1), 10.0, frame)
    edit = prepare_face_offset(shape, [top.face_id], frame=frame)
    beyond = jnp.array([top.lower - 0.1])
    assert not bool(edit.model.is_valid(beyond))
    assert np.isnan(float(edit.model.volume(beyond)))
    with pytest.raises(ValueError, match="Invalid offsets"):
        edit.rebuild(beyond)


def test_unsupported_faces_are_reported_with_reasons():
    _, rejected = recognize_face_offsets(read_step(FIXTURES / "box_with_holes.step"))
    reasons = {face.reason for face in rejected}
    assert "inner boundary loops" in reasons
    with pytest.raises(ValueError, match="cannot be offset"):
        prepare_face_offset(
            read_step(FIXTURES / "box_with_holes.step"),
            [next(f.face_id for f in rejected if f.reason == "inner boundary loops")],
        )


@pytest.mark.parametrize("scale", [0.001, 25.4])
def test_units_and_placement(scale):
    move = gp_Trsf()
    move.SetTranslation(gp_Vec(100.0, -50.0, 7.0))
    resize = gp_Trsf()
    resize.SetScale(gp_Pnt(0, 0, 0), scale)
    shape = BRepBuilderAPI_Transform(
        BRepBuilderAPI_Transform(boss_block(), resize, True).Shape(), move, True
    ).Shape()
    faces, _ = recognize_face_offsets(shape)
    top = max(
        (f for f in faces if np.allclose(f.normal, (0, 0, 1))),
        key=lambda f: f.centroid[2],
    )
    assert top.area == pytest.approx(100.0 * scale**2, rel=1e-9)
    # OCCT keeps absolute tolerances when scaling, so the 10-tolerance margin
    # is relatively larger on the shrunk part.
    assert top.lower == pytest.approx(-4.0 * scale, rel=1e-3)
    assert top.lower >= -4.0 * scale
    edit = prepare_face_offset(shape, [top.face_id])
    offsets = jnp.array([1.5 * scale])
    gradient = jax.grad(lambda d: edit.model.volume(d)[()])(offsets)
    assert float(gradient[0]) == pytest.approx(100.0 * scale**2, rel=1e-9)
    volume = mass(edit.rebuild(offsets))[0]
    assert volume == pytest.approx(float(edit.model.volume(offsets)), rel=1e-9)
