"""Independent CAD, remeshing, and STEP checks for an edited roof/hole solid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

from brepax._occt.types import TopoDS_Shape
from brepax.brep.triangulate import (
    divergence_volume,
    mesh_surface_area,
    triangulate_shape,
)
from brepax.experimental.applications.cad_frame import CadFrame
from brepax.io.step import read_step

from .geometry import LENGTH, WIDTH, bottom_landmark_error, cad_measures, make_shape

RELATIVE_TOLERANCE = 1e-4


@dataclass(frozen=True)
class Placement:
    """Transform applied to CAD before independent meshing, not to a saved mesh."""

    name: str
    scale: float
    origin: tuple[float, float, float]
    angle: float


PLACEMENTS = (
    Placement("identity", 1.0, (0.0, 0.0, 0.0), 0.0),
    Placement("rotated", 1.0, (3.0, -2.0, 4.0), 0.7),
    Placement("small_far", 0.001, (1000.0, -2000.0, 3000.0), 0.7),
    Placement("large", 1000.0, (3000.0, -2000.0, 4000.0), 0.7),
)


@dataclass
class GeometryCheck:
    """Completed measurements survive a later export or extraction failure."""

    placement: Placement
    passed: bool = False
    error: str | None = None
    cad_error: list[float] | None = None
    mesh_error: list[float] | None = None
    roundtrip_cad_error: list[float] | None = None
    roundtrip_mesh_error: list[float] | None = None
    landmark_error: float | None = None
    roundtrip_landmark_error: float | None = None
    triangles: int | None = None
    roundtrip_triangles: int | None = None
    step_file: str | None = None


def relative_errors(
    actual: NDArray[np.float64], expected: NDArray[np.float64]
) -> list[float]:
    """Compare positive volume/area targets; fail before recording nonfinite data."""
    errors = abs(actual / expected - 1)
    if not np.isfinite(errors).all():
        raise FloatingPointError("Nonfinite geometry comparison")
    return [float(error) for error in errors]


def remesh_measures(
    shape: TopoDS_Shape, frame: CadFrame, deflection: float
) -> tuple[NDArray[np.float64], int]:
    """Fresh external mesh, measured in working coordinates to avoid cancellation."""
    triangles, _ = triangulate_shape(shape, deflection=deflection * frame.length_scale)
    working = (triangles - jnp.asarray(frame.origin)) / frame.length_scale
    values = np.asarray([divergence_volume(working), mesh_surface_area(working)])
    return values, len(triangles)


def verify_edit(
    dimensions: NDArray[np.float64],
    target: NDArray[np.float64],
    deflection: float,
    artifact_dir: Path,
) -> list[GeometryCheck]:
    """Check value invariance/unit scaling and placement, including STEP roundtrip.

    Meshing tolerance scales with external units. These checks establish scalar
    value accuracy for this family, not gradients through OCCT remeshing.
    """
    checks = []
    for placement in PLACEMENTS:
        check = GeometryCheck(placement)
        try:
            rotation = gp_Trsf()
            rotation.SetRotation(gp_Ax1(gp_Pnt(), gp_Dir(1, 2, 3)), placement.angle)
            matrix = np.array(
                [[rotation.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)]
            )
            frame = CadFrame(placement.scale, placement.origin)
            rotated = BRepBuilderAPI_Transform(
                make_shape(dimensions), rotation, True, False
            ).Shape()
            shape = frame.restore_shape(rotated)
            powers = np.array([placement.scale**3, placement.scale**2])
            corners = np.array(
                [[0, 0, 0], [LENGTH, 0, 0], [0, WIDTH, 0], [LENGTH, WIDTH, 0]]
            )
            landmarks = placement.scale * (corners @ matrix.T) + placement.origin
            check.cad_error = relative_errors(cad_measures(shape) / powers, target)
            values, check.triangles = remesh_measures(shape, frame, deflection)
            check.mesh_error = relative_errors(values, target)
            check.landmark_error = bottom_landmark_error(
                shape, landmarks, frame.length_scale
            )

            artifact_dir.mkdir(parents=True, exist_ok=True)
            path = artifact_dir / f"{placement.name}.step"
            writer = STEPControl_Writer()
            if writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone:
                raise RuntimeError("STEP transfer failed")
            if writer.Write(str(path)) != IFSelect_RetDone:
                raise RuntimeError("STEP write failed")
            check.step_file = str(path)
            restored = read_step(path)
            if not BRepCheck_Analyzer(restored).IsValid():
                raise RuntimeError("Invalid CAD after STEP roundtrip")
            check.roundtrip_cad_error = relative_errors(
                cad_measures(restored) / powers, target
            )
            values, check.roundtrip_triangles = remesh_measures(
                restored, frame, deflection
            )
            check.roundtrip_mesh_error = relative_errors(values, target)
            check.roundtrip_landmark_error = bottom_landmark_error(
                restored, landmarks, frame.length_scale
            )
            check.passed = bool(
                max(
                    *check.cad_error,
                    *check.mesh_error,
                    *check.roundtrip_cad_error,
                    *check.roundtrip_mesh_error,
                )
                <= RELATIVE_TOLERANCE
                and max(check.landmark_error, check.roundtrip_landmark_error) <= 1e-6
            )
        except Exception as error:
            check.error = f"{type(error).__name__}: {error}"
        checks.append(check)
    return checks
