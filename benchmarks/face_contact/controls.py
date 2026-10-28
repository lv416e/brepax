"""Actual-CAD counterexamples for the selected-face preparation contract."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from brepax._occt.backend import (
    BRepAlgoAPI_Cut,
    BRepAlgoAPI_Fuse,
    BRepCheck_Analyzer,
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCylinder,
    gp_Ax2,
    gp_Dir,
    gp_Pnt,
)
from brepax._occt.types import TopoDS_Shape
from brepax.experimental.planar_contact import planar_contact_volume_direction
from brepax.experimental.planar_contact_cad import prepare_planar_contact_reference

from .cad import transform_shape
from .preparation import known_support_face_id

_Pair = tuple[TopoDS_Shape, TopoDS_Shape]


@dataclass(frozen=True)
class _Control:
    name: str
    build: Callable[[], _Pair]
    expected_error: str | None
    mode: Literal["zero_residual", "projected_nominal"] = "zero_residual"
    moving_normal: tuple[float, float, float] = (0, 0, -1)
    prescribed_ids: tuple[int, int] | None = None
    source_gap: float | None = None


def _box_pair(gap: float = 0, x: float = -1, y: float = -1) -> _Pair:
    return (
        BRepPrimAPI_MakeBox(gp_Pnt(-2, -2, -1), 4, 4, 1).Shape(),
        BRepPrimAPI_MakeBox(gp_Pnt(x, y, gap), 2, 2, 1).Shape(),
    )


def _curved_pair() -> _Pair:
    fixed, _ = _box_pair()
    cylinder = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0.5, 1
    ).Shape()
    return fixed, cylinder


def _cut_pair(hole: bool) -> _Pair:
    fixed, moving = _box_pair()
    if hole:
        cutter = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(0, 0, -1), gp_Dir(0, 0, 1)), 0.4, 3
        ).Shape()
    else:
        cutter = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, -1), 2, 2, 3).Shape()
    return fixed, BRepAlgoAPI_Cut(moving, cutter).Shape()


def _competing_pair(side: bool) -> _Pair:
    fixed, moving = _box_pair()
    if side:
        wall = BRepPrimAPI_MakeBox(gp_Pnt(1, -2, -1), 1, 4, 3).Shape()
        fixed = BRepAlgoAPI_Fuse(fixed, wall).Shape()
    else:
        slot = BRepPrimAPI_MakeBox(gp_Pnt(-0.5, -3, -0.5), 1, 6, 1).Shape()
        fixed = BRepAlgoAPI_Cut(fixed, slot).Shape()
    return fixed, moving


def _placed_pair() -> _Pair:
    cosine, sine = np.cos(0.37), np.sin(0.37)
    rotation = np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
    fixed, moving = _box_pair()
    return (
        transform_shape(fixed, rotation, (3, -2, 5)),
        transform_shape(moving, rotation, (3, -2, 5)),
    )


def _controls() -> list[_Control]:
    result = [_Control("axis_contact", _box_pair, None, source_gap=0)]
    for gap in (1e-3, 1e-8, 1e-16, -1e-8):
        result.append(
            _Control(
                f"source_gap_{gap:g}",
                partial(_box_pair, gap=gap),
                "Nonzero contact residual",
                source_gap=gap,
            )
        )
    result.extend(
        [
            _Control(
                "subroundoff_gap_explicit_nominal_reference",
                partial(_box_pair, gap=1e-16),
                None,
                "projected_nominal",
                source_gap=1e-16,
            ),
            _Control(
                "occt_subtolerance_gap_exceeds_roundoff",
                partial(_box_pair, gap=1e-8),
                "nominal roundoff bound",
                "projected_nominal",
                source_gap=1e-8,
            ),
            _Control("coplanar_nonoverlap", partial(_box_pair, x=3), "positive-area"),
            _Control("edge_only_overlap", partial(_box_pair, x=2), "positive-area"),
            _Control(
                "vertex_only_overlap", partial(_box_pair, x=2, y=2), "positive-area"
            ),
            _Control(
                "wrong_normal",
                _box_pair,
                "normals must oppose",
                moving_normal=(0, 0, 1),
            ),
            _Control("curved_solid", _curved_pair, "planar faces"),
            _Control("holed_face", partial(_cut_pair, True), "one wire"),
            _Control("nonconvex_face", partial(_cut_pair, False), "convex"),
            _Control(
                "competing_bottom_and_side", partial(_competing_pair, True), "convex"
            ),
            _Control(
                "disconnected_top_patches", partial(_competing_pair, False), "convex"
            ),
            _Control(
                "rigid_placement_zero_residual",
                _placed_pair,
                "Nonzero contact residual",
                prescribed_ids=(5, 4),
            ),
            _Control(
                "rigid_placement_nominal",
                _placed_pair,
                None,
                "projected_nominal",
                prescribed_ids=(5, 4),
            ),
        ]
    )
    return result


def collect_preparation_controls() -> dict[str, Any]:
    """Retain expected acceptance, rejection, residuals, and unexpected errors.

    The subroundoff positive-gap case intentionally accepts only a *different
    nominal reference*. Its original CAD has zero initial overlap derivative;
    the returned contact reference has rate four. This is not a successful
    derivative prediction for that source geometry.
    """
    rows = []
    for control in _controls():
        row: dict[str, Any] = {
            "case": control.name,
            "mode": control.mode,
            "expected_prepared": control.expected_error is None,
            "expected_error_contains": control.expected_error,
            "prescribed_source_gap": control.source_gap,
            "source_cad_derivative_validation": False,
        }
        phase = "build"
        try:
            fixed, moving = control.build()
            row["source_validity"] = [
                bool(BRepCheck_Analyzer(shape).IsValid()) for shape in (fixed, moving)
            ]
            phase = "select"
            ids = control.prescribed_ids or (
                known_support_face_id(fixed, (0, 0, 1), 0),
                known_support_face_id(moving, control.moving_normal),
            )
            phase = "prepare"
            prepared = prepare_planar_contact_reference(
                fixed,
                moving,
                fixed_face_id=ids[0],
                moving_face_id=ids[1],
                mode=control.mode,
            )
            row["diagnostics"] = asdict(prepared.diagnostics)
            row["prepared"] = True
            row["matched_expectation"] = control.expected_error is None
            if control.source_gap is not None and control.source_gap > 0:
                row["known_source_initial_derivative"] = 0.0
                row["nominal_reference_normal_approach_rate"] = float(
                    planar_contact_volume_direction(
                        prepared.patch,
                        jnp.array([0.0, 0, -1]),
                        jnp.zeros(3),
                        jnp.zeros(3),
                    )
                )
                # Acceptance is useful only if the original positive gap stays
                # visible and the changed target remains explicit.
                row["matched_expectation"] = bool(
                    row["matched_expectation"]
                    and prepared.diagnostics.signed_plane_gap > 0
                    and not prepared.diagnostics.original_cad_contact_certified
                )
        except Exception as error:
            row["prepared"] = False
            row["error"] = f"{type(error).__name__}: {error}"
            row["error_phase"] = phase
            row["matched_expectation"] = bool(
                phase == "prepare"
                and isinstance(error, ValueError)
                and control.expected_error is not None
                and control.expected_error in str(error)
            )
        rows.append(row)
    return {
        "rows": rows,
        "expected_cases": len(rows),
        "matched_expectations": sum(row["matched_expectation"] for row in rows),
        "passed": all(row["matched_expectation"] for row in rows),
    }
