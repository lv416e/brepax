"""Validate a local OCCT edit against the volume law of the edited features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from brepax._occt.backend import (
    Bnd_Box,
    BRepAlgoAPI_Common,
    BRepAlgoAPI_Cut,
    BRepBndLib,
    BRepBuilderAPI_Copy,
    BRepGProp,
    GProp_GProps,
    TopAbs_SOLID,
    TopExp_Explorer,
)
from brepax._occt.types import TopoDS_Shape


@dataclass(frozen=True)
class LocalEdit:
    """Where an edit may change material and what it must remove there.

    ``envelope`` contains every old and new feature boundary. ``zones`` are
    axis-aligned boxes, each of shape ``(2, 3)`` (lower and upper corners);
    every region of the difference between the solids must fit inside one.
    ``removed_volume`` and ``removed_moment`` are the volume and first moment
    the edit removes, negative where it adds material, and the bounds are
    their allowed absolute errors.
    """

    envelope: TopoDS_Shape
    zones: tuple[NDArray[np.float64], ...]
    removed_volume: float
    removed_moment: NDArray[np.float64]
    volume_bound: float
    moment_bound: float
    mismatch_message: str
    away_message: str


def volume_and_moment(shape: TopoDS_Shape) -> tuple[float, NDArray[np.float64]]:
    """Volume and first moment by adaptive integration to 1e-9 relative.

    The default fixed-order rule misintegrates rebuilt B-spline faces by
    about 1e-7 relative, which would hide or fake small material changes.
    """
    measured = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, measured, 1e-9)
    center = measured.CentreOfMass()
    return measured.Mass(), measured.Mass() * np.array(
        [center.X(), center.Y(), center.Z()]
    )


def local_edit_error(
    shape: TopoDS_Shape, changed: TopoDS_Shape, edit: LocalEdit
) -> str | None:
    """Check an edit inside its envelope, then the material outside it.

    Whole-solid properties can drift when a Boolean rebuilds faces far from
    the edit: by about 1e-7 relative under OCCT's default integration rule,
    and by up to about 1e-8 relative in geometry on the NIST parts measured,
    although nothing is edited there. Inside the envelope, the change in
    volume and first moment must match the edit within the given bounds.
    Outside it, volume and first moment may change by 1e-7 relative, the
    bound the whole-solid checks also use. A larger outside change passes
    only if every region of the difference between the two solids lies
    within one zone: a Boolean on a part with a large CAD tolerance can move
    faces within that tolerance, which changes measured properties without
    leaving a difference region. Each solid is intersected with the envelope
    separately: OCCT can return a wrong region when subtracting two nearly
    coincident solids, so the inside comparison never depends on that
    subtraction.
    """
    inside = []
    for solid in (shape, changed):
        builder = BRepAlgoAPI_Common(
            BRepBuilderAPI_Copy(solid, True, False).Shape(), edit.envelope
        )
        if not builder.IsDone():
            return "OCCT could not isolate the edit envelope"
        inside.append(volume_and_moment(builder.Shape()))
    if abs(
        inside[0][0] - inside[1][0] - edit.removed_volume
    ) > edit.volume_bound or np.any(
        np.abs(inside[0][1] - inside[1][1] - edit.removed_moment) > edit.moment_bound
    ):
        return edit.mismatch_message
    total = [volume_and_moment(solid) for solid in (shape, changed)]
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    solid_reach = float(np.abs(np.asarray(box.Get())).max())
    outside_volume = (total[0][0] - inside[0][0]) - (total[1][0] - inside[1][0])
    outside_moment = (total[0][1] - inside[0][1]) - (total[1][1] - inside[1][1])
    drift = 1e-7 * abs(total[0][0]) + 1e-9
    if abs(outside_volume) <= drift and np.all(
        np.abs(outside_moment) <= drift * solid_reach
    ):
        return None
    return _difference_outside_zones(shape, changed, edit)


def _difference_outside_zones(
    shape: TopoDS_Shape, changed: TopoDS_Shape, edit: LocalEdit
) -> str | None:
    """Reject a difference region that does not fit inside a single zone.

    Each region must fit in one zone: the box around several features
    together would also cover the material between them.
    """
    for minuend, subtrahend in ((shape, changed), (changed, shape)):
        builder = BRepAlgoAPI_Cut(
            BRepBuilderAPI_Copy(minuend, True, False).Shape(),
            BRepBuilderAPI_Copy(subtrahend, True, False).Shape(),
        )
        if not builder.IsDone():
            return "OCCT could not isolate the rebuilt regions"
        explorer = TopExp_Explorer(builder.Shape(), TopAbs_SOLID)
        while explorer.More():
            region = Bnd_Box()
            BRepBndLib.Add_s(explorer.Current(), region)
            lower, upper = np.asarray(region.Get()).reshape(2, 3)
            if not any(
                np.all(lower >= zone[0]) and np.all(upper <= zone[1])
                for zone in edit.zones
            ):
                return edit.away_message
            explorer.Next()
    return None
