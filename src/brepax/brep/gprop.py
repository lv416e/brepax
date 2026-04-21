"""OCCT reference quantities for volume, surface area, and inertia.

Internal module. Used by benchmarks and integration tests to anchor
BRepAX differentiable metric outputs against OCCT's analytic surface-integral
quadrature on the exact B-Rep, avoiding any mesh discretization error.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from brepax._occt.backend import BRepGProp, GProp_GProps
from brepax._occt.types import TopoDS_Shape


class GPropGroundTruth(TypedDict):
    """OCCT reference quantities for a solid shape.

    All values are in the shape's native unit system (mm for STEP
    imports by default). Moment of inertia is computed about the
    center of mass, following OCCT's convention.
    """

    volume: float
    surface_area: float
    center_of_mass: NDArray[np.float64]
    moment_of_inertia: NDArray[np.float64]


def compute_gprop_ground_truth(shape: TopoDS_Shape) -> GPropGroundTruth:
    """Compute OCCT reference volume, area, CoM, and inertia for a shape.

    OCCT's BRepGProp uses analytic surface-integral quadrature over the
    exact B-Rep representation, so these values have no mesh discretization
    error and serve as the anchor for validating BRepAX's differentiable
    metric outputs.

    Args:
        shape: A solid or compound. Non-solid shapes yield zero volume.

    Returns:
        Dict with:

        - ``volume``: scalar, shape volume.
        - ``surface_area``: scalar, total surface area across all faces.
        - ``center_of_mass``: shape ``(3,)``, Cartesian coordinates.
        - ``moment_of_inertia``: shape ``(3, 3)``, about the center of mass.

    Examples:
        >>> from brepax.io.step import read_step
        >>> from brepax.brep.gprop import compute_gprop_ground_truth
        >>> shape = read_step("tests/fixtures/sample_box.step")
        >>> gt = compute_gprop_ground_truth(shape)
        >>> bool(gt["volume"] > 0)
        True
    """
    vol_props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, vol_props)

    surf_props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, surf_props)

    com_pnt = vol_props.CentreOfMass()
    center_of_mass = np.array(
        [com_pnt.X(), com_pnt.Y(), com_pnt.Z()],
        dtype=np.float64,
    )

    # gp_Mat uses 1-based indexing per OCCT C++ convention.
    inertia_mat = vol_props.MatrixOfInertia()
    moment_of_inertia = np.array(
        [[inertia_mat.Value(r, c) for c in (1, 2, 3)] for r in (1, 2, 3)],
        dtype=np.float64,
    )

    return GPropGroundTruth(
        volume=float(vol_props.Mass()),
        surface_area=float(surf_props.Mass()),
        center_of_mass=center_of_mass,
        moment_of_inertia=moment_of_inertia,
    )
