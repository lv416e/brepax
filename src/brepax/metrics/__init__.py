"""Differentiable geometric metrics computed from SDF grid integration."""

from brepax.metrics.draft_angle import (
    draft_angle_violation,
    integrate_sdf_draft_angle_violation,
)
from brepax.metrics.inertia import (
    center_of_mass,
    integrate_sdf_center_of_mass,
    integrate_sdf_moment_of_inertia,
    moment_of_inertia,
)
from brepax.metrics.surface_area import integrate_sdf_surface_area, surface_area
from brepax.metrics.wall_thickness import (
    integrate_sdf_min_wall_thickness,
    integrate_sdf_thin_wall_volume,
    min_wall_thickness,
    thin_wall_volume,
)

__all__ = [
    "center_of_mass",
    "draft_angle_violation",
    "integrate_sdf_center_of_mass",
    "integrate_sdf_draft_angle_violation",
    "integrate_sdf_min_wall_thickness",
    "integrate_sdf_moment_of_inertia",
    "integrate_sdf_surface_area",
    "integrate_sdf_thin_wall_volume",
    "min_wall_thickness",
    "moment_of_inertia",
    "surface_area",
    "thin_wall_volume",
]
