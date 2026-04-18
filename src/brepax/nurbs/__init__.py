"""B-spline surface evaluation and differentiable SDF computation."""

from brepax.nurbs.evaluate import evaluate_surface, evaluate_surface_derivs
from brepax.nurbs.projection import closest_point
from brepax.nurbs.sdf import bspline_sdf
from brepax.nurbs.trim import signed_distance_polygon, trim_indicator

__all__ = [
    "bspline_sdf",
    "closest_point",
    "evaluate_surface",
    "evaluate_surface_derivs",
    "signed_distance_polygon",
    "trim_indicator",
]
