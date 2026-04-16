"""Experimental application demonstrations.

Showcases built on top of BRepAX primitives and Boolean operations.
APIs here are subject to change without notice.
"""

from brepax.experimental.applications.mold_direction import (
    MoldDirectionResult,
    optimize_mold_direction,
    undercut_volume,
)

__all__ = [
    "MoldDirectionResult",
    "optimize_mold_direction",
    "undercut_volume",
]
