"""BRepAX: Differentiable rasterizer for CAD Boolean operations."""

from brepax._version import __version__
from brepax.compilation_cache import enable_compilation_cache

__all__ = ["__version__", "enable_compilation_cache"]
