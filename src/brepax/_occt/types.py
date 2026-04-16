"""BRepAX-specific type aliases for OCCT entities.

Provides a stable type vocabulary decoupled from the underlying
OCCT binding's class hierarchy.
"""

from OCP.TopoDS import (
    TopoDS_Edge,
    TopoDS_Face,
    TopoDS_Shape,
    TopoDS_Vertex,
)

__all__ = [
    "TopoDS_Edge",
    "TopoDS_Face",
    "TopoDS_Shape",
    "TopoDS_Vertex",
]
