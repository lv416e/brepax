"""STEP file I/O via the OCCT abstraction layer."""

from __future__ import annotations

from pathlib import Path

from brepax._occt.backend import IFSelect_RetDone, STEPControl_Reader
from brepax._occt.types import TopoDS_Shape


def read_step(path: str | Path) -> TopoDS_Shape:
    """Read a STEP file and return the top-level shape.

    Args:
        path: Path to the STEP file (.step or .stp).

    Returns:
        The parsed OCCT shape.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the STEP file cannot be parsed.
    """
    resolved = Path(path).resolve()
    if not resolved.is_file():
        msg = f"STEP file not found: {resolved}"
        raise FileNotFoundError(msg)

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(resolved))
    if status != IFSelect_RetDone:
        msg = f"Failed to read STEP file (status={status}): {resolved}"
        raise ValueError(msg)

    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        msg = f"STEP file produced a null shape: {resolved}"
        raise ValueError(msg)

    return shape


__all__ = ["read_step"]
