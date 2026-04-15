"""Thin wrapper around cadquery-ocp-novtk providing the OCCT subset BRepAX needs.

All OCP imports are centralized here so that swapping the underlying
OCCT binding (e.g., to a custom pybind11 wrapper) requires changing
only this module.
"""
