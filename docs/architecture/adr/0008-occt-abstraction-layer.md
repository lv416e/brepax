# ADR-0008: OCCT Abstraction Layer

## Status

Accepted

## Context

BRepAX depends on `cadquery-ocp-novtk` for OCCT access (see ADR-0007). However, exposing `OCP.*` imports directly in BRepAX's public and internal code creates two risks:

1. **Brand coupling**: Users may perceive BRepAX as a CadQuery derivative rather than an independent library
2. **Binding lock-in**: If a better OCCT binding emerges (or if we build a minimal custom one), replacing cadquery-ocp would require changes throughout the codebase

## Decision

All OCCT access within BRepAX goes through `brepax._occt.backend`, a thin wrapper module that re-exports the OCCT subset BRepAX needs.

- `brepax._occt.backend`: Centralized OCP imports and re-exports
- `brepax._occt.types`: BRepAX-specific type aliases decoupled from OCP class hierarchy
- No BRepAX source file outside `_occt/` may contain `import OCP.*`

## Consequences

- Swapping the OCCT binding requires changing only `_occt/backend.py` and `_occt/types.py`
- Users never see CadQuery or OCP in BRepAX's public API
- Slight indirection cost (negligible for the non-hot-path operations OCCT handles)
- The abstraction layer must be maintained as OCCT API evolves
