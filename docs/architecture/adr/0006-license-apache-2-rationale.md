# ADR-0006: License Apache 2.0 Rationale

## Status

Accepted

## Context

BRepAX needs an open-source license that encourages both academic and industrial adoption. The primary candidates are MIT and Apache 2.0.

## Decision

Apache License 2.0.

Key reasons:

- **Patent protection**: Apache 2.0 includes an explicit patent grant, protecting contributors and users from patent claims related to the algorithms implemented
- **Academic safety**: Collaborators from mathematics and CAD research communities get clear IP protection
- **Industrial adoption**: Companies have established legal frameworks for Apache 2.0 adoption
- **Ecosystem alignment**: JAX, Equinox, and many JAX ecosystem libraries use Apache 2.0

## Consequences

- Slightly more restrictive than MIT (patent retaliation clause)
- Well-understood by both academic and corporate legal teams
- Compatible with other Apache 2.0 and MIT licensed dependencies
