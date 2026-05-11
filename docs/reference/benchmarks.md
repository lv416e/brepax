# Integration benchmark report

A reproducible measurement of BRepAX's volume paths and face-level
metric coverage on the project's standard STEP fixture set.

The full report lives at
[`benchmarks/integration_report/REPORT.md`](https://github.com/lv416e/brepax/blob/main/benchmarks/integration_report/REPORT.md).
It is the output of one command:

```bash
uv run python -m benchmarks.integration_report.run_benchmark
```

## What it measures

- **Volume accuracy** across five BRepAX paths versus OCCT
  `BRepGProp` (analytic surface-integral quadrature):
  - `divergence_volume` (mesh divergence theorem on BRepMesh)
  - `mesh_sdf` (mesh signed distance + sigmoid grid integration)
  - `gwn` (generalized winding number indicator + grid integration)
  - `DifferentiableCSGStump.volume` (analytical primitive DNF + sigmoid)
  - `TrimmedCSGStump.volume` (same DNF, **bit-identical** to
    `DifferentiableCSGStump.volume` per
    [ADR-0019](../architecture/adr/0019-marschner-composition-scope.md)
    and [ADR-0020](../architecture/adr/0020-marschner-bspline-csg-scope.md))
- **Per-face metric coverage** for the four metrics shipped with
  this release line:
  `surface_area_per_face`, `min_draft_angle_per_face`,
  `mean_curvature_per_face`, `min_wall_thickness_per_face`.
- **Qualitative competitor landscape** versus Manifold,
  PyTorch3D, and JAX-FEM. The landscape is feature-coverage only,
  not a numerical head-to-head; cells are marked "hypothesis"
  where they reflect best-effort reading of public documentation.

## What it does not claim

- No "first" or "unique" claim about any BRepAX path.
- No quantitative speed comparison against external systems.
- No claim that BRepAX outperforms any external system on tasks
  those systems were designed for.

See
[`benchmarks/integration_report/competitor_landscape.md`](https://github.com/lv416e/brepax/blob/main/benchmarks/integration_report/competitor_landscape.md)
for the explicit framing and its limitations.
