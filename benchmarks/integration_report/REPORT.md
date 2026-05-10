# BRepAX Integration Benchmark Report

Reproducible measurement of the BRepAX volume paths and face-level metric coverage on the project's standard STEP fixture set.

**This report is the output of one benchmark command** — `uv run python -m benchmarks.integration_report.run_benchmark`. No claim of novelty is made beyond what the tables below directly show.  Comparisons against external systems (Manifold, PyTorch3D, JAX-FEM) are qualitative; see `competitor_landscape.md` for the framing and its limitations.

Volume paths use a sigmoid grid integration at `resolution=32` (per-axis cell count); the mesh divergence path uses the OCCT BRepMesh tessellation at the default deflection.  OCCT BRepGProp is the reference and is computed analytically on the exact B-Rep, not on the mesh.

## Volume accuracy

| Fixture | OCCT (ref) | divergence | div err | div t | CSG-Stump | csg err | csg t | TrimmedCSG | trim err | trim t | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `sample_box` | 6000.0000 | 6000.0000 | 0.00% | 0.74s | 5689.5962 | 5.17% | 2.02s | 5689.5962 | 5.17% | 0.95s |  |
| `sample_cylinder` | 1178.0972 | 1176.5602 | 0.13% | 0.90s | 1186.0311 | 0.67% | 1.35s | 1186.0311 | 0.67% | 1.89s |  |
| `sample_sphere` | 113.0973 | 112.5589 | 0.48% | 0.59s | 118.4262 | 4.71% | 0.31s | 118.4262 | 4.71% | 0.32s |  |
| `sample_cone` | 54.9779 | 54.6540 | 0.59% | 0.72s | 60.8456 | 10.67% | 1.02s | 60.8456 | 10.67% | 0.97s |  |
| `sample_torus` | 222.0661 | 220.6733 | 0.63% | 0.59s | 242.5633 | 9.23% | 0.02s | 242.5633 | 9.23% | 0.07s |  |
| `box_with_holes` | 22429.2037 | 22432.0938 | 0.01% | 0.90s | 20692.2324 | 7.74% | 0.38s | 20692.2324 | 7.74% | 0.40s |  |
| `box_with_pocket` | 23214.6018 | 23215.6230 | 0.00% | 0.70s | 20293.1797 | 12.58% | 0.17s | 20293.1797 | 12.58% | 0.20s |  |
| `box_with_slot` | 20800.0000 | 20800.0000 | 0.00% | 0.69s | 17470.7598 | 16.01% | 1.00s | 17470.7598 | 16.01% | 0.90s |  |
| `l_bracket` | 12000.0000 | 12000.0000 | 0.00% | 0.37s | 11025.0674 | 8.12% | 1.11s | 11025.0674 | 8.12% | 1.14s |  |
| `nurbs_box` | 480.0000 | 480.0000 | 0.00% | 0.42s | 481.2568 | 0.26% | 24.14s | 481.2568 | 0.26% | 40.38s |  |
| `nurbs_revol` | 40.8856 | — | — | 0.00s | — | — | 0.00s | — | — | 0.00s | div: shell (volume paths require closed solid) |
| `nurbs_saddle` | -0.0000 | — | — | 0.00s | — | — | 0.00s | — | — | 0.00s | div: shell (volume paths require closed solid) |

**How to read this table.**

- `divergence` is the mesh divergence-theorem volume (Stokes' theorem on the BRepMesh tessellation).  Differentiable through triangle vertex positions.  Strongest production path.
- `CSG-Stump` is the analytical primitive DNF, integrated with a sigmoid indicator.  Differentiable through primitive parameters.  Bounded by the BSpline half-space limitation (see `project_bspline_halfspace.md` in memory; concretely the CSG-Stump DNF cannot consume a finite trimmed BSpline patch as a half-space ingredient — ADR-0019, ADR-0020).
- `TrimmedCSGStump` carries per-face trim metadata for standalone trimmed-face SDF queries; on the DNF path it is **bit-equivalent** to `DifferentiableCSGStump` per ADR-0019 / ADR-0020.  Equality of the `csg` and `trim` columns is the expected outcome.

## Face-level metric coverage

Each cell shows `(finite / nan / inf)` counts out of the fixture's total face count.  Single-face shapes return `+inf` for `min_wall_thickness_per_face` (no other surface to measure against).  `mean_curvature_per_face` returns NaN on cone, torus, and BSpline faces (analytical handler not yet added).

| Fixture | n_faces | surface_area | min_draft_angle | mean_curvature | min_wall_thickness |
|---|---|---|---|---|---|
| `sample_box` | 6 | 6/0/0 | 6/0/0 | 6/0/0 | 6/0/0 |
| `sample_cylinder` | 3 | 3/0/0 | 3/0/0 | 3/0/0 | 3/0/0 |
| `sample_sphere` | 1 | 1/0/0 | 1/0/0 | 1/0/0 | 0/0/1 |
| `sample_cone` | 3 | 3/0/0 | 3/0/0 | 2/1/0 | 3/0/0 |
| `sample_torus` | 1 | 1/0/0 | 1/0/0 | 0/1/0 | 0/0/1 |
| `box_with_holes` | 8 | 8/0/0 | 8/0/0 | 8/0/0 | 8/0/0 |
| `box_with_pocket` | 8 | 8/0/0 | 8/0/0 | 8/0/0 | 8/0/0 |
| `box_with_slot` | 11 | 11/0/0 | 11/0/0 | 11/0/0 | 11/0/0 |
| `l_bracket` | 14 | 14/0/0 | 14/0/0 | 14/0/0 | 14/0/0 |
| `nurbs_box` | 6 | 6/0/0 | 6/0/0 | 0/6/0 | 6/0/0 |
| `nurbs_revol` | 0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| `nurbs_saddle` | 1 | 1/0/0 | 1/0/0 | 0/1/0 | 0/0/1 |

## Qualitative competitor landscape

**Important caveat.**  This is a *qualitative* feature-coverage table,
not a head-to-head numerical benchmark.  Numerical comparison would
require running the same problem (e.g. volume of a Linkrods-class
trimmed-BSpline shape with derivatives w.r.t. control points) through
each system, which none of the listed competitors are directly
designed to express.  The table reports what each system is built to
do, not who is faster on a shared benchmark.  Cells marked
"hypothesis" are best-effort observations from public documentation;
treat them as starting points, not citations.

| System / Path        | STEP / B-Rep input | Per-face awareness | Trimmed BSpline | Differentiable volume | Per-face metrics | Notes |
|---|---|---|---|---|---|---|
| BRepAX `divergence_volume`         | yes (via OCCT)       | derived from face triangulation | via OCCT BRepMesh        | yes — through triangle vertex positions          | yes (this PR's coverage table) | strongest production path on this report |
| BRepAX `evaluate_stump_volume`     | yes (via OCCT + reconstruction) | primitive-based                | weak (ADR-0020: BSpline patches are not a CSG-Stump DNF half-space ingredient) | yes — through primitive parameters               | not directly                  | analytical-dominant fixtures |
| BRepAX `TrimmedCSGStump.volume`    | yes                  | per-face trim metadata stored  | metadata stored, not on DNF SDF path (ADR-0019, ADR-0020) | yes — through primitive parameters               | not directly                  | bit-equivalent to `evaluate_stump_volume` on the DNF path |
| Manifold (manifold3d)              | indirect (mesh only) | mesh-level                     | no (mesh)                | no (Boolean is exact, not differentiable)        | no                            | hypothesis: robust mesh boolean, not a CAD-differentiable kernel |
| PyTorch3D mesh ops                 | no (mesh only)       | mesh-level                     | no (mesh)                | partial (mesh deformations are differentiable)   | no                            | hypothesis: differentiable mesh operations, not B-Rep |
| JAX-FEM                            | no (FE mesh)         | element-level                  | no (mesh)                | yes — but for FE objectives, not CAD volume      | no                            | hypothesis: physics solver, not a CAD geometry kernel |

The framing BRepAX is built around — *gradients flow from STEP /
B-Rep through to a JAX scalar* — does not have an obvious head-to-head
counterpart in the listed systems, because they enter the pipeline at
a different stage (mesh, or FE mesh).  This is the BRepAX position to
test, not assume; the right way to falsify it is to run a published
benchmark that asks each system to solve the same B-Rep-rooted
optimisation problem.  No such benchmark is bundled with this PR.

What is *not* claimed by this report:

- No "first" or "unique" claim about any BRepAX path.  Other research
  projects exist on differentiable B-Rep, signed distance functions on
  trimmed surfaces, and topology-aware boolean differentiation; this
  report does not survey them.
- No quantitative speed comparison against any external system.
- No claim that BRepAX outperforms any competitor on tasks they were
  designed for.  Manifold's exact mesh boolean and JAX-FEM's PDE
  solver are *not* in scope here.

What this report *does* assert:

- The numbers in `volume accuracy` and `face-level metric coverage`
  are reproducible from the bundled command on the bundled fixtures.
- The CSG-Stump and TrimmedCSGStump rows being equal is consistent
  with the negative-result ADRs (ADR-0019, ADR-0020).
- Per-face metric NaN counts on cone / torus / BSpline faces in
  `mean_curvature_per_face` are the documented analytical-handler
  deferral, not failures.

---

Generated by `benchmarks/integration_report/run_benchmark.py`.
