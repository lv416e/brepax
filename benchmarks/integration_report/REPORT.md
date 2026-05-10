# BRepAX Integration Benchmark Report

Reproducible measurement of the BRepAX volume paths and face-level metric coverage on the project's standard STEP fixture set.

**This report is the output of one benchmark command** — `uv run python -m benchmarks.integration_report.run_benchmark`. No claim of novelty is made beyond what the tables below directly show.  Comparisons against external systems (Manifold, PyTorch3D, JAX-FEM) are qualitative; see `competitor_landscape.md` for the framing and its limitations.

Volume paths use a sigmoid grid integration at `resolution=32` (per-axis cell count); the mesh divergence path uses the OCCT BRepMesh tessellation at the default deflection.  OCCT BRepGProp is the reference and is computed analytically on the exact B-Rep, not on the mesh.

## Volume accuracy

| Fixture | OCCT (ref) | divergence | mesh_sdf | gwn | CSG-Stump | TrimmedCSG | timings (s) | Notes |
|---|---|---|---|---|---|---|---|---|
| `sample_box` | 6000.0000 | 6000.0000 (0.00%) | 5689.0195 (5.18%) | 5900.4824 (1.66%) | 5689.5962 (5.17%) | 5689.5962 (5.17%) | divergence=1.18 mesh_sdf=4.30 gwn=0.92 csg_stump=4.19 trimmed_csg=2.67 |  |
| `sample_cylinder` | 1178.0972 | 1176.5602 (0.13%) | 1184.0016 (0.50%) | 1164.7401 (1.13%) | 1186.0311 (0.67%) | 1186.0311 (0.67%) | divergence=1.16 mesh_sdf=2.70 gwn=1.22 csg_stump=1.77 trimmed_csg=2.16 |  |
| `sample_sphere` | 113.0973 | 112.5589 (0.48%) | 117.7642 (4.13%) | 111.8235 (1.13%) | 118.4262 (4.71%) | 118.4262 (4.71%) | divergence=1.52 mesh_sdf=15.60 gwn=6.31 csg_stump=0.62 trimmed_csg=0.66 |  |
| `sample_cone` | 54.9779 | 54.6540 (0.59%) | 60.3504 (9.77%) | 56.6688 (3.08%) | 60.8456 (10.67%) | 60.8456 (10.67%) | divergence=3.44 mesh_sdf=4.04 gwn=2.04 csg_stump=2.58 trimmed_csg=2.46 |  |
| `sample_torus` | 222.0661 | 220.6733 (0.63%) | 241.2626 (8.64%) | 221.0463 (0.46%) | 242.5633 (9.23%) | 242.5633 (9.23%) | divergence=1.83 mesh_sdf=23.25 gwn=8.83 csg_stump=0.05 trimmed_csg=0.31 |  |
| `box_with_holes` | 22429.2037 | 22432.0938 (0.01%) | 20694.4512 (7.73%) | 21917.6699 (2.28%) | 20692.2324 (7.74%) | 20692.2324 (7.74%) | divergence=1.63 mesh_sdf=3.73 gwn=1.70 csg_stump=1.27 trimmed_csg=1.25 |  |
| `box_with_pocket` | 23214.6018 | 23215.6230 (0.00%) | 21492.3789 (7.42%) | 22700.0762 (2.22%) | 20293.1797 (12.58%) | 20293.1797 (12.58%) | divergence=1.46 mesh_sdf=6.72 gwn=1.29 csg_stump=0.86 trimmed_csg=1.10 |  |
| `box_with_slot` | 20800.0000 | 20800.0000 (0.00%) | 19218.2852 (7.60%) | 20331.0352 (2.25%) | 17470.7598 (16.01%) | 17470.7598 (16.01%) | divergence=1.49 mesh_sdf=2.72 gwn=0.58 csg_stump=4.31 trimmed_csg=2.99 |  |
| `l_bracket` | 12000.0000 | 12000.0000 (0.00%) | 11267.1504 (6.11%) | 11729.4434 (2.25%) | 11025.0674 (8.12%) | 11025.0674 (8.12%) | divergence=0.64 mesh_sdf=1.69 gwn=0.57 csg_stump=1.78 trimmed_csg=1.91 |  |
| `nurbs_box` | 480.0000 | 480.0000 (0.00%) | 481.2567 (0.26%) | 497.4170 (3.63%) | 481.2568 (0.26%) | 481.2568 (0.26%) | divergence=0.66 mesh_sdf=1.01 gwn=0.49 csg_stump=55.30 trimmed_csg=44.21 |  |
| `nurbs_revol` | 40.8856 | — | — | — | — | — | divergence=0.00 mesh_sdf=0.00 gwn=0.00 csg_stump=0.00 trimmed_csg=0.00 | shell (volume paths require closed solid) |
| `nurbs_saddle` | -0.0000 | — | — | — | — | — | divergence=0.00 mesh_sdf=0.00 gwn=0.00 csg_stump=0.00 trimmed_csg=0.00 | shell (volume paths require closed solid) |

**How to read this table.**

- `divergence` is the mesh divergence-theorem volume (Stokes' theorem on the BRepMesh tessellation).  Differentiable through triangle vertex positions.  Strongest production path.
- `mesh_sdf` is the mesh-based signed distance field (unsigned distance to the nearest triangle, signed by generalized winding number) integrated with a sigmoid indicator on the same grid as the CSG paths.  Differentiable through triangle vertex positions and the sigmoid integrator.
- `gwn` is the generalized-winding-number indicator integrated directly: ``sigmoid((wn - 0.5) * GWN_SHARPNESS)`` summed over the grid.  Sharpness is fixed at module level (the field is dimensionless 0..1, so the cell-width-based sharpness used by the SDF integrators is not appropriate here).  This single fixed configuration is one operating point; sweeping sharpness / resolution / mesh deflection is out of scope for this PR.
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
