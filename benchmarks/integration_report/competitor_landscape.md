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
