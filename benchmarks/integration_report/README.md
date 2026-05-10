# Integration benchmark report

A reproducible measurement of BRepAX's volume paths and face-level
metric coverage on the project's standard STEP fixture set, plus a
qualitative landscape table against external systems (Manifold,
PyTorch3D, JAX-FEM).

## Run

From the repository root:

```bash
uv run python -m benchmarks.integration_report.run_benchmark
```

This regenerates [`REPORT.md`](REPORT.md) in place.  The default run
covers all twelve fixtures under `tests/fixtures/*.step` at grid
resolution 32 for the sigmoid volume integrators.

To iterate on a smaller subset:

```bash
uv run python -m benchmarks.integration_report.run_benchmark \
    --fixtures sample_box sample_cylinder \
    --skip-csg
```

## What it measures

- **Volume accuracy** — three BRepAX paths versus OCCT BRepGProp's
  analytic surface-integral quadrature.
- **Per-face metric coverage** — finite / NaN / inf counts for the
  four metrics shipped in PR #81-#84.

## What it does not measure

- No quantitative speed comparison against external systems.  The
  competitor landscape table in [`competitor_landscape.md`](competitor_landscape.md)
  is qualitative feature coverage, with explicit caveats.
- No "first" / "unique" claims about any BRepAX path.
