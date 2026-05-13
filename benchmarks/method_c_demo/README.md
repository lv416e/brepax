# Method-C applied convergence demo

A one-command reproducible comparison of the stratum-aware Boolean
volume gradient (Method C) against the temperature-smoothed gradient
(Method A) on a disk-disk fixture with closed-form ground truth.

Exposes the within-stratum result from
[`tests/benchmarks/test_optimization_trajectory.py`](../../tests/benchmarks/test_optimization_trajectory.py)
as a docs-site artifact.  No new theory, no new metrics, no
production-code changes.

## Run

From the repository root:

```bash
uv sync --extra viz
uv run python -m benchmarks.method_c_demo.run
```

Regenerates [`REPORT.md`](REPORT.md) and `convergence.png` in place.

Iteration flags:

```bash
uv run python -m benchmarks.method_c_demo.run --max-steps 50 --lr 0.05
```

## What it shows

- Both methods optimise the same loss `(union_area(a, b) - target)^2`
  on the same fixture with the same `lr=0.01`, `max_steps=200`.
- Method C reaches the grid-discretisation floor of its
  stratum-aware integrator.  Method A's loss looks lower because it
  is the loss of a *bias-shifted* objective; its position error
  reveals the bias.
- The convergence plot makes the asymmetry visible: low loss does
  not imply the right answer when the objective is smoothed.
