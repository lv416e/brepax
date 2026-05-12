# Method-C applied convergence demo

A docs-side mirror of the within-stratum optimisation result that
already lives in
[`tests/benchmarks/test_optimization_trajectory.py`](https://github.com/lv416e/brepax/blob/main/tests/benchmarks/test_optimization_trajectory.py).
The numbers and plot on this page are the bundled benchmark output —
see
[`benchmarks/method_c_demo/README.md`](https://github.com/lv416e/brepax/blob/main/benchmarks/method_c_demo/README.md)
for the regeneration command.

## What this page does not claim

- This is **not a new result**.  The within-stratum precision gap
  between Method C and Method A is documented in ADR-0009 / ADR-0011
  and exercised by the test cited above.  This page just surfaces it
  alongside the rest of the documentation.
- No claim that Method A is "wrong".  The residual it leaves is a
  documented bias property of the sigmoid-temperature smoothed
  objective at finite `beta`.

## The report

--8<-- "benchmarks/method_c_demo/REPORT.md"
