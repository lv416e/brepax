"""Method-C applied convergence demo.

Surfaces the within-stratum optimisation result from
``tests/benchmarks/test_optimization_trajectory.py`` as a single
reproducible command that writes a markdown report and a matplotlib
convergence plot side by side.

The fixture is identical to that test (two disks, ``C1=(0, 0)``,
``R1`` optimised, ``C2=(1, 0)``, ``R2=1``, analytical ground truth at
``r1=1.5``) so the headline finding here and the test's regression
gate stay in sync.

Run:
    uv run python -m benchmarks.method_c_demo.run

Output (overwritten in place):
    benchmarks/method_c_demo/REPORT.md
    benchmarks/method_c_demo/convergence.png

Scope:
    No new algorithm.  No new metric.  No changes under
    ``src/brepax/``.  The script orchestrates the existing public
    ``union_area(..., method=...)`` API and renders the per-step
    trajectory both methods produce on the same optimisation problem.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

from brepax.analytical.disk_disk import disk_disk_union_area
from brepax.boolean import union_area
from brepax.primitives import Disk

REPORT_PATH = Path(__file__).resolve().parent / "REPORT.md"
PLOT_PATH = Path(__file__).resolve().parent / "convergence.png"
# Mirror the PNG into docs/reference/ so the transclusion in
# ``docs/reference/method_c_demo.md`` resolves its relative
# ``convergence.png`` reference against the docs output tree.
# Without this copy the in-tree REPORT.md renders correctly but the
# mkdocs page reports a missing asset.
DOCS_PLOT_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "reference" / "convergence.png"
)

# Fixture — held identical to
# ``tests/benchmarks/test_optimization_trajectory.py:49-101`` so the
# demo and the test cannot drift.
C1 = jnp.array([0.0, 0.0])
C2 = jnp.array([1.0, 0.0])
R2 = jnp.array(1.0)
TARGET_R1 = jnp.array(1.5)
INIT_R1 = jnp.array(0.8)

# Smoothing operating point — single, fixed.  Sweeping ``beta`` is a
# separate question and explicitly out of scope for this PR.
SMOOTHING_K = 0.1
SMOOTHING_BETA = 0.1
SMOOTHING_RESOLUTION = 128

DEFAULT_MAX_STEPS = 200
DEFAULT_LR = 0.01


@dataclass
class Trajectory:
    method: str
    final_r1: float
    final_loss: float
    final_position_err: float
    steps: int
    r1_history: list[float]
    loss_history: list[float]
    position_err_history: list[float]


def _loss_factory(
    method: str, target_area: float
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    def loss(r1: jnp.ndarray) -> jnp.ndarray:
        a = Disk(center=C1, radius=r1)
        b = Disk(center=C2, radius=R2)
        if method == "stratum":
            area = union_area(a, b, method="stratum")
        elif method == "smoothing":
            area = union_area(
                a,
                b,
                method="smoothing",
                k=SMOOTHING_K,
                beta=SMOOTHING_BETA,
                resolution=SMOOTHING_RESOLUTION,
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        return (area - target_area) ** 2

    return loss


def _optimize(
    method: str,
    target_area: float,
    *,
    init_r1: jnp.ndarray,
    lr: float,
    max_steps: int,
) -> Trajectory:
    loss_fn = _loss_factory(method, target_area)
    grad_fn = jax.grad(loss_fn)

    r1 = init_r1
    r1_history = [float(r1)]
    loss_history = [float(loss_fn(r1))]
    position_err_history = [float(jnp.abs(r1 - TARGET_R1))]

    # The test in tests/benchmarks/test_optimization_trajectory.py
    # short-circuits on ``loss < 1e-8`` which Method A hits in ~6 steps
    # at its bias-shifted minimum.  For the demo plot we run both
    # methods the full ``max_steps`` so Method A's biased plateau
    # (the headline finding) is visible as a flat line, not a
    # truncated trajectory.
    for _ in range(max_steps):
        g = grad_fn(r1)
        r1 = r1 - lr * g
        r1_history.append(float(r1))
        loss = float(loss_fn(r1))
        loss_history.append(loss)
        position_err_history.append(float(jnp.abs(r1 - TARGET_R1)))

    return Trajectory(
        method=method,
        final_r1=float(r1),
        final_loss=loss_history[-1],
        final_position_err=position_err_history[-1],
        steps=len(loss_history) - 1,
        r1_history=r1_history,
        loss_history=loss_history,
        position_err_history=position_err_history,
    )


def _render_plot(traj_c: Trajectory, traj_a: Trajectory, out: Path) -> None:
    """Render the two-panel convergence figure.

    Import matplotlib lazily so callers that only need the numbers can
    use the rest of this module without the ``viz`` extra installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dep
        raise ImportError(
            "matplotlib is required to render the convergence plot; "
            "install the viz extra via `uv sync --extra viz`"
        ) from exc

    fig, (ax_loss, ax_pos) = plt.subplots(1, 2, figsize=(10, 4))

    for traj, label, colour in (
        (traj_c, "Method C (stratum)", "tab:blue"),
        (traj_a, "Method A (smoothing, beta=0.1)", "tab:orange"),
    ):
        steps = range(len(traj.loss_history))
        ax_loss.semilogy(steps, traj.loss_history, label=label, color=colour)
        ax_pos.semilogy(steps, traj.position_err_history, label=label, color=colour)

    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss  (area - target)^2")
    ax_loss.set_title("Loss vs step")
    ax_loss.grid(True, which="both", ls=":", alpha=0.4)
    ax_loss.legend()

    ax_pos.set_xlabel("step")
    ax_pos.set_ylabel("|r1 - r1*|")
    ax_pos.set_title("Position error vs step")
    ax_pos.grid(True, which="both", ls=":", alpha=0.4)
    ax_pos.legend()

    fig.suptitle("Method-C vs Method-A within the intersecting stratum")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _render_report(
    traj_c: Trajectory,
    traj_a: Trajectory,
    target_area: float,
    *,
    lr: float,
    max_steps: int,
) -> str:
    lines: list[str] = []
    lines.append("# Method-C applied convergence demo")
    lines.append("")
    lines.append(
        "Two disks, ``C1 = (0, 0)``, ``R1`` optimised toward the "
        "analytical ground truth ``r1* = 1.5``, ``C2 = (1, 0)``, "
        "``R2 = 1``.  The objective is "
        "``loss(r1) = (union_area(a, b) - target)^2`` with the target "
        "computed in closed form by "
        "``brepax.analytical.disk_disk.disk_disk_union_area``.  The same "
        "gradient-descent loop runs twice, once with "
        '``method="stratum"`` (Method C) and once with '
        '``method="smoothing", k=0.1, beta=0.1, resolution=128`` '
        "(Method A).  The numbers below are the output of one command."
    )
    lines.append("")
    lines.append(
        f"Settings: ``lr={lr}``, ``max_steps={max_steps}``, "
        f"``init_r1=0.8``, ``target_area={target_area:.6f}`` (closed "
        "form)."
    )
    lines.append("")
    lines.append("## Final-step numbers")
    lines.append("")
    lines.append("| Method | final r1 | |r1 - r1*| | final loss | steps |")
    lines.append("|---|---|---|---|---|")
    for traj, label in (
        (traj_c, "Method C (stratum)"),
        (traj_a, "Method A (smoothing, beta=0.1)"),
    ):
        lines.append(
            f"| {label} "
            f"| {traj.final_r1:.6f} "
            f"| {traj.final_position_err:.2e} "
            f"| {traj.final_loss:.2e} "
            f"| {traj.steps} |"
        )
    lines.append("")
    lines.append(
        "Method C converges to the grid-discretisation floor of its "
        "stratum-aware integrator.  Method A's residual is the bias "
        "introduced by the sigmoid temperature ``beta``; it does not "
        "shrink with more gradient-descent steps because the bias is "
        "inherent to the smoothed objective, not a transient of the "
        "optimiser.  This is the documented behaviour from "
        "``tests/benchmarks/test_optimization_trajectory.py``; the "
        "purpose of this page is to make it visible alongside the rest "
        "of the documentation rather than to claim it as a new result."
    )
    lines.append("")
    lines.append("## Convergence plot")
    lines.append("")
    lines.append("![convergence](convergence.png)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("Regenerate with `uv run python -m benchmarks.method_c_demo.run`.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Gradient descent step cap (default 200, matches the test)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Gradient descent learning rate (default 0.01)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory for REPORT.md and convergence.png",
    )
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "REPORT.md"
    plot_path = out_dir / "convergence.png"

    target_area = float(disk_disk_union_area(C1, TARGET_R1, C2, R2))

    print("[run]  Method C (stratum)")
    traj_c = _optimize(
        "stratum",
        target_area,
        init_r1=INIT_R1,
        lr=args.lr,
        max_steps=args.max_steps,
    )
    print(
        f"       final r1={traj_c.final_r1:.6f}, "
        f"final loss={traj_c.final_loss:.2e}, "
        f"steps={traj_c.steps}"
    )

    print("[run]  Method A (smoothing, beta=0.1)")
    traj_a = _optimize(
        "smoothing",
        target_area,
        init_r1=INIT_R1,
        lr=args.lr,
        max_steps=args.max_steps,
    )
    print(
        f"       final r1={traj_a.final_r1:.6f}, "
        f"final loss={traj_a.final_loss:.2e}, "
        f"steps={traj_a.steps}"
    )

    _render_plot(traj_c, traj_a, plot_path)
    print(f"\nWrote {plot_path}")
    if args.out == Path(__file__).resolve().parent:
        # Default invocation also mirrors the PNG into docs/reference/
        # so the mkdocs transclusion of REPORT.md resolves the relative
        # ``convergence.png`` against the docs output tree.  Non-default
        # ``--out`` is intended for local iteration and skips the mirror.
        DOCS_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        DOCS_PLOT_PATH.write_bytes(plot_path.read_bytes())
        print(f"Wrote {DOCS_PLOT_PATH}")
    report_path.write_text(
        _render_report(
            traj_c,
            traj_a,
            target_area,
            lr=args.lr,
            max_steps=args.max_steps,
        )
    )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
