"""Measure contact-volume gradients and a radius-driven CAD breakout.

Run from the repository root:
    uv run --group benchmark python -m benchmarks.contact_transition.run \
        --output /tmp/contact.json

Artifacts retain every method, temperature, failure, finite-difference step,
and optimizer trajectory. No report is written into the repository by default.
Timings synchronize JAX and separate compilation from warmed execution. These
bounded fixtures do not establish general superiority or arbitrary CAD support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import jax

from .experiment import run


def provenance() -> dict[str, object]:
    """Record both the checkout commit and source bytes for uncommitted trials."""
    root = Path(__file__).resolve().parents[2]
    source = root / "src/brepax/experimental/contact.py"
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "candidate_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "benchmark_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-resolutions", type=int, nargs="+", default=[24, 48])
    parser.add_argument(
        "--plate-resolutions", type=int, nargs="+", default=[64, 128, 256]
    )
    parser.add_argument(
        "--suites",
        choices=("contact", "plate", "breakout"),
        nargs="+",
        default=["contact", "plate", "breakout"],
    )
    parser.add_argument("--optimization-steps", type=int, default=100)
    parser.add_argument(
        "--cad",
        action="store_true",
        help="Rebuild, transform and remesh independent OCCT solids",
    )
    args = parser.parse_args()
    if (
        min(*args.contact_resolutions, *args.plate_resolutions) < 2
        or args.optimization_steps < 1
    ):
        parser.error("Resolutions >= 2 and positive optimization steps are required")
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    report = run(
        tuple(args.contact_resolutions),
        tuple(args.plate_resolutions),
        tuple(args.suites),
        args.optimization_steps,
        include_cad=args.cad,
    )
    report["provenance"] = provenance()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "execution_success": report["execution_success"],
                "candidate_accuracy": report["candidate_accuracy"]["passed"],
                "cad_validation_passed": report["cad_validation_passed"],
                "output": str(args.output),
            }
        )
    )
    if (
        not report["execution_success"]
        or report["candidate_accuracy"]["passed"] is False
        or report["cad_validation_passed"] is False
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
