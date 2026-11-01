"""Compare one-parameter contact-boundary edits at matched actual CAD accuracy.

Run ``uv run --group benchmark python -m benchmarks.multi_contact.event_boundary
--output /tmp/event-boundary.json``. These are explicit synthetic convex cells,
not a general CAD optimizer or evidence for arbitrary STEP input.
"""

import argparse
from pathlib import Path

from .event_boundary_evaluation import METHODS, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--method", choices=METHODS, action="append")
    parser.add_argument("--mode", choices=("cold", "sequential"), action="append")
    parser.add_argument("--theta-index", type=int)
    parser.add_argument("--skip-audits", action="store_true")
    args = parser.parse_args()
    result = run(
        args.output,
        names=args.family,
        methods=tuple(args.method or METHODS),
        modes=tuple(args.mode or ("cold", "sequential")),
        theta_index=args.theta_index,
        audits=not args.skip_audits,
    )
    if not result["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
